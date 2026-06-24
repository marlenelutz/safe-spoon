import io

from flask import Flask, Response, jsonify, request, send_file
from pathlib import Path
import json, csv
import numpy as np

from safe_spoon.clustering import AnnotationUnitModel
from safe_spoon.utils.common import load_annotation_unit_config

_au_cfg = load_annotation_unit_config()
MIN_SIZE = _au_cfg["min_size"]
MAX_PURITY = _au_cfg["max_purity"]
PW_MIXTURE = _au_cfg["pw_mixture"]
PW_SIZE = _au_cfg["pw_size"]
PW_BALANCE = _au_cfg["pw_balance"]

app = Flask(__name__, static_folder="static")

DATA_FILE = "data/output/viz_v5_data.json"
LABELS_FILE = "data/output/labels.csv"

_data   = None
_labels = {}


def get_data():
    global _data
    if _data is None:
        print("Loading data...", flush=True)
        _data = json.load(open(DATA_FILE, encoding="utf-8"))
        print("Data loaded.", flush=True)
    return _data


def _write_labels_csv(writer, labels_dict, data_by_category):
    writer.writerow(["query_id", "category", "query_index", "query",
                     "TRASH_PRESENT", "DEMOGRAPHICS_DESCRIBED"])
    for key, state in labels_dict.items():
        cat, gi_str = key.rsplit(":", 1)
        gi   = int(gi_str)
        info = data_by_category.get(cat, {})
        ids  = info.get("query_ids") or []
        qs   = info.get("queries")   or []
        qid  = ids[gi] if gi < len(ids) else gi
        text = qs[gi]  if gi < len(qs)  else ""
        writer.writerow([qid, cat, gi, text, state["trash"], state["demo"]])


def load_labels():
    global _labels
    _labels = {}
    p = Path(LABELS_FILE)
    if not p.exists():
        return
    with p.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = f"{row['category']}:{row['query_index']}"
            _labels[key] = {
                "trash": row["TRASH_PRESENT"] == "True",
                "demo":  row["DEMOGRAPHICS_DESCRIBED"] == "True",
            }


def save_labels():
    d = get_data()
    p = Path(LABELS_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        _write_labels_csv(csv.writer(f), _labels, d["data_by_category"])


load_labels()


@app.route("/")
def index():
    return send_file("static/viz_v5_template.html")


@app.route("/api/init")
def api_init():
    d = get_data()
    return jsonify({"categories": d["categories"], "n_repr": d["n_repr"]})


@app.route("/api/category/<path:cat>")
def api_category(cat):
    d = get_data()
    info = d["data_by_category"].get(cat, {})
    return jsonify({
        "queries": info.get("queries",[]),
        "query_ids": info.get("query_ids",[]),
        "topic_keys": info.get("topic_keys",   []),
        "topic_labels": info.get("topic_labels", []),
        "alphas": info.get("alphas", []),
        "tpc_coords": info.get("tpc_coords",   []),
        "top_docs": info.get("top_docs",[]),
        "n":info.get("n", 0),
    })


@app.route("/api/tree/<path:cat>")
def api_tree(cat):
    d = get_data()
    return jsonify(d["trees_by_category"].get(cat, {}))


@app.route("/api/topic_dist", methods=["POST"])
def api_topic_dist():
    body = request.json
    cat = body["cat"]
    indices = body["indices"]
    d = get_data()
    thetas = d["data_by_category"].get(cat, {}).get("thetas") or []
    if not thetas or not indices:
        return jsonify({"mean": []})
    arr  = np.array([thetas[i] for i in indices if i < len(thetas)], dtype=np.float32)
    mean = arr.mean(axis=0).round(4).tolist() if len(arr) else []
    return jsonify({"mean": mean})


@app.route("/api/labels", methods=["GET"])
def get_labels():
    return jsonify(_labels)


@app.route("/api/labels/export")
def export_labels():
    d   = get_data()
    buf = io.StringIO()
    _write_labels_csv(csv.writer(buf), _labels, d["data_by_category"])
    buf.seek(0)
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=labels.csv"},
    )


@app.route("/api/labels", methods=["POST"])
def post_label():
    body = request.json
    cat = body["cat"]
    gi = int(body["gi"])
    key = f"{cat}:{gi}"
    _labels[key] = {"trash": bool(body["trash"]), "demo": bool(body["demo"])}
    save_labels()
    return jsonify({"ok": True})


@app.route("/api/annotation_units/<path:cat>")
def api_annotation_units(cat):
    d = get_data()
    info      = d["data_by_category"].get(cat, {})
    tree_info = d["trees_by_category"].get(cat, {})
    thetas = info.get("thetas") or []
    if not thetas or not tree_info:
        return jsonify(
            {"unit_tree": None, 
             "n_units": 0,
            "topic_keys": [], 
            "topic_labels": [], 
            "config": {}
            }
        )
    aum = AnnotationUnitModel(
        flat_nodes = tree_info["nodes"],
        root_id = tree_info["root_id"],
        thetas = np.array(thetas, dtype=np.float32),
        topic_keys = info.get("topic_keys",[]),
        topic_labels = info.get("topic_labels", []),
        queries = info.get("queries",[]),
        min_size = MIN_SIZE,
        max_purity = MAX_PURITY,
        pw_mixture = PW_MIXTURE,
        pw_size = PW_SIZE,
        pw_balance = PW_BALANCE,
    )
    aum.build()
    return jsonify({
        "unit_tree": aum.unit_tree,
        "n_units": aum.n_units,
        "topic_keys": info.get("topic_keys",   []),
        "topic_labels": info.get("topic_labels", []),
        "config": {"min_size": MIN_SIZE, "max_purity": MAX_PURITY},
    })


if __name__ == "__main__":
    app.run(debug=False, port=5000)
