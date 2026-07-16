from pathlib import Path

import numpy as np
from flask import Blueprint, jsonify, request, send_file

from safe_spoon.webapp.state import get_state

bp = Blueprint("core", __name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_STATIC_DIR = _REPO_ROOT / "static"


@bp.route("/")
def index():
    return send_file(_STATIC_DIR / "front.html")


@bp.route("/support.js")
def serve_support():
    return send_file(_STATIC_DIR / "support.js")


@bp.route("/jelly_logo4.png")
def serve_logo():
    return send_file(_STATIC_DIR / "jelly_logo4.png")


@bp.route("/js/ui_helpers.js")
def serve_ui_helpers():
    return send_file(_STATIC_DIR / "js" / "ui_helpers.js")


@bp.route("/api/init")
def api_init():
    state = get_state()
    d = state.get_data()
    return jsonify({
        "categories": d["categories"],
        "n_repr": d["n_repr"],
        "query_labels": state.query_labels,
        "ui_display": state.ui_display,
    })


@bp.route("/api/category/<path:cat>")
def api_category(cat):
    d = get_state().get_data()
    info = d["data_by_category"].get(cat, {})
    return jsonify({
        "queries": info.get("queries", []),
        "query_ids": info.get("query_ids", []),
        "topic_keys": info.get("topic_keys", []),
        "topic_labels": info.get("topic_labels", []),
        "alphas": info.get("alphas", []),
        "tpc_coords": info.get("tpc_coords", []),
        "top_docs": info.get("top_docs", []),
        "thetas": info.get("thetas", []),
        "n": info.get("n", 0),
    })


@bp.route("/api/tree/<path:cat>")
def api_tree(cat):
    d = get_state().get_data()
    return jsonify(d["trees_by_category"].get(cat, {}))


@bp.route("/api/topic_dist", methods=["POST"])
def api_topic_dist():
    body = request.json
    cat = body["cat"]
    indices = body["indices"]
    d = get_state().get_data()
    thetas = d["data_by_category"].get(cat, {}).get("thetas") or []
    if not thetas or not indices:
        return jsonify({"mean": []})
    arr = np.array([thetas[i]
                   for i in indices if i < len(thetas)], dtype=np.float32)
    mean = arr.mean(axis=0).round(4).tolist() if len(arr) else []
    return jsonify({"mean": mean})
