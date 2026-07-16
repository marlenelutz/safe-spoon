import csv
import io

from flask import Blueprint, Response, jsonify, request

from safe_spoon.webapp.state import get_state

bp = Blueprint("labels", __name__)


@bp.route("/api/labels", methods=["GET"])
def get_labels():
    return jsonify(get_state().labels)


@bp.route("/api/labels/export")
def export_labels():
    state = get_state()
    buf = io.StringIO()
    state.write_labels_csv(csv.writer(buf))
    buf.seek(0)
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=labels.csv"},
    )


@bp.route("/api/labels", methods=["POST"])
def post_label():
    state = get_state()
    body = request.json
    cat = body["cat"]
    gi = int(body["gi"])
    key = f"{cat}:{gi}"
    state.labels[key] = {lf["field"]: bool(
        body.get(lf["field"], False)) for lf in state.query_labels}
    state.save_labels()
    return jsonify({"ok": True})
