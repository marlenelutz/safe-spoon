import logging

from flask import Blueprint, jsonify, request

from safe_spoon.annotation import risk_profiles, rubrics, suggestions
from safe_spoon.annotation.risk_profiles import RiskProfileInUseError
from safe_spoon.webapp.annotation_units_routes import _build_unit_tree_payload
from safe_spoon.webapp.state import get_state

log = logging.getLogger(__name__)

bp = Blueprint("rubric", __name__)


def _profile_to_json(rp) -> dict:
    return {
        "id": rp.id,
        "category": rp.category,
        "name": rp.name,
        "description": rp.description,
        "severity_rank": rp.severity_rank,
    }


def _rubric_to_json(rubric) -> dict:
    if rubric is None:
        return None
    return {
        "id": rubric.id,
        "category": rubric.category,
        "unit_stable_id": rubric.unit_stable_id,
        "annotator": rubric.annotator,
        "status": rubric.status,
        "source": rubric.source,
        "criteria": [
            {
                "id": c.id,
                "order_index": c.order_index,
                "title": c.title,
                "description": c.description,
                "cells": {
                    cell.risk_profile_id: {
                        "id": cell.id,
                        "expected_behavior": cell.expected_behavior,
                        "risk_signals": cell.risk_signals,
                        "inherited_from_cell_id": cell.inherited_from_cell_id,
                        "is_override": cell.is_override,
                    }
                    for cell in c.cells
                },
            }
            for c in rubric.criteria
        ],
    }


# ---------------------------------------------------------------------------
# Risk profiles
# ---------------------------------------------------------------------------
@bp.route("/api/risk_profiles/<path:cat>", methods=["GET"])
def list_risk_profiles_route(cat):
    conn = get_state().annotation_conn()
    try:
        profiles = risk_profiles.list_risk_profiles(conn, cat)
        return jsonify([_profile_to_json(p) for p in profiles])
    finally:
        conn.close()


@bp.route("/api/risk_profiles/<path:cat>", methods=["POST"])
def create_risk_profile_route(cat):
    body = request.get_json(force=True, silent=True) or {}
    conn = get_state().annotation_conn()
    try:
        rp = risk_profiles.create_risk_profile(
            conn, cat,
            name=body["name"],
            description=body.get("description", ""),
            severity_rank=int(body["severity_rank"]),
        )
        return jsonify(_profile_to_json(rp))
    finally:
        conn.close()


@bp.route("/api/risk_profiles/<path:cat>/<int:profile_id>", methods=["PUT"])
def update_risk_profile_route(cat, profile_id):
    body = request.get_json(force=True, silent=True) or {}
    conn = get_state().annotation_conn()
    try:
        rp = risk_profiles.update_risk_profile(conn, profile_id, **body)
        if rp is None:
            return jsonify({"error": "not found"}), 404
        return jsonify(_profile_to_json(rp))
    finally:
        conn.close()


@bp.route("/api/risk_profiles/<path:cat>/<int:profile_id>", methods=["DELETE"])
def delete_risk_profile_route(cat, profile_id):
    conn = get_state().annotation_conn()
    try:
        risk_profiles.delete_risk_profile(conn, profile_id)
        return jsonify({"ok": True})
    except RiskProfileInUseError as e:
        return jsonify({"error": str(e)}), 409
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Rubrics
# ---------------------------------------------------------------------------
def _get_unit_tree_cached(state, cat):
    cached = state.unit_cache.get(cat)
    if cached is None:
        cached = _build_unit_tree_payload(state, cat)
        if cached is not None:
            state.unit_cache[cat] = cached
    return cached


def _find_unit(unit_nodes, stable_id):
    for n in unit_nodes:
        if n.get("is_unit") and n.get("stable_id") == stable_id:
            return n
    return None


@bp.route("/api/units/<path:cat>/<stable_id>/rubric", methods=["GET"])
def get_unit_rubric(cat, stable_id):
    annotator = request.args.get("annotator")
    if not annotator:
        return jsonify({"error": "annotator query param is required"}), 400

    state = get_state()
    conn = state.annotation_conn()
    try:
        cached = _get_unit_tree_cached(state, cat)
        this_unit = None
        similar = []
        if cached and cached.get("unit_tree"):
            unit_nodes = cached["unit_tree"]["nodes"]
            this_unit = _find_unit(unit_nodes, stable_id)
            if this_unit is not None:
                unit_thetas = {
                    n["stable_id"]: n["mean_theta"]
                    for n in unit_nodes
                    if n.get("is_unit") and n.get("stable_id")
                }
                similar = rubrics.find_similar_confirmed_rubrics(
                    conn, cat, this_unit["mean_theta"], unit_thetas,
                )

        rubric = rubrics.get_or_create_draft(conn, cat, stable_id, annotator)

        return jsonify({
            "rubric": _rubric_to_json(rubric),
            "similar_confirmed": [
                {"rubric": _rubric_to_json(r), "similarity": round(sim, 4)}
                for r, sim in similar
            ],
        })
    finally:
        conn.close()


@bp.route("/api/units/<path:cat>/<stable_id>/rubric/suggest", methods=["POST"])
def suggest_unit_rubric(cat, stable_id):
    annotator = request.args.get("annotator") or (
        request.get_json(silent=True) or {}).get("annotator")
    if not annotator:
        return jsonify({"error": "annotator is required"}), 400

    state = get_state()
    cfg = state.cfg
    cached = _get_unit_tree_cached(state, cat)
    if cached is None or not cached.get("unit_tree"):
        return jsonify({"error": "category has no annotation units"}), 404

    unit_nodes = cached["unit_tree"]["nodes"]
    unit = _find_unit(unit_nodes, stable_id)
    if unit is None:
        return jsonify({"error": "unit not found"}), 404

    d = state.get_data()
    queries = d["data_by_category"].get(cat, {}).get("queries", [])
    topic_keys = cached.get("topic_keys", [])
    topic_labels = cached.get("topic_labels", [])
    dom_id = unit.get("dominant_topic", 0)
    dominant_topic = topic_labels[dom_id] if dom_id < len(
        topic_labels) else f"Topic {dom_id}"
    topic_keywords = ", ".join(
        topic_keys[dom_id][:10]) if dom_id < len(topic_keys) else ""
    repr_texts = [queries[i] for i in unit.get("repr", []) if i < len(queries)]

    body = request.get_json(silent=True) or {}
    n_candidates = int(body.get("n_candidates", 3))

    conn = state.annotation_conn()
    try:
        profiles = risk_profiles.list_risk_profiles(conn, cat)
        if not profiles:
            return jsonify({"error": "define risk profiles for this category before requesting suggestions"}), 400

        from safe_spoon.prompting import Prompter
        prompter = Prompter(
            model_type=cfg["rubric_model"],
            llm_provider=cfg["rubric_provider"],
            api_key=cfg["llm_api_key"],
            llm_server=cfg["rubric_server"],
        )

        candidates = suggestions.generate_rubric_candidates(
            prompter, cat, repr_texts, dominant_topic, topic_keywords,
            profiles, n_candidates=n_candidates,
        )

        now_rubric = rubrics.get_or_create_draft(conn, cat, stable_id, annotator)
        for idx, candidate in enumerate(candidates):
            import json as _json
            from datetime import datetime, timezone
            conn.execute(
                """INSERT INTO llm_rubric_suggestions
                   (category, unit_stable_id, annotator, candidate_index, criteria_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (cat, stable_id, annotator, idx, _json.dumps(
                    candidate), datetime.now(timezone.utc).isoformat()),
            )
        conn.commit()

        return jsonify({"candidates": candidates, "rubric_id": now_rubric.id})
    finally:
        conn.close()


@bp.route("/api/units/<path:cat>/<stable_id>/rubric", methods=["POST"])
def submit_unit_rubric(cat, stable_id):
    body = request.get_json(force=True, silent=True) or {}
    annotator = body.get("annotator")
    criteria = body.get("criteria")
    source = body.get("source", "manual")
    if not annotator or criteria is None:
        return jsonify({"error": "annotator and criteria are required"}), 400

    state = get_state()
    conn = state.annotation_conn()
    try:
        draft = rubrics.get_or_create_draft(conn, cat, stable_id, annotator)
        rubric = rubrics.submit_rubric(conn, draft.id, criteria, source=source)
        if body.get("confirm"):
            rubric = rubrics.confirm_rubric(conn, rubric.id)
        return jsonify(_rubric_to_json(rubric))
    finally:
        conn.close()
