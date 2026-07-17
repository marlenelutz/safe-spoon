import numpy as np
from flask import Blueprint, jsonify

from safe_spoon.annotation.priority import assign_priorities
from safe_spoon.clustering import AnnotationUnitModel
from safe_spoon.webapp.state import get_state

bp = Blueprint("annotation_units", __name__)


def _flatten_unit_tree(tree, aum=None):
    """Convert a nested unit-tree dict to {nodes, root_id} to avoid deep recursion in json.dumps.

    When aum is given, each unit node's "label" is overridden with the
    LLM-generated label and an "note" field is added with the LLM's annotation guidance, if any.
    """
    if tree is None:
        return None, None
    nodes = []
    stack = [tree]
    while stack:
        node = stack.pop()
        children = node.get("children", [])
        flat = {k: v for k, v in node.items() if k != "children"}
        flat["children_ids"] = [c["node_id"] for c in children]
        if aum is not None and flat.get("is_unit"):
            flat["label"] = aum.get_unit_label(flat["node_id"])
            flat["note"] = aum.get_unit_note(flat["node_id"])
        nodes.append(flat)
        stack.extend(children)
    return nodes, tree["node_id"]


@bp.route("/api/annotation_units/<path:cat>")
def api_annotation_units(cat):
    state = get_state()
    cfg = state.cfg

    cached = state.unit_cache.get(cat)
    if cached is None:
        cached = _build_unit_tree_payload(state, cat)
        if cached is None:
            return jsonify({
                "unit_tree": None,
                "n_units": 0,
                "topic_keys": [],
                "topic_labels": [],
                "config": {},
            })
        state.unit_cache[cat] = cached

    # Priority is recomputed on every request
    unit_nodes = cached["unit_tree"]["nodes"] if cached["unit_tree"] else []
    unit_list = [n for n in unit_nodes if n.get("is_unit")]
    assign_priorities(
        unit_list,
        pw_heterogeneity=cfg["pw_heterogeneity"],
        pw_size=cfg["pw_size"],
        pw_balance=cfg["pw_balance"],
    )

    payload = {k: v for k, v in cached.items() if not k.startswith("_")}
    return jsonify(payload)


def _build_unit_tree_payload(state, cat):
    """Build (once) and cache the clustering-derived unit tree for a category.
    Does NOT set priority/priority_rank; those are assigned fresh on every
    request by the caller.
    """
    cfg = state.cfg
    d = state.get_data()
    info = d["data_by_category"].get(cat, {})
    tree_info = d["trees_by_category"].get(cat, {})
    thetas_raw = info.get("thetas") or []
    if not thetas_raw or not tree_info:
        return None

    thetas_all = np.array(thetas_raw, dtype=np.float32)
    tree_indices = tree_info.get("indices", list(range(len(thetas_all))))
    thetas_for_tree = thetas_all[tree_indices]

    model_path = info.get("model_info", {}).get("model_path")

    # Reuse the exact thresholds the pipeline built this tree with.
    saved_params = AnnotationUnitModel.load_saved_params(model_path)
    if saved_params is not None:
        min_size = int(saved_params["min_size"])
        max_rel_dist = float(saved_params["max_rel_dist"])
    else:
        min_size = cfg["min_size"]
        max_rel_dist = float(cfg["max_rel_dist"])

    aum = AnnotationUnitModel(
        flat_nodes=tree_info["nodes"],
        root_id=tree_info["root_id"],
        thetas=thetas_all,
        topic_keys=info.get("topic_keys", []),
        topic_labels=info.get("topic_labels", []),
        queries=info.get("queries", []),
        model_path=model_path,
        min_size=min_size,
        max_rel_dist=max_rel_dist,
    )
    aum.build()
    aum.load_unit_labels()
    unit_nodes, unit_root_id = _flatten_unit_tree(aum.unit_tree, aum)

    return {
        "unit_tree": {"nodes": unit_nodes, "root_id": str(unit_root_id)} if unit_nodes is not None else None,
        "n_units": aum.n_units,
        "topic_keys": info.get("topic_keys", []),
        "topic_labels": info.get("topic_labels", []),
        "config": {
            "min_size": min_size,
            "max_rel_dist": max_rel_dist,
        },
        # internal only, stripped before jsonify
        "_queries": info.get("queries", []),
    }
