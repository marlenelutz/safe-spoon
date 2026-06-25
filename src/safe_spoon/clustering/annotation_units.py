"""Annotation unit (i.e., group of queries for which we can generate a "guideline") selection from a hierarchical clustering dendrogram."""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def resolve_topic_label(
    id: int,
    topic_labels: Optional[List[str]],
    topic_keys: Optional[List[List[str]]],
    sep: str = " · ",
) -> str:
    """Return the best available label for topic id.

    Priority: LLM label (if not a placeholder) -> keyword join -> "Topic N".
    """
    if topic_labels and id < len(topic_labels) and not topic_labels[id].startswith("Topic "):
        return topic_labels[id]
    if topic_keys and id < len(topic_keys):
        return sep.join(topic_keys[id][:3])
    return f"Topic {id + 1}"


def compute_leaf_indices(nodes_by_id: dict, root_id: str) -> Dict[str, List[int]]:
    """Bottom-up pass: map each node id -> list of global leaf indices under it."""
    leaf_indices: Dict[str, List[int]] = {}
    stack = [(root_id, False)]
    while stack:
        nid, done = stack.pop()
        node = nodes_by_id[nid]
        if done:
            cids = node.get("children_ids", [])
            if not cids:
                leaf_indices[nid] = [node["id"]]
            else:
                merged: List[int] = []
                for cid in cids:
                    merged.extend(leaf_indices[cid])
                leaf_indices[nid] = merged
        else:
            stack.append((nid, True))
            for cid in node.get("children_ids", []):
                stack.append((cid, False))
    return leaf_indices


def make_unit_label(
    mean_theta: List[float],
    topic_labels: Optional[List[str]],
    topic_keys: Optional[List[List[str]]],
    n_top: int = 2,
) -> str:
    """Short human-readable label derived from the top-N dominant topics."""
    ranked = sorted(enumerate(mean_theta), key=lambda x: -x[1])
    parts = []
    for id, weight in ranked[:n_top]:
        if weight < 0.05:
            break
        lbl = resolve_topic_label(id, topic_labels, topic_keys)
        parts.append(lbl)
    return " / ".join(parts) if parts else "Mixed themes"


def build_unit_tree(
    nodes_by_id: dict,
    root_id: str,
    leaf_indices: Dict[str, List[int]],
    thetas_arr: np.ndarray,
    min_size: int = 50,
    max_purity: float = 0.70,
    topic_labels: Optional[List[str]] = None,
    topic_keys: Optional[List[List[str]]] = None,
    pw_mixture: float = 0.5,
    pw_size: float = 0.3,
    pw_balance: float = 0.2,
) -> Tuple[Optional[dict], int]:
    """Return a pruned annotation-unit tree (two-pass iterative) and unit count.

    Stopping rule (top-down, Pass 1): a node becomes a unit when any of the
    following hold:
      - size <= min_size  (small enough — literal, no hidden multiplier)
      - dominant_weight >= max_purity  (thematically pure)
      - it is a leaf (no children)

    Pass 1b (bottom-up promotion): any unit whose size < min_size is suppressed
    and its parent is promoted to a unit instead, so every emitted unit contains
    at least min_size queries.

    Otherwise both children are recursed into.
    """
    # Pass 1: classify every reachable node as unit or branching
    classified: dict = {}
    visit = [root_id]
    while visit:
        nid = visit.pop()
        if nid in classified:
            continue
        node = nodes_by_id[nid]
        size = node["size"]
        ids = leaf_indices[nid]
        mean_theta = thetas_arr[ids].mean(axis=0)
        dominant_weight = float(mean_theta.max())
        dominant_topic  = int(mean_theta.argmax())
        cids = node.get("children_ids", [])
        if len(cids) == 2:
            ls = nodes_by_id[cids[0]]["size"]
            rs = nodes_by_id[cids[1]]["size"]
            merge_balance = min(ls, rs) / max(ls, rs)
        else:
            merge_balance = 1.0
        if not cids:
            stop_reason = "leaf"
        elif size <= min_size:
            stop_reason = "size"
        elif dominant_weight >= max_purity:
            stop_reason = "purity"
        else:
            stop_reason = "recurse"
        is_unit = stop_reason != "recurse"
        classified[nid] = {
            "is_unit": is_unit,
            "stop_reason": stop_reason,
            "size": size,
            "dist": round(float(node.get("dist", 0.0)), 4),
            "repr": node.get("repr", ids[:5]),
            "mean_theta": mean_theta.round(4).tolist(),
            "dominant_topic": dominant_topic,
            "dominant_weight": round(dominant_weight, 4),
            "topic_mixture": round(1.0 - dominant_weight, 4),
            "merge_balance": round(merge_balance, 4),
            "cids": cids,
        }
        if not is_unit:
            visit.extend(cids)

    # Pass 1b: promote tiny units upward.
    # Any node classified as a unit solely because size < min_size is suppressed
    # and its parent is promoted to a unit instead, so every emitted unit
    # contains at least min_size queries.  We build parent pointers from the
    # already-classified set (children of unit nodes were never classified, so
    # we must not dereference their cids), then sweep bottom-up.
    parent_of: dict = {}
    topo_order: list = []
    seen_tp: set = set()
    tp_stack = [root_id]
    while tp_stack:
        nid = tp_stack.pop()
        if nid in seen_tp:
            continue
        seen_tp.add(nid)
        topo_order.append(nid)
        nc = classified[nid]
        if not nc["is_unit"]:          # only branch nodes have classified children
            for cid in nc["cids"]:
                parent_of[cid] = nid
                tp_stack.append(cid)

    for nid in reversed(topo_order):  # bottom-up
        nc = classified[nid]
        # Promote a unit upward only when it is both small AND not thematically pure.
        # A node with dominant_weight >= max_purity is already a coherent unit and
        # should stay as-is even if its size is below min_size — promoting it would
        # collapse the parent (which may be huge) into a single unit unnecessarily.
        if nc["is_unit"] and nc["size"] < min_size and nc["dominant_weight"] < max_purity:
            pid = parent_of.get(nid)
            if pid is not None:
                classified[pid]["is_unit"] = True   # parent absorbs this tiny node
                classified[nid]["is_unit"] = False  # suppress the tiny unit

    # Pass 2: build tree bottom-up (post-order)
    built: dict = {}
    units: list = []
    post = [(root_id, False)]
    while post:
        nid, done = post.pop()
        nc = classified[nid]
        if nc["is_unit"]:
            if nid not in built:
                label = make_unit_label(nc["mean_theta"], topic_labels, topic_keys)
                u = {
                    "is_unit": True,
                    "node_id": str(nid),
                    "size": nc["size"],
                    "dist": nc["dist"],
                    "label": label,
                    "repr": nc["repr"],
                    "mean_theta": nc["mean_theta"],
                    "dominant_topic": nc["dominant_topic"],
                    "dominant_weight": nc["dominant_weight"],
                    "topic_mixture": nc["topic_mixture"],
                    "merge_balance": nc["merge_balance"],
                    "children": [],
                }
                units.append(u)
                built[nid] = u
        elif done:
            built[nid] = {
                "is_unit": False,
                "node_id": str(nid),
                "size": nc["size"],
                "dist": nc["dist"],
                "dominant_topic": nc["dominant_topic"],
                "dominant_weight": nc["dominant_weight"],
                "topic_mixture": nc["topic_mixture"],
                "children": [built[cid] for cid in nc["cids"]],
            }
        else:
            post.append((nid, True))
            for cid in nc["cids"]:
                post.append((cid, False))

    # Score units and assign priority ranks
    if units:
        max_size = max(u["size"] for u in units)
        for u in units:
            size_norm = math.log(1 + u["size"]) / math.log(1 + max_size)
            u["priority"] = round(
                pw_mixture * u["topic_mixture"]
                + pw_size  * size_norm
                + pw_balance * u["merge_balance"],
                4,
            )
        for rank, u in enumerate(sorted(units, key=lambda x: -x["priority"]), 1):
            u["priority_rank"] = rank

    return built.get(root_id), len(units)


def generate_unit_labels(
    unit_tree: Optional[dict],
    queries: List[str],
    topic_keys: List[List[str]],
    topic_labels: List[str],
    prompter: Any,
    prompt_template: str,
    topn_docs: int = 5,
    max_retries: int = 3,
) -> Dict[str, str]:
    """Generate LLM labels for all annotation units in the tree.

    Returns a dict mapping node_id -> llm_label for each is_unit node.
    """
    if unit_tree is None:
        return {}

    results: Dict[str, str] = {}

    def _walk(node: dict) -> None:
        if node.get("is_unit"):
            nid = node["node_id"]
            repr_indices = node.get("repr", [])[:topn_docs]
            doc_texts = [queries[i] for i in repr_indices if i < len(queries)]
            docs_str = "\n- " + "\n- ".join(doc_texts) if doc_texts else "(none)"

            dom_id = node.get("dominant_topic", 0)
            dominant_topic = resolve_topic_label(dom_id, topic_labels, topic_keys)

            if topic_keys and dom_id < len(topic_keys):
                topic_keywords = ", ".join(topic_keys[dom_id][:10])
            else:
                topic_keywords = ""

            filled = prompt_template.format(
                dominant_topic=dominant_topic,
                topic_keywords=topic_keywords,
                docs=docs_str,
            )

            output = ""
            for attempt in range(max_retries):
                temperature = attempt * 0.1 if attempt > 0 else None
                raw, _ = prompter.prompt(
                    question=filled,
                    system_prompt_template_path=None,
                    temperature=temperature,
                )
                if raw and raw.strip():
                    output = raw.replace("\n", " ").strip()
                    break
            results[nid] = output

        for child in node.get("children", []):
            _walk(child)

    _walk(unit_tree)
    return results