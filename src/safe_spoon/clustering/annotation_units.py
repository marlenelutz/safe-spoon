"""Annotation unit selection from a hierarchical clustering dendrogram.

An "annotation unit" is a subtree of the dendrogram that is cohesive enough
for a single guideline to apply to it.

Stopping criteria (evaluated top-down):
  (a) leaf: Leaf node                                -> always stop
  (b) size: size <= min_size                         -> too small to split 
  (c) distance: node.dist / max_dist <= max_rel_dist -> fusion distance is low enough that the two sub-groups were already similar; one guideline covers the whole node
  
  Otherwise recurse into both children.
  ** max_rel_dist = fraction of the tree's total distance range below which a fusion is considered "close enough"
  ** node.dist = the linkage distance at which this node's two children were merged.
  ** max_dist = the root's distance, i.e., it expresses how "far" this merge was relative to the most heterogeneous split in the tree.
  
  *** node.dist / max_dist is a normalised measure of how "far up" the tree this node sits, relative to the root. A low value means the two groups were already similar before they were joined, so the whole node can be covered by one rubric.  A high value means the merge joined dissimilar groups and the node should be split further.
"""

import hashlib
import math
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def content_stable_id(member_indices: List[int]) -> str:
    """Content-derived identifier for a set of member query indices.

    Persists across pipeline re-runs on the same dataset as long as the same
    queries are present, even when clustering tree node IDs change.
    """
    digest = hashlib.sha1(
        ",".join(str(i) for i in sorted(member_indices)).encode()
    )
    return digest.hexdigest()[:16]


def resolve_topic_label(
    id: int,
    topic_labels: Optional[List[str]],
    topic_keys: Optional[List[List[str]]],
    sep: str = " · ",
) -> str:
    """Return the best available human-readable label for a topic index.

    Priority order:
      1. LLM-generated label (when present and not a placeholder "Topic N")
      2. Top-3 keywords joined by sep
      3. "Topic N" fallback
    """
    if topic_labels and id < len(topic_labels) and not topic_labels[id].startswith("Topic "):
        return topic_labels[id]
    if topic_keys and id < len(topic_keys):
        return sep.join(topic_keys[id][:3])
    return f"Topic {id + 1}"


def make_unit_label(
    mean_theta: List[float],
    topic_labels: Optional[List[str]],
    topic_keys: Optional[List[List[str]]],
    n_top: int = 2,
) -> str:
    """Derive a short keyword label from a unit's mean topic distribution.

    Takes the top-N topics by weight (skipping any below 0.05) and joins
    their labels with "/".  Falls back to "Mixed themes" when nothing clears
    the threshold.
    """
    ranked = sorted(enumerate(mean_theta), key=lambda x: -x[1])
    parts = []
    for tid, weight in ranked[:n_top]:
        if weight < 0.05:
            break
        parts.append(resolve_topic_label(tid, topic_labels, topic_keys))
    return " / ".join(parts) if parts else "Mixed themes"


def compute_leaf_indices(
    nodes_by_id: dict,
    root_id: str
) -> Dict[str, List[int]]:
    """Map every node id to the global leaf indices in its subtree.

    Parameters
    ----------
    nodes_by_id : dict
        {node_id: node_dict} as produced by flatten_tree().
    root_id : str
        ID of the root node.

    Returns
    -------
    leaf_indices : dict
        {node_id: [global_query_idx, ...]} for every node reachable from root.
    """
    nodes_by_id = {str(k): v for k, v in nodes_by_id.items()}
    leaf_indices: Dict[str, List[int]] = {}
    stack = [(str(root_id), False)]
    while stack:
        nid, done = stack.pop()
        node = nodes_by_id[nid]
        if done:
            cids = [str(cid) for cid in node.get("children_ids", [])]
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
                stack.append((str(cid), False))
    return leaf_indices


def build_unit_tree(
    nodes_by_id: dict,
    root_id: str,
    leaf_indices: Dict[str, List[int]],
    thetas_arr: np.ndarray,
    min_size: int = 20,
    max_rel_dist: float = 0.40,
    topic_labels: Optional[List[str]] = None,
    topic_keys: Optional[List[List[str]]] = None,
) -> Tuple[Optional[dict], int]:
    """Construct the annotation units from the dendogram.

    Three-pass algorithm:

    Pass 1: top-down classification
        Walk from the root.  At each node decide: STOP (emit a unit) or
        RECURSE into both children based on three criteria (leaf, size or distance).

    Pass 2: bottom-up promotion of undersized units
        Units that stopped only because of size may be too small to generalise from.  We suppress them and promote their parent to a unit instead. We only consider here size and not distance, because a distance-stopped unit is genuinely cohesive regardless of size. 

    Pass 3: output tree construction
        Post-order traversal.  Unit nodes become leaves of the annotation tree
        (their internal queries are collapsed).  Branching nodes become
        internal nodes.  Four geometric signals are attached to each unit for
        downstream priority scoring:
          topic_mixture  = 1 − dominant_weight
          heterogeneity  = 1 − intra_sim (fallback: 1 − dominant_weight if no embeddings)
          size_norm = log(1 + size) / log(1 + max_size)
          merge_balance = min(size, sibling_size) / max(size, sibling_size)

    Parameters
    ----------
    nodes_by_id : dict
        {node_id: node_dict} as produced by flatten_tree().
    root_id : str
        ID of the root node.
    leaf_indices : dict
        {node_id: [global_leaf_idx, ...]} from compute_leaf_indices().
    thetas_arr : np.ndarray  (n_docs × n_topics)
        Full document-topic matrix, globally indexed.
    min_size : int
        Hard floor: nodes at or below this size always stop.
    max_rel_dist : float  (0, 1]
        Normalised distance threshold.  Nodes whose dist / max_dist is at or
        below this value are declared cohesive and stop.
        Recommended range: 0.30 (fine-grained) – 0.50 (coarse).
    topic_labels / topic_keys : optional
        For keyword-based fallback unit labels.

    Returns
    -------
    unit_tree : dict
        {nodes: [...], root_id: ...} for the annotation-unit tree.  
    n_units : int
        Number of annotation units.
    """

    nodes_by_id = {str(k): v for k, v in nodes_by_id.items()}
    root_id = str(root_id)

    # Normalisation constant: the root's distance is the maximum in the tree.
    max_dist = float(nodes_by_id[root_id].get("dist", 1.0)) or 1.0

    ##########
    # Pass 1 #
    ##########
    classified: Dict[str, dict] = {}
    sibling_size: Dict[str, int] = {}   # node_id → size of its sibling

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
        dominant_topic = int(mean_theta.argmax())

        cids = [str(cid) for cid in node.get("children_ids", [])]

        # Record sibling sizes for merge_balance computation.
        if len(cids) == 2:
            sibling_size[cids[0]] = nodes_by_id[cids[1]]["size"]
            sibling_size[cids[1]] = nodes_by_id[cids[0]]["size"]

        # merge_balance: ratio of this node's size to its sibling's at the
        # parent merge.  Root gets 1.0 (no sibling).
        own_sibling = sibling_size.get(nid)
        if own_sibling is not None and size > 0:
            merge_balance = round(
                min(size, own_sibling) / max(size, own_sibling), 4)
        else:
            merge_balance = 1.0

        # rel_dist: how far up the tree this node sits, relative to the root.
        # 0 = leaf-level (two near-identical docs), 1 = root (most dissimilar).
        node_dist = float(node.get("dist", 0.0))
        rel_dist = round(node_dist / max_dist, 4)

        # Stopping decision
        if not cids:
            stop_reason = "leaf"
        elif size <= min_size:
            stop_reason = "size"
        elif rel_dist <= max_rel_dist:
            stop_reason = "distance"
        else:
            stop_reason = "recurse"

        is_unit = stop_reason != "recurse"

        intra_sim = node.get("intra_sim")
        heterogeneity = (
            round(1.0 - intra_sim, 4)
            if intra_sim is not None
            else round(1.0 - dominant_weight, 4)
        )

        classified[nid] = {
            "is_unit": is_unit,
            "stop_reason": stop_reason,
            "stable_id": content_stable_id(ids),
            "size": size,
            "dist": round(node_dist, 4),
            "rel_dist": rel_dist,
            "repr": node.get("repr", ids[:5]),
            "mean_theta": mean_theta.round(4).tolist(),
            "dominant_topic": dominant_topic,
            "dominant_weight": round(dominant_weight, 4),
            "topic_mixture": round(1.0 - dominant_weight, 4),
            "merge_balance": merge_balance,
            "intra_sim": intra_sim,
            "heterogeneity": heterogeneity,
            "cids": cids,
        }

        if not is_unit:
            visit.extend(cids)

    ##########
    # Pass 2 #
    ##########
    parent_of:  Dict[str, str] = {}
    topo_order: List[str] = []
    seen: set = set()

    # Build a parent->child traversal order (root first, leaves last) so that
    # reversing it gives a bottom-up order (leaves first, root last) for Pass 2.
    # We also record each node's parent so promotions can walk up the tree.
    tp_stack = [root_id]
    while tp_stack:
        nid = tp_stack.pop()
        if nid in seen:
            continue
        seen.add(nid)
        topo_order.append(nid)
        nc = classified[nid]
        if not nc["is_unit"]:
            for cid in nc["cids"]:
                parent_of[cid] = nid
                tp_stack.append(cid)

    # Promote size-stopped units whose parent is a branching node
    for nid in reversed(topo_order):
        nc = classified[nid]
        if nc["is_unit"] and nc["stop_reason"] == "size":
            pid = parent_of.get(nid)
            if pid is not None and not classified[pid]["is_unit"]:
                classified[nid]["is_unit"] = False
                classified[nid]["stop_reason"] = "promoted"
                classified[pid]["is_unit"] = True
                classified[pid]["stop_reason"] = "promotion"

    ##########
    # Pass 3 #
    ##########
    built: Dict[str, dict] = {}
    units: List[dict] = []

    post = [(root_id, False)]
    while post:
        nid, done = post.pop()
        nc = classified[nid]

        if nc["is_unit"]:
            if nid not in built:
                # at this
                label = make_unit_label(
                    nc["mean_theta"], topic_labels, topic_keys)
                u = {
                    "is_unit": True,
                    "node_id": str(nid),
                    "stable_id": nc["stable_id"],
                    "size": nc["size"],
                    "dist": nc["dist"],
                    "rel_dist": nc["rel_dist"],
                    "label": label,
                    "repr": nc["repr"],
                    "mean_theta": nc["mean_theta"],
                    "dominant_topic": nc["dominant_topic"],
                    "dominant_weight": nc["dominant_weight"],
                    "topic_mixture": nc["topic_mixture"],
                    "merge_balance": nc["merge_balance"],
                    "intra_sim": nc["intra_sim"],
                    "heterogeneity": nc["heterogeneity"],
                    "stop_reason": nc["stop_reason"],
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

    # Geometric priority signals.
    # size_norm is log-normalised so large units don't dominate.
    # topic_mixture and merge_balance are already on each unit from Pass 1.
    if units:
        max_size = max(u["size"] for u in units)
        for u in units:
            u["size_norm"] = round(
                math.log(1 + u["size"]) / math.log(1 + max_size), 4
            )

    return built.get(root_id), len(units)


_LABEL_NOTE_RE = re.compile(
    r"LABEL\s*:\s*(.*?)\s*(?:NOTE\s*:\s*(.*))?$",
    re.IGNORECASE | re.DOTALL,
)


def parse_label_note(raw: str) -> Tuple[str, str]:
    """Split a LABEL: / NOTE: LLM response into (label, note)."""
    if not raw:
        return "", ""
    text = raw.strip()
    m = _LABEL_NOTE_RE.search(text)
    if not m or not m.group(1):
        return text, ""
    return m.group(1).strip(), (m.group(2) or "").strip()


def generate_unit_labels(
    unit_tree: Optional[dict],
    queries: List[str],
    topic_keys: List[List[str]],
    topic_labels: List[str],
    prompter: Any,
    prompt_template: str,
    topn_docs: int = 5,
    max_retries: int = 3,
) -> Dict[str, Dict[str, str]]:
    """Generate LLM labels + annotation notes for every unit in the tree.

    Parameters
    ----------
    unit_tree : dict
        Annotation-unit tree as produced by build_unit_tree().
    queries : list of str
        Original query texts, globally indexed.
    topic_keys : list of list of str
        Top keywords for each topic, globally indexed.
    topic_labels : list of str
        LLM-generated labels for each topic, globally indexed.
    prompter : Prompter
        LLM interface for generating labels.
    prompt_template : str
        Template for the LLM prompt, with placeholders for dominant_topic,
        topic_keywords, and docs.
    topn_docs : int
        Number of representative docs to include in the prompt.
    max_retries : int
        Number of attempts to get a non-empty label from the LLM.

    Returns
    -------
    dict
        {node_id: {"label": str, "note": str}} for every is_unit node.
    """
    if unit_tree is None:
        return {}

    results: Dict[str, Dict[str, str]] = {}

    def _walk(node: dict) -> None:
        if node.get("is_unit"):
            nid = node["node_id"]
            repr_indices = node.get("repr", [])[:topn_docs]
            doc_texts = [queries[i] for i in repr_indices if i < len(queries)]
            docs_str = "\n- " + \
                "\n- ".join(doc_texts) if doc_texts else "(none)"

            dom_id = node.get("dominant_topic", 0)
            dominant_topic = resolve_topic_label(
                dom_id, topic_labels, topic_keys)
            topic_keywords = (
                ", ".join(topic_keys[dom_id][:10])
                if topic_keys and dom_id < len(topic_keys)
                else ""
            )

            filled = prompt_template.format(
                dominant_topic=dominant_topic,
                topic_keywords=topic_keywords,
                docs=docs_str,
            )

            output = {"label": "", "note": ""}
            for attempt in range(max_retries):
                temperature = attempt * 0.1 if attempt > 0 else None
                raw, _ = prompter.prompt(
                    question=filled,
                    system_prompt_template_path=None,
                    temperature=temperature,
                )
                label, note = parse_label_note(raw)
                if label:
                    output = {"label": label, "note": note}
                    break

            results[nid] = output

        for child in node.get("children", []):
            _walk(child)

    _walk(unit_tree)
    return results
