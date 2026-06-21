"""Hierarchical clustering utilities for topic-model outputs."""

from typing import List, Tuple

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, to_tree
from scipy.spatial.distance import squareform


def bhattacharyya_matrix(X: np.ndarray, eps = 1e-10) -> np.ndarray:
    """Compute Bhattacharyya distance matrix."""
    
    X_sqrt = np.sqrt(X + eps)
    BC = X_sqrt @ X_sqrt.T
    BC = np.clip(BC, eps, 1.0)
    D = -np.log(BC)
    np.fill_diagonal(D, 0.0)
    return D


def most_representative(
    indices: List[int],
    X: np.ndarray,
    n: int = 5,
    max_medoid: int = 200,
) -> List[int]:
    """Return the indices of the n-most representative items among indices. Uses exact medoid search for clusters up to max_medoid items and a
    centroid-approximation for larger clusters.
    """
    if len(indices) <= n:
        return list(indices)
    eps = 1e-10
    vecs = X[indices]
    if len(indices) <= max_medoid:
        sq = np.sqrt(vecs + eps)
        D = -np.log(np.clip(sq @ sq.T, eps, 1.0))
        scores = D.mean(axis=1)
    else:
        centroid = vecs.mean(axis=0)
        scores = -np.log(np.clip(
            np.sqrt(vecs + eps) @ np.sqrt(centroid + eps), eps, 1.0
        ))
    return [indices[i] for i in np.argsort(scores)[:n]]


def build_tree(
    root_node,
    global_indices: List[int],
    X: np.ndarray,
    queries: List[str],
    n_repr: int = 5,
) -> dict:
    """Build a nested-dict tree from a scipy ClusterNode.

    Parameters
    ----------
    root_node:
        Root ClusterNode from scipy.cluster.hierarchy.to_tree.
    global_indices:
        Mapping from local cluster indices to global row indices in X.
    X:
        The full document-topic matrix (all documents, not just this cluster).
    queries:
        Full list of query strings (indexed by global index).
    n_repr:
        Number of representative documents to store per internal node.
    """
    stack = [(root_node, None, False)]
    ordered = []
    node_dict = {}

    while stack:
        node, parent_id, is_right = stack.pop()

        if node.is_leaf():
            gidx = global_indices[node.id]
            d = {
                "id": f"leaf_{gidx}",
                "idx": gidx,
                "name": queries[gidx][:72] + ("…" if len(queries[gidx]) > 72 else ""),
                "full": queries[gidx],
                "size": 1,
                "dist": 0.0,
                "depth": 0,
                "repr": [gidx],
                "children": [],
                "_parent_id": parent_id,
                "_is_right": is_right,
                "_scipy_id": node.id,
            }
            node_dict[node.id] = d
            ordered.append(node.id)
        else:
            d = {
                "id": f"inner_{id(node)}",
                "name": "",
                "size": 0,
                "dist": round(float(node.dist), 4),
                "depth": 0,
                "repr": [],
                "children": [],
                "_parent_id": parent_id,
                "_is_right": is_right,
                "_scipy_id": node.id,
                "_left_id": node.left.id,
                "_right_id": node.right.id,
            }
            node_dict[node.id] = d
            ordered.append(node.id)
            stack.append((node.right, node.id, True))
            stack.append((node.left, node.id, False))

    for scipy_id in reversed(ordered):
        d = node_dict[scipy_id]

        if not d["children"] and "idx" in d:
            continue

        left_id = d.get("_left_id")
        right_id = d.get("_right_id")
        if left_id is None:
            continue

        left = node_dict[left_id]
        right = node_dict[right_id]

        d["children"] = [left, right]
        d["size"] = left["size"] + right["size"]
        d["name"] = f"{d['size']} queries"

        all_idxs = _gather_leaf_indices(left) + _gather_leaf_indices(right)
        d["repr"] = most_representative(all_idxs, X, n=n_repr)

    for d in node_dict.values():
        for k in ["_parent_id", "_is_right", "_scipy_id", "_left_id", "_right_id"]:
            d.pop(k, None)

    return node_dict[root_node.id]


def _gather_leaf_indices(node: dict) -> List[int]:
    """Recursively gather global indices of all leaf nodes under *node*."""
    result = []
    stack = [node]
    while stack:
        n = stack.pop()
        if not n["children"]:
            if "idx" in n:
                result.append(n["idx"])
        else:
            stack.extend(n["children"])
    return result


def flatten_tree(root: dict) -> Tuple[List[dict], str]:
    """Convert the nested tree dict to a flat node list.

    Returns
    -------
    flat : List[dict]
        All nodes in the tree, each with a children_ids list.
    root_id : str
        The id of the root node.
    """
    flat = []
    stack = [root]
    while stack:
        node = stack.pop()
        flat_node = {k: v for k, v in node.items() if k != "children"}
        flat_node["children_ids"] = [c["id"] for c in node.get("children", [])]
        flat.append(flat_node)
        stack.extend(node.get("children", []))
    return flat, root["id"]


def cluster_by_category(
    X: np.ndarray,
    labels: List[str],
    queries: List[str],
    n_cut_levels: int = 40,
    linkage_method: str = "average",
    n_repr: int = 5,
) -> dict:
    """Run Bhattacharyya agglomerative clustering per category.

    Parameters
    ----------
    X:
        Document-topic matrix, shape (n_docs, n_topics).
    labels:
        Category label for each document (same length as X).
    queries:
        Raw query strings (same length as X).
    n_cut_levels:
        Number of distance thresholds at which to compute flat cluster assignments.
    linkage_method:
        Linkage criterion passed to scipy.cluster.hierarchy.linkage.
    n_repr:
        Number of representative documents per tree node.

    Returns
    -------
    trees_by_category : dict
        Mapping from category name to a dict with keys
        nodes, root_id, indices, cuts, min_dist,
        max_dist, n.
    """
    categories = sorted(set(labels))
    trees_by_category = {}

    for cat in categories:
        cat_indices = [i for i, l in enumerate(labels) if l == cat]
        cat_n = len(cat_indices)
        if cat_n < 2:
            continue

        X_cat = X[cat_indices]
        D_full = bhattacharyya_matrix(X_cat)
        D_cond = squareform(D_full, checks=False)
        Z = linkage(D_cond, method=linkage_method)

        min_d = float(Z[:, 2].min())
        max_d = float(Z[:, 2].max())

        cuts = []
        for d in np.linspace(min_d * 0.99, max_d * 1.01, n_cut_levels):
            assignment = fcluster(Z, t=d, criterion="distance").tolist()
            cuts.append({
                "distance": round(float(d), 4),
                "n_clusters": len(set(assignment)),
                "assignment": assignment,
            })

        root_node, _ = to_tree(Z, rd=True)
        tree = build_tree(root_node, cat_indices, X, queries, n_repr=n_repr)
        flat_nodes, root_id = flatten_tree(tree)

        trees_by_category[cat] = {
            "nodes": flat_nodes,
            "root_id": root_id,
            "indices": cat_indices,
            "cuts": cuts,
            "min_dist": round(min_d, 4),
            "max_dist": round(max_d, 4),
            "n": cat_n,
        }

    return trees_by_category
