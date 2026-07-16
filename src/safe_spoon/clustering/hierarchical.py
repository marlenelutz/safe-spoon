"""Hierarchical clustering utilities for topic-model outputs / embeddings."""

from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform
from sentence_transformers.util import cos_sim, normalize_embeddings


def bhattacharyya_matrix(X: np.ndarray, eps=1e-10) -> np.ndarray:
    """Compute Bhattacharyya distance matrix over a theta matrix.

    Parameters
    ----------
    X:
        Dense matrix of shape (n_docs, n_topics).  Intended for LDA theta
        distributions but works for any dense representation.
    eps:
        Small value added to avoid log(0) and division by zero.

    Returns
    -------
    D : np.ndarray of shape (n_docs, n_docs)
        Symmetric distance matrix with zeros on the diagonal.
    """

    X_sqrt = np.sqrt(X + eps)
    BC = X_sqrt @ X_sqrt.T
    BC = np.clip(BC, eps, 1.0)
    D = -np.log(BC)
    np.fill_diagonal(D, 0.0)
    return D


def cosine_distance_matrix(X: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine distance matrix (1 - cosine_similarity).

    Parameters
    ----------
    X:
        Dense matrix of shape (n_docs, n_dims).

    Returns
    -------
    D : np.ndarray of shape (n_docs, n_docs)
        Symmetric distance matrix with zeros on the diagonal and values
        in [0, 2].  Values are clipped to [0, 2] to guard against
        floating-point rounding outside that range.
    """
    X_norm = normalize_embeddings(torch.from_numpy(X.astype(np.float32)))
    sim = cos_sim(X_norm, X_norm).numpy()
    D = 1.0 - sim
    D = np.clip(D, 0.0, 2.0)
    np.fill_diagonal(D, 0.0)
    return D


def intra_cluster_similarity(
    indices: List[int],
    embeddings: np.ndarray,
) -> float:
    """Mean pairwise cosine similarity among the embeddings of a set of documents.

    This is the cohesion signal used as a stopping criterion in build_unit_tree:
    a high value means the documents in this node are semantically similar to
    each other and can likely be covered by a single annotation rubric.

    Parameters
    ----------
    indices:
        Global document indices whose embeddings to compare.
    embeddings:
        Dense embedding matrix, globally indexed (shape n_docs x n_dims).

    Returns
    -------
    float in [0, 1]
        Mean pairwise cosine similarity.  Returns 1.0 for single-document
        clusters (trivially self-similar) and 0.0 when embeddings is None.
    """
    if embeddings is None or len(indices) <= 1:
        return 1.0
    vecs = normalize_embeddings(
        torch.from_numpy(embeddings[indices].astype(np.float32))
    )
    sim = cos_sim(vecs, vecs).numpy()
    # Mean of upper triangle (exclude self-similarity on diagonal)
    n = len(indices)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return float(sim[mask].mean())


def most_representative(
    indices: List[int],
    X: np.ndarray,
    n: int = 5,
    max_medoid: int = 200,
    min_dominant_weight: float = 0.15,
    embeddings: Optional[np.ndarray] = None,
) -> List[int]:
    """Return the indices of the n most representative items among indices.

    Uses exact medoid search for small clusters and centroid-based
    approximation for large ones.

    The representation space is chosen adaptively:

    * Embeddings are used when "embeddings" is provided AND the mean
      dominant LDA weight of the candidates is weak (i.e., < min_dominant_weight x 2). .

    * Thetas are used otherwise. The medoid in theta space is the
      query whose topic distribution is closest to the cluster average, i.e.
      the most thematically representative query.

    Parameters
    ----------
    indices:
        Global document indices to select from.
    X:
        Full document-topic matrix, globally indexed.
    n:
        Number of representatives to return.
    max_medoid:
        Use exact pairwise medoid search below this size; centroid
        approximation above it.
    min_dominant_weight:
        OOV threshold.  Queries whose dominant topic weight is below this
        value (flat LDA prior) or above 0.999 (single surviving token) are
        excluded before selection.  Falls back to full candidate set only
        when no informative items remain.
    embeddings:
        Full embedding matrix, globally indexed (same row order as X).
        When provided and LDA signal is weak, representatives are selected
        as the queries closest to the embedding centroid of the cluster.
    """

    candidates = list(indices)

    # exclude queries with flat thetas
    informative = [
        i for i in candidates
        if X[i].max() >= min_dominant_weight and X[i].max() < 0.999
    ]
    if informative:
        candidates = informative

    if len(candidates) <= n:
        return candidates

    # decide embbeddings vs thetas
    mean_dom = float(X[candidates].max(axis=1).mean())
    use_embeddings = (
        embeddings is not None
        and mean_dom < min_dominant_weight * 2
    )

    if use_embeddings:
        vecs = embeddings[candidates].astype(np.float32)
        centroid = vecs.mean(axis=0, keepdims=True)
        sims = cos_sim(vecs, centroid).numpy().squeeze()
        top_idx = np.argsort(-sims)[:n]
        return [candidates[int(i)] for i in top_idx]

    eps = 1e-10
    vecs = X[candidates]
    if len(candidates) <= max_medoid:
        # Exact medoid: minimize mean Bhattacharyya distance to all others
        sq = np.sqrt(vecs + eps)
        D = -np.log(np.clip(sq @ sq.T, eps, 1.0))
        scores = D.mean(axis=1)
    else:
        # Centroid approximation: distance to mean theta vector
        centroid = vecs.mean(axis=0)
        scores = -np.log(np.clip(
            np.sqrt(vecs + eps) @ np.sqrt(centroid + eps), eps, 1.0
        ))
    return [candidates[i] for i in np.argsort(scores)[:n]]


def build_tree(
    root_node,
    global_indices: List[int],
    X: np.ndarray,
    queries: List[str],
    n_repr: int = 5,
    min_dominant_weight: float = 0.15,
    embeddings_full: Optional[np.ndarray] = None,
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
    min_dominant_weight:
        Forwarded to most_representative()
    embeddings_full:
        Full embedding matrix, globally indexed.  When provided, forwarded
        to most_representative() so that representatives are selected in
        embedding space when LDA signal is weak.

    Returns
    -------
    tree : dict
        Nested dict representation of the tree, with each node containing
        id, name, size, dist, depth, repr, and children keys.
    """

    stack = [(root_node, None, False)]
    ordered = []
    node_dict = {}

    while stack:
        node, parent_id, is_right = stack.pop()

        if node.is_leaf():
            gid = global_indices[node.id]
            d = {
                "id": gid,
                "name": queries[gid][:72] + ("..." if len(queries[gid]) > 72 else ""),
                "full": queries[gid],
                "size": 1,
                "dist": 0.0,
                "depth": 0,
                "repr": [gid],
                "children": [],
                "_parent_id": parent_id,
                "_is_right": is_right,
                "_scipy_id": node.id,
            }
            node_dict[node.id] = d
            ordered.append(node.id)
        else:
            d = {
                "id": f"inner_{node.id}",
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

        left_id = d.get("_left_id")
        right_id = d.get("_right_id")
        if left_id is None:
            continue

        left = node_dict[left_id]
        right = node_dict[right_id]

        d["children"] = [left, right]
        d["size"] = left["size"] + right["size"]
        d["name"] = f"{d['size']} queries"

        all_ids = _gather_leaf_indices(left) + _gather_leaf_indices(right)
        d["repr"] = most_representative(
            all_ids,
            X,
            n=n_repr,
            min_dominant_weight=min_dominant_weight,
            embeddings=embeddings_full,
        )

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
            if "id" in n:
                result.append(n["id"])
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


def build_flat_tree(
    thetas,
    indices,
    queries,
    linkage_method: str = "average",
    n_repr: int = 5,
    X_full: Optional[np.ndarray] = None,
    embeddings: Optional[np.ndarray] = None,
    embeddings_full: Optional[np.ndarray] = None,
    min_dominant_weight: float = 0.15,
) -> Tuple[list, str, np.ndarray]:
    """Wrapper around scipy linkage and to_tree to build a flat tree representation from a document-topic or embedding matrix.

    Parameters
    ----------
    thetas:
        Document-topic matrix for the clustered subset (shape n_valid x K).
    indices:
        Global indices mapping scipy leaf ids  rows in X_full / positions
        in queries.
    queries:
        Full query list indexed by global position.
    linkage_method:
        Linkage criterion forwarded to scipy.cluster.hierarchy.linkage.
    n_repr:
        Number of representative documents stored per internal node.
    X_full:
        Full document-topic matrix (all docs, globally indexed).  Used for
        representative selection and topic characterisation.  Defaults to
        thetas when omitted.
    embeddings:
        Optional dense embedding matrix for the clustered subset
        (n_valid x n_dims).  When supplied, pairwise cosine distances are
        used for clustering instead of Bhattacharyya distances.
    embeddings_full:
        Full embedding matrix, globally indexed.  Used for two purposes:
        (1) intra-cluster cohesion (intra_sim) stored on every flat node,
        and (2) representative selection when LDA signal is weak.
    min_dominant_weight:
        OOV threshold forwarded to most_representative.

    Returns
    -------
    flat_nodes : list[dict]
    root_id    : str
    Z : np.ndarray
        Linkage matrix from scipy.cluster.hierarchy.linkage.
    """
    if embeddings is not None:
        D = cosine_distance_matrix(embeddings)
    else:
        D = bhattacharyya_matrix(thetas)

    Z = linkage(squareform(D, checks=False), method=linkage_method)
    root_node, _ = to_tree(Z, rd=True)

    tree = build_tree(
        root_node,
        indices,
        X_full if X_full is not None else thetas,
        queries,
        n_repr=n_repr,
        min_dominant_weight=min_dominant_weight,
        embeddings_full=embeddings_full,
    )
    flat_nodes, root_id = flatten_tree(tree)

    # Annotate each node with its intra-cluster embedding similarity.
    if embeddings_full is not None:
        nodes_by_id_flat = {str(n["id"]): n for n in flat_nodes}

        def _leaf_ids(nid_str):
            acc = {}
            stk = [(nid_str, False)]
            while stk:
                cur, done = stk.pop()
                nd = nodes_by_id_flat[cur]
                cids = [str(c) for c in nd.get("children_ids", [])]
                if done:
                    if not cids:
                        acc[cur] = [nd["id"]]
                    else:
                        merged = []
                        for c in cids:
                            merged.extend(acc[c])
                        acc[cur] = merged
                else:
                    stk.append((cur, True))
                    for c in cids:
                        stk.append((c, False))
            return acc

        leaf_idx_map = _leaf_ids(str(root_id))
        for n in flat_nodes:
            nid = str(n["id"])
            leaf_ids = leaf_idx_map.get(nid, [])
            n["intra_sim"] = round(
                intra_cluster_similarity(leaf_ids, embeddings_full), 4
            )
    else:
        for n in flat_nodes:
            n["intra_sim"] = None

    return flat_nodes, root_id, Z
