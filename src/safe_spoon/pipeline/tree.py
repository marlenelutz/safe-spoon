"""Hierarchical tree construction + dendrogram cut levels for one category."""

import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy.cluster.hierarchy import fcluster

from safe_spoon.clustering import build_flat_tree
from safe_spoon.pipeline.config import PipelineConfig

log = logging.getLogger(__name__)


def diagnose_oov(
    category: str,
    X_cat: np.ndarray
) -> Tuple[int, float]:
    """Log how many queries have a near-uniform (flat) LDA theta. Returns the OOV count and the OOV threshold; queries are never excluded from clustering.

    Parameters
    ----------
    category : str
        The name of the category being analyzed.
    X_cat : np.ndarray
        The LDA theta matrix for the category, where each row corresponds to a query and each column corresponds to a topic.

    Returns
    -------
    n_oov : int
        The number of queries with flat thetas (max_theta < oov_thr).
    oov_thr : float
        The threshold for determining out-of-vocabulary queries based on their LDA thetas.
    """
    n = X_cat.shape[0]
    K = X_cat.shape[1]
    oov_thr = (1.0 / K) + 1e-4
    oov_mask = X_cat.max(axis=1) < oov_thr
    n_oov = int(oov_mask.sum())
    if n_oov > 0:
        log.info(
            "  [%s] %d OOV queries (%.1f%%) have flat thetas (max_theta < %.2f) "
            "they will cluster via embeddings but their topic descriptions "
            "will be less reliable.",
            category, n_oov, 100 * n_oov / n, oov_thr,
        )
    return n_oov, oov_thr


def build_category_tree(
    cfg: PipelineConfig,
    X_valid: np.ndarray,
    valid_local: List[int],
    queries_ordered: List[str],
    X_full: np.ndarray,
    E_valid: Optional[np.ndarray],
    oov_thr: Optional[float] = 0.1,
) -> Tuple[list, str, np.ndarray]:
    """Build a hierarchical tree for one category, given the LDA thetas and optional embeddings.

    Parameters
    ----------
    cfg : PipelineConfig
        The pipeline configuration object.
    X_valid : np.ndarray
        The LDA thetas for the valid queries in this category.
    valid_local : List[int]
        The indices of the valid queries in the original corpus.
    queries_ordered : List[str]
        The queries in the order corresponding to X_valid.
    X_full : np.ndarray
        The LDA thetas for all queries in this category.
    E_valid : Optional[np.ndarray]
        The embeddings for the valid queries in this category, if available.
    oov_thr : Optional[float]
        The threshold for determining out-of-vocabulary queries based on their LDA thetas.
    """

    flat_nodes, root_id, Z = build_flat_tree(
        X_valid,
        valid_local,
        queries_ordered,
        linkage_method=cfg.linkage_method,
        n_repr=cfg.n_repr_queries,
        X_full=X_full,
        embeddings=E_valid,
        embeddings_full=E_valid,
        min_dominant_weight=oov_thr,
    )
    return flat_nodes, root_id, Z


def compute_cut_levels(
    cfg: PipelineConfig,
    category: str,
    Z: np.ndarray
) -> Tuple[list, float, float]:
    """Compute cut levels for the dendrogram of a category's hierarchical tree.

    Parameters
    ----------
    cfg : PipelineConfig
        The pipeline configuration object.
    category : str
        The name of the category for which to compute cut levels.
    Z : np.ndarray
        The linkage matrix representing the hierarchical clustering of the category's queries.

    Returns
    -------
    cuts : list
        A list of dictionaries, each containing the cut distance, number of clusters, and cluster assignments for that cut level.
    min_d : float
        The minimum distance in the linkage matrix Z.
    max_d : float
        The maximum distance in the linkage matrix Z.
    """
    min_d, max_d = float(Z[:, 2].min()), float(Z[:, 2].max())
    log.info("  [%s] Computing %d cut levels...", category, cfg.n_cut_levels)
    cuts = []
    for d in np.linspace(min_d * 0.99, max_d * 1.01, cfg.n_cut_levels):
        assignment = fcluster(Z, t=d, criterion="distance").tolist()
        cuts.append({
            "distance": round(float(d), 4),
            "n_clusters": len(set(assignment)),
            "assignment": assignment,
        })
    log.info("  [%s] Cuts: %d->%d clusters across levels",
             category, cuts[0]["n_clusters"], cuts[-1]["n_clusters"])
    return cuts, min_d, max_d
