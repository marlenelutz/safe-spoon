"""Corpus loading for the pipeline with PipelineConfig."""

from typing import List, Tuple

from safe_spoon.pipeline.config import PipelineConfig
from safe_spoon.utils.data_utils import load_corpus_df


def load_queries_and_labels(cfg: PipelineConfig) -> Tuple[List[str], List[str]]:
    """Load the full corpus, returning queries and their category labels."""
    _, queries, labels = load_corpus_df(
        cfg.input_file, content_col=cfg.content_col, label_col=cfg.label_col
    )
    return queries, labels


def resolve_categories(cfg: PipelineConfig, labels: List[str]) -> List[str]:
    """Resolve which categories to process: active_categories filtered to those actually present in the data, or all present categories if active_categories is empty."""
    all_categories = sorted(set(labels))
    if cfg.active_categories:
        return [c for c in cfg.active_categories if c in all_categories]
    return all_categories
