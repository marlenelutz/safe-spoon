"""Dataclasses for pipeline configuration, and a loader function to read from config.yaml."""
from dataclasses import dataclass, fields
from typing import List, Optional

from safe_spoon.utils.common import load_yaml_config_file


@dataclass
class PipelineConfig:
    # Paths
    input_file: str
    reference_corpus: str
    reference_corpus_preprocessed: str
    output_json: str

    # Data / corpus
    content_col: str
    label_col: str
    active_categories: List[str]
    oov_threshold: float

    # LDA
    n_topics: int
    lda_iters: int
    lda_alpha: float
    lda_eta: float
    min_df: int
    max_df: float
    spacy_model: str
    retrain: bool
    optimize: bool
    optimize_range: List[int]

    # Clustering
    linkage_method: str
    n_repr_queries: int
    n_cut_levels: int

    # Annotation units 
    min_size: int
    max_rel_dist: float

    # Embedding
    use_embedding_clustering: bool
    embedding_model: str
    embedding_batch_size: int

    # LLM
    llm_provider: str
    llm_model: str
    llm_server: Optional[str]
    llm_api_key: Optional[str]
    top_docs_per_topic: int

    @property
    def optimize_topic_range(self) -> range:
        start, stop, step = self.optimize_range
        return range(start, stop + 1, step)


def load(
    config_path: str = "config/config.yaml",
    *,
    retrain: Optional[bool] = None,
    optimize: Optional[bool] = None,
    categories: Optional[List[str]] = None,
    use_embedding_clustering: bool = True,
) -> PipelineConfig:
    """Load pipeline configuration from config.yaml, with optional CLI overrides."""
    cfg = load_yaml_config_file(config_path)

    field_names = {f.name for f in fields(PipelineConfig)}
    kwargs = {k: v for k, v in cfg.items() if k in field_names}
    kwargs.update(
        active_categories=list(
            categories) if categories else cfg["active_categories"],
        retrain=cfg["retrain"] if retrain is None else retrain,
        optimize=cfg["optimize"] if optimize is None else optimize,
        use_embedding_clustering=use_embedding_clustering,
    )
    return PipelineConfig(**kwargs)
