"""CLI entrypoint for the Safe Spoon pipeline (to generate the data needed for the front-end).
"""

import logging
import time

import click
from dotenv import load_dotenv
from pathlib import Path
from safe_spoon.clustering import AnnotationUnitModel, resolve_topic_label
from safe_spoon.embeddings import SentenceEncoder
from safe_spoon.logging_config import configure_logging
from safe_spoon.pipeline import config as pipeline_config
from safe_spoon.pipeline.corpus import load_queries_and_labels, resolve_categories
from safe_spoon.pipeline.serialize import (
    build_category_data,
    build_category_tree_payload,
    save_payload,
)
from safe_spoon.pipeline.training import (
    build_preprocessor,
    load_or_preprocess_reference_corpus,
    train_or_load_lda,
)
from safe_spoon.pipeline.tree import build_category_tree, compute_cut_levels, diagnose_oov
from safe_spoon.utils.data_utils import corpus_for_category

log = logging.getLogger(__name__)


def _run(cfg: "pipeline_config.PipelineConfig") -> None:
    """Run the Safe Spoon pipeline end-to-end, given a PipelineConfig object."""
    
    repo_root = Path(__file__).resolve().parents[3]

    log.info("Loading data from %s", cfg.input_file)
    queries, labels = load_queries_and_labels(cfg)
    categories = resolve_categories(cfg, labels)
    log.info("Loaded %d queries * %d categories: %s", len(queries), len(categories), categories)

    E_all = None
    if cfg.use_embedding_clustering:
        log.info("Computing sentence embeddings for full corpus (%d queries)...", len(queries))
        encoder = SentenceEncoder(
            model_name=cfg.embedding_model,
            batch_size=cfg.embedding_batch_size,
            logger=log,
        )
        E_all = encoder.encode(queries)
        log.info("Embeddings ready  shape=%s", E_all.shape)

    data_by_category = {}
    trees_by_category = {}
    t_total = time.time()

    for cat_id, cat in enumerate(categories, 1):
        df_corpus_cat = corpus_for_category(queries, labels, cat)
        n = len(df_corpus_cat)
        log.info("** Category %d/%d: '%s' (%d queries)", cat_id, len(categories), cat, n)
        if n < 2:
            log.warning("  Skipping '%s': fewer than 2 queries", cat)
            continue

        log.info("Initialising spaCy preprocessor (model=%s)", cfg.spacy_model)
        preprocessor = build_preprocessor(cfg, cat, repo_root)
        log.info("Spacy preprocessor initializzed * min_df=%d, max_df=%.2f",
                 preprocessor.min_df, preprocessor.max_df)

        df_ref = load_or_preprocess_reference_corpus(cfg, preprocessor)

        lda, selected_model_info = train_or_load_lda(cfg, cat, df_corpus_cat, preprocessor, df_ref)

        # get queries ordered in the same way as the LDA model's corpus, so that we can align the thetas and embeddings
        queries_ordered, query_ids_ordered = lda.get_ordered_corpus()
        n = len(queries_ordered)
        if n < 2:
            log.warning("  Skipping '%s': fewer than 2 queries", cat)
            continue

        X_cat = lda.get_thetas()
        topic_keys = lda.get_topic_keys()
        tm = lda.tm
        tm.load_tpc_labels()
        llm_labels = getattr(tm, "_tpc_labels", None)
        using_llm = bool(llm_labels and not all(l.startswith("Topic ") for l in llm_labels))
        topic_labels = llm_labels if using_llm else [
            resolve_topic_label(i, None, topic_keys) for i in range(len(topic_keys))
        ]
        log.info("  [%s] Using %s topic labels", cat, "LLM-generated" if using_llm else "keyword-based")
        log.info("  [%s] Thetas: %s * vocab: %d words", cat, X_cat.shape,
                 len(set(w for kw in topic_keys for w in kw)))

        log.info("  [%s] Building hierarchy (%d docs)...", cat, n)
        t0 = time.time()

        # All queries participate in clustering; OOV ones just get a less
        # reliable topic description.
        n_oov, oov_thr = diagnose_oov(cat, X_cat)
        valid_local = list(range(n))
        X_valid = X_cat

        E_valid = None
        if E_all is not None:
            global_ids_all = [query_ids_ordered[i] for i in range(n)]
            E_valid = E_all[global_ids_all]
            log.info("  [%s] Embedding matrix for clustering: %s (cosine distance)", cat, E_valid.shape)
        else:
            log.info("  [%s] No embeddings; falling back to Bhattacharyya over thetas", cat)

        flat_nodes, root_id, Z = build_category_tree(
            cfg, X_valid, valid_local, queries_ordered, X_cat, E_valid, oov_thr
        )
        log.info("  [%s] Hierarchy done in %.1fs * %d nodes (%d valid docs)",
                 cat, time.time() - t0, len(flat_nodes), len(valid_local))

        log.info("  [%s] Building annotation units + generating LLM labels...", cat)
        t1 = time.time()

        aum = AnnotationUnitModel(
            flat_nodes=flat_nodes,
            root_id=root_id,
            thetas=X_cat,
            topic_keys=topic_keys,
            topic_labels=topic_labels,
            queries=queries_ordered,
            model_path=lda.model_path,
            min_size=cfg.min_size,
            max_rel_dist=cfg.max_rel_dist,
            llm_provider=cfg.llm_provider,
            llm_model_type=cfg.llm_model,
            llm_api_key=cfg.llm_api_key,
            llm_server=cfg.llm_server,
            logger=log,
        )
        aum.build()
        aum.save_params()
        aum.generate_unit_outputs()
        aum.save_unit_labels()
        log.info("  [%s] %d annotation units labelled in %.1fs",
                 cat, aum.n_units, time.time() - t1)

        cuts, min_d, max_d = compute_cut_levels(cfg, cat, Z)

        data_by_category[cat] = build_category_data(
            cfg, lda, queries_ordered, query_ids_ordered, topic_keys, topic_labels,
            X_cat, n, selected_model_info,
            embedding_model_used=cfg.embedding_model if cfg.use_embedding_clustering else None,
        )
        trees_by_category[cat] = build_category_tree_payload(
            flat_nodes, root_id, valid_local, cuts, min_d, max_d, n, n_oov,
        )
        log.info("  [%s] Done", cat)

    log.info("All categories processed in %.1f min", (time.time() - t_total) / 60)
    save_payload(cfg, categories, data_by_category, trees_by_category)


@click.command()
@click.option("--config", "config_path", default="config/config.yaml", help="Path to config.yaml")
@click.option("--retrain", is_flag=True, default=None, help="Force LDA retraining for every category")
@click.option("--optimize", is_flag=True, default=None, help="Run num_topics optimisation for every category")
@click.option("--categories", multiple=True, default=None, help="Override which categories to process")
def main(config_path, retrain, optimize, categories):
    load_dotenv()
    configure_logging(config_path)
    cfg = pipeline_config.load(
        config_path,
        retrain=retrain,
        optimize=optimize,
        categories=list(categories) if categories else None,
    )
    _run(cfg)


if __name__ == "__main__":
    main()
