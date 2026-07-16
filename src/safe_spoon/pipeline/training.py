"""Per-category preprocessing + LDA training/loading."""

import json
import logging
import time
from pathlib import Path
from typing import Tuple

import pandas as pd

from safe_spoon.pipeline.config import PipelineConfig
from safe_spoon.preprocessing import SimpleTMPreprocessor
from safe_spoon.topic_modeling import LDATopicModel

log = logging.getLogger(__name__)


def build_preprocessor(
    cfg: PipelineConfig, 
    category: str, 
    repo_root: Path
) -> SimpleTMPreprocessor:
    """Build a SimpleTMPreprocessor for a given category, using the specified configuration and repository root.
    
    Parameters
    ----------
    cfg : PipelineConfig
        The pipeline configuration object containing various settings.
    category : str
        The category for which to build the preprocessor.
    repo_root : Path
        The root path of the repository, used to locate stopword files.
    
    Returns
    -------
    SimpleTMPreprocessor
        An instance of SimpleTMPreprocessor configured for the specified category.
    """
    return SimpleTMPreprocessor(
        spacy_model=cfg.spacy_model,
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        stopword_files=[str(repo_root / "static" / "stops" / f"{category}.txt")],
    )


def load_or_preprocess_reference_corpus(
    cfg: PipelineConfig, 
    preprocessor: SimpleTMPreprocessor
) -> pd.DataFrame:
    """Load the preprocessed reference corpus if it exists; otherwise, preprocess the reference corpus using the given preprocessor and save it.
    
    Parameters
    ----------
    cfg : PipelineConfig
        The pipeline configuration object containing various settings.
    preprocessor : SimpleTMPreprocessor
        The preprocessor to use for text preprocessing.
        
    Returns
    -------
    pd.DataFrame
        The preprocessed reference corpus as a DataFrame.
    """
    ref_preproc_path = Path(cfg.reference_corpus_preprocessed)
    if ref_preproc_path.exists():
        log.info("Loading preprocessed reference corpus from %s", ref_preproc_path)
        df_ref = pd.read_csv(ref_preproc_path, encoding="utf-8")
        log.info("Preprocessed reference corpus loaded * %d docs", len(df_ref))
        return df_ref

    log.info("Preprocessing reference corpus from %s", cfg.reference_corpus)
    t0 = time.time()
    df_ref = preprocessor.fit_transform(
        pd.read_csv(cfg.reference_corpus, encoding="latin-1"),
        text_col="text", id_col="id",
        compute_bow=False, compute_tfidf=False,
    )
    log.info("Reference corpus preprocessed in %.1f sec * %d docs", time.time() - t0, len(df_ref))
    ref_preproc_path.parent.mkdir(parents=True, exist_ok=True)
    df_ref.to_csv(ref_preproc_path, index=False, encoding="utf-8")
    log.info("Preprocessed reference corpus saved to %s", ref_preproc_path)
    return df_ref


def train_or_load_lda(
    cfg: PipelineConfig,
    category: str,
    df_corpus_cat: pd.DataFrame,
    preprocessor: SimpleTMPreprocessor,
    df_ref: pd.DataFrame,
) -> Tuple[LDATopicModel, dict]:
    """Optimize / retrain / load an LDA model for one category.add()
    
    Parameters
    ----------
    cfg : PipelineConfig
        The pipeline configuration object containing various settings.
    category : str
        The category for which to train or load the LDA model.  
    df_corpus_cat : pd.DataFrame
        The DataFrame containing the queries for the specified category.
    preprocessor : SimpleTMPreprocessor
        The preprocessor to use for text preprocessing.
    df_ref : pd.DataFrame
        The preprocessed reference corpus DataFrame.
        
    Returns
    -------
    Tuple[LDATopicModel, dict]
        A tuple containing the trained or loaded LDA model and a dictionary with information about the selected model (e.g., model path, number of topics, mean coherence).
    """
    model_dir = f"./data/models/{category.replace(' ', '_')}"
    opt_base = Path(f"./data/models/{category.replace(' ', '_')}_optimize")
    opt_results_path = opt_base / "optimization_results.json"
    selected_model_info: dict = {}

    if cfg.optimize:
        log.info("  [%s] Optimising num_topics over %s -> %s", category, list(cfg.optimize_topic_range), opt_base)
        opt = LDATopicModel.optimize_num_topics(
            data=df_corpus_cat.to_dict("records"),
            base_path=opt_base,
            topic_range=cfg.optimize_topic_range,
            num_iters=cfg.lda_iters,
            alpha=cfg.lda_alpha,
            eta=cfg.lda_eta,
            preprocessor=preprocessor,
            reference_corpus=df_ref["text"].tolist(),
            logger=log,
        )
        best = opt["selected"][0]
        log.info(
            "  [%s] Best k=%d (coherence=%.4f) -> %s",
            category, best["k"], best["mean_coherence"], best["model_path"],
        )
        selected_model_info = {
            "model_path": best["model_path"],
            "k": best["k"],
            "mean_coherence": best["mean_coherence"],
        }
        lda = LDATopicModel.load(best["model_path"], corpus=df_corpus_cat)
        lda.llm_provider   = cfg.llm_provider
        lda.llm_model_type = cfg.llm_model
        lda.llm_api_key    = cfg.llm_api_key
        log.info("  [%s] Generating topic labels for best model (k=%d)...", category, best["k"])
        tm_opt = lda.tm
        results = tm_opt.generate_topic_outputs(task="label", topn=3)
        tm_opt._tpc_labels = [lbl for _, lbl in sorted(results)]
        (Path(best["model_path"]) / "TMmodel" / "tpc_labels.txt").write_text(
            "\n".join(tm_opt._tpc_labels), encoding="utf-8"
        )
    elif not cfg.retrain and not Path(model_dir).exists() and opt_results_path.exists():
        with opt_results_path.open(encoding="utf-8") as f:
            opt = json.load(f)
        best = opt["selected"][0]
        log.info(
            "  [%s] Loading previously-optimized model (k=%d) from %s",
            category, best["k"], best["model_path"],
        )
        selected_model_info = {
            "model_path": best["model_path"],
            "k": best["k"],
            "mean_coherence": best["mean_coherence"],
        }
        lda = LDATopicModel.load(best["model_path"], corpus=df_corpus_cat)
    elif cfg.retrain or not Path(model_dir).exists():
        log.info("  [%s] Training LDA (%d topics, %d iters)...", category, cfg.n_topics, cfg.lda_iters)
        lda = LDATopicModel(
            model_dir,
            num_topics=cfg.n_topics,
            num_iters=cfg.lda_iters,
            alpha=cfg.lda_alpha,
            eta=cfg.lda_eta,
            preprocessor=preprocessor,
            do_labeller=True,
            do_summarizer=False,
            llm_provider=cfg.llm_provider,
            llm_model_type=cfg.llm_model,
            llm_api_key=cfg.llm_api_key,
            llm_server=cfg.llm_server,
        )
        _, elapsed = lda.train(df_corpus_cat.to_dict("records"))
        log.info("  [%s] LDA done in %.1f min", category, elapsed / 60)
    else:
        log.info("  [%s] Loading existing model from %s", category, model_dir)
        lda = LDATopicModel.load(model_dir, corpus=df_corpus_cat)

    return lda, selected_model_info
