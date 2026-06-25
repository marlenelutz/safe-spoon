import json
import logging
import os
import time
import ast

import numpy as np
from pathlib import Path
import pandas as pd

from dotenv import load_dotenv
from scipy.cluster.hierarchy import fcluster

from safe_spoon.clustering import build_flat_tree, resolve_topic_label
from safe_spoon.utils.data_utils import corpus_for_category, load_corpus_df
from safe_spoon.utils.renderer import save_json
from safe_spoon.utils.common import load_annotation_unit_config
from safe_spoon.preprocessing import SimpleTMPreprocessor
from safe_spoon.topic_modeling import LDATopicModel
from safe_spoon.topic_modeling.tm_model import top_docs_per_topic


def load_tm_coords(tm_folder: Path, n_topics: int) -> list[list[float]]:
    """Read topic coordinates from TMmodel/tpc_coords.txt.

    The file stores one tuple per line, e.g. ``(x, y)``.
    """
    coords_path = tm_folder / "tpc_coords.txt"
    if not coords_path.is_file():
        log.warning("tpc_coords.txt not found at %s; using zero coordinates", coords_path)
        return [[0.0, 0.0] for _ in range(n_topics)]

    coords = []
    with coords_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            x, y = ast.literal_eval(line)
            coords.append([float(x), float(y)])

    if len(coords) != n_topics:
        log.warning(
            "tpc_coords length mismatch at %s (got %d, expected %d); truncating/padding",
            coords_path,
            len(coords),
            n_topics,
        )
    if len(coords) < n_topics:
        coords.extend([[0.0, 0.0] for _ in range(n_topics - len(coords))])
    return coords[:n_topics]

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# config
INPUT_FILE = "data/high_risk_automatically_labelled_filtered_cleaned.csv" # this inludes the filtering done in aux_scripts/data_filtering.py
INPUT_REF_FILE = "data/reference_corpus.csv" # this is the reference corpus used for coherence calculation
INPUT_REF_PREPROC_FILE = "data/reference_corpus_preprocessed.csv" # this is the reference corpus used for coherence calculation
CONTENT_COL = "content"
LABEL_COL = "high_risk_label"
OUTPUT_JSON = "data/output/viz_v5_data.json"

CATEGORIES = ['Economic and Financial', 'Health', 'Moral Values and Religion']
#CATEGORIES = ['Health']

RETRAIN = False
OPTIMIZE = False
OPTIMIZE_RANGE = range(30, 101, 5)

LLM_PROVIDER = "openai"
LLM_MODEL    = "gpt-5.4-nano-2026-03-17"
LLM_API_KEY  = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")

N_TOPICS = 30
LDA_ITERS = 1500
N_REPR_QUERIES = 50
N_CUT_LEVELS = 40
LINKAGE_METHOD = "average"

_cfg = load_annotation_unit_config()
TOP_DOCS_PER_TOPIC = _cfg["ui_display"]["topic_top_docs_generated"]

if __name__ == "__main__":
    log.info("Loading data from %s", INPUT_FILE)
    _, queries, labels = load_corpus_df(INPUT_FILE, content_col=CONTENT_COL, label_col=LABEL_COL)

    all_categories = sorted(set(labels))
    categories = [c for c in CATEGORIES if c in all_categories] if CATEGORIES else all_categories
    log.info("Loaded %d queries * %d categories: %s", len(queries), len(categories), categories)

    data_by_category = {}
    trees_by_category = {}
    t_total = time.time()

    for cat_id, cat in enumerate(categories, 1):
        df_corpus_cat = corpus_for_category(queries, labels, cat)
        n = len(df_corpus_cat)
        log.info("── Category %d/%d: '%s' (%d queries)", cat_id, len(categories), cat, n)
        if n < 2:
            log.warning("  Skipping '%s' — fewer than 2 queries", cat)
            continue
        
        log.info("Initialising spaCy preprocessor (model=en_core_web_lg)")
        preprocessor = SimpleTMPreprocessor(spacy_model="en_core_web_lg", min_df=10, max_df=0.6, stopword_files=[str(Path(__file__).parent / "static" / "stops" / f"{cat}.txt")])
        log.info("Spacy preprocessor initialised * min_df=%d, max_df=%.2f", preprocessor.min_df, preprocessor.max_df)
        
        #preprocess reference text
        if Path(INPUT_REF_PREPROC_FILE).exists():
            log.info("Loading preprocessed reference corpus from %s", INPUT_REF_PREPROC_FILE)
            df_ref = pd.read_csv(INPUT_REF_PREPROC_FILE, encoding="utf-8")
            log.info("Preprocessed reference corpus loaded * %d docs", len(df_ref))
        else:
            log.info("Preprocessing reference corpus from %s", INPUT_REF_FILE)
            time_ref = time.time()
            df_ref = preprocessor.fit_transform(pd.read_csv(INPUT_REF_FILE,  encoding="latin-1"), text_col="text", id_col="id",
                    compute_bow=False, compute_tfidf=False,
                )
            time_ref_elapsed = time.time() - time_ref
            log.info("Reference corpus preprocessed in %.1f sec * %d docs", time_ref_elapsed, len(df_ref))
            # save preprocessed reference corpus to file for future use
            df_ref.to_csv(INPUT_REF_PREPROC_FILE, index=False, encoding="utf-8")
            log.info("Preprocessed reference corpus saved to %s", INPUT_REF_PREPROC_FILE)


        model_dir = f"./data/models/{cat.replace(' ', '_')}"
        opt_base = Path(f"./data/models/{cat.replace(' ', '_')}_optimize")
        opt_results_path = opt_base / "optimization_results.json"
        selected_model_info = {}

        if OPTIMIZE:
            log.info("  [%s] Optimising num_topics over %s -> %s", cat, list(OPTIMIZE_RANGE), opt_base)
            opt = LDATopicModel.optimize_num_topics(
                data=df_corpus_cat.to_dict("records"),
                base_path=opt_base,
                topic_range=OPTIMIZE_RANGE,
                num_iters=LDA_ITERS,
                alpha=0.1,
                eta=0.01,
                preprocessor=preprocessor,
                reference_corpus=df_ref["text"].tolist(),
                logger=log,
            )
            best = opt["selected"][0]
            log.info(
                "  [%s] Best k=%d (coherence=%.4f) -> %s",
                cat, best["k"], best["mean_coherence"], best["model_path"],
            )
            selected_model_info = {
                "model_path": best["model_path"],
                "k": best["k"],
                "mean_coherence": best["mean_coherence"],
            }
            lda = LDATopicModel.load(best["model_path"], corpus=df_corpus_cat)
            lda.llm_provider   = LLM_PROVIDER
            lda.llm_model_type = LLM_MODEL
            lda.llm_api_key    = LLM_API_KEY
            log.info("  [%s] Generating topic labels for best model (k=%d)...", cat, best["k"])
            tm_opt = lda.tm
            results = tm_opt.generate_topic_outputs(task="label", topn=3)
            tm_opt._tpc_labels = [lbl for _, lbl in sorted(results)]
            (Path(best["model_path"]) / "TMmodel" / "tpc_labels.txt").write_text(
                "\n".join(tm_opt._tpc_labels), encoding="utf-8"
            )
        elif not RETRAIN and not Path(model_dir).exists() and opt_results_path.exists():
            # Models were previously trained with OPTIMIZE=True — load the saved best.
            with opt_results_path.open(encoding="utf-8") as f:
                opt = json.load(f)
            best = opt["selected"][0]
            log.info(
                "  [%s] Loading previously-optimized model (k=%d) from %s",
                cat, best["k"], best["model_path"],
            )
            selected_model_info = {
                "model_path": best["model_path"],
                "k": best["k"],
                "mean_coherence": best["mean_coherence"],
            }
            lda = LDATopicModel.load(best["model_path"], corpus=df_corpus_cat)
        elif RETRAIN or not Path(model_dir).exists():
            log.info("  [%s] Training LDA (%d topics, %d iters)...", cat, N_TOPICS, LDA_ITERS)
            lda = LDATopicModel(
                model_dir,
                num_topics=N_TOPICS,
                num_iters=LDA_ITERS,
                alpha=0.1,
                eta=0.01,
                preprocessor=preprocessor,
                do_labeller=False,
                do_summarizer=False,
                llm_provider=LLM_PROVIDER,
                llm_model_type=LLM_MODEL,
                llm_api_key=LLM_API_KEY,
            )
            _, elapsed = lda.train(df_corpus_cat.to_dict("records"))
            log.info("  [%s] LDA done in %.1f min", cat, elapsed / 60)
        else:
            log.info("  [%s] Loading existing model from %s", cat, model_dir)
            lda = LDATopicModel.load(model_dir, corpus=df_corpus_cat)

        queries_ordered, query_ids_ordered = lda.get_ordered_corpus()
        n = len(queries_ordered)
        if n < 2:
            log.warning("  Skipping '%s' — fewer than 2 docs after preprocessing", cat)
            continue

        X_cat = lda.get_thetas()
        topic_keys = lda.get_topic_keys()
        tm = lda.tm
        tm.load_tpc_labels()
        llm_labels = getattr(tm, "_tpc_labels", None)
        using_llm  = bool(llm_labels and not all(l.startswith("Topic ") for l in llm_labels))
        topic_labels = llm_labels if using_llm else [resolve_topic_label(i, None, topic_keys) for i in range(len(topic_keys))]
        log.info("  [%s] Using %s topic labels", cat, "LLM-generated" if using_llm else "keyword-based")
        log.info("  [%s] Thetas: %s * vocab: %d words", cat, X_cat.shape, len(set(w for kw in topic_keys for w in kw)))

        log.info("  [%s] Building hierarchy (%d docs)...", cat, n)
        t0 = time.time()

        # Filter out OOV before clustering.
        # These have max(theta) < OOV_THRESHOLD (the LDA prior gives them a
        # perfectly flat distribution across all topics), which makes their
        # pairwise Bhattacharyya distance exactly 0.
        OOV_THRESHOLD = 0.15
        oov_mask = X_cat.max(axis=1) < OOV_THRESHOLD
        n_oov = int(oov_mask.sum())
        if n_oov > 0:
            log.info(
                "  [%s] Filtering %d OOV queries (%.1f%%) with max_theta < %.2f before clustering",
                cat, n_oov, 100 * n_oov / n, OOV_THRESHOLD,
            )
        valid_local = [i for i in range(n) if not oov_mask[i]]
        X_valid = X_cat[valid_local]

        flat_nodes, root_id, Z = build_flat_tree(
            X_valid, valid_local, queries_ordered,
            linkage_method=LINKAGE_METHOD, n_repr=N_REPR_QUERIES,
            X_full=X_cat,
        )
        min_d, max_d = float(Z[:, 2].min()), float(Z[:, 2].max())
        log.info("  [%s] Hierarchy done in %.1fs * %d nodes (%d valid docs)", cat, time.time() - t0, len(flat_nodes), len(valid_local))

        # cut levels
        log.info("  [%s] Computing %d cut levels...", cat, N_CUT_LEVELS)
        cuts = []
        for d in np.linspace(min_d * 0.99, max_d * 1.01, N_CUT_LEVELS):
            assignment = fcluster(Z, t=d, criterion="distance").tolist()
            cuts.append({
                "distance": round(float(d), 4),
                "n_clusters": len(set(assignment)),
                "assignment": assignment,
            })
        log.info("  [%s] Cuts: %d->%d clusters across levels", cat, cuts[0]["n_clusters"], cuts[-1]["n_clusters"])

        tm._compute_s3()
        s3_mat = tm._s3.toarray() if tm._s3 is not None else None
        top_docs = top_docs_per_topic(X_cat, queries_ordered, topn=TOP_DOCS_PER_TOPIC, s3=s3_mat)

        alphas = tm.get_alphas().tolist()
        tpc_coords = load_tm_coords(Path(lda.model_path) / "TMmodel", len(alphas))

        data_by_category[cat] = {
            "queries": queries_ordered,
            "query_ids": query_ids_ordered,
            "topic_keys": topic_keys,
            "topic_labels": topic_labels,
            "thetas": [[round(float(v), 4) for v in row] for row in X_cat],
            "alphas": [round(float(a), 6) for a in alphas],
            "tpc_coords": [[round(c, 4) for c in xy] for xy in tpc_coords],
            "top_docs": top_docs,
            "n": n,
            "model_info": {
                "model_path": (
                    lda.model_path.relative_to(Path(__file__).parent)
                    if lda.model_path.is_absolute()
                    else lda.model_path
                ).as_posix(),
                "n_topics": lda.num_topics,
                **selected_model_info,
            },
        }
        trees_by_category[cat] = {
            "nodes": flat_nodes,
            "root_id": root_id,
            "indices": valid_local,
            "cuts": cuts,
            "min_dist": round(min_d, 4),
            "max_dist": round(max_d, 4),
            "n": len(valid_local),
        }
        log.info("  [%s] Done", cat)

    log.info("All categories processed in %.1f min", (time.time() - t_total) / 60)

    # save output 
    payload = {
        "categories": categories,
        "data_by_category": data_by_category,
        "trees_by_category": trees_by_category,
        "n_repr": N_REPR_QUERIES,
    }

    log.info("Saving JSON to %s", OUTPUT_JSON)
    save_json(payload, OUTPUT_JSON)
    log.info("Done * %s", OUTPUT_JSON)