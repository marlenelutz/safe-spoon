import logging
import os
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.cluster.hierarchy import fcluster, linkage, to_tree
from scipy.spatial.distance import squareform

from safe_spoon.clustering import bhattacharyya_matrix, build_tree, flatten_tree
from safe_spoon.preprocessing import SimpleTMPreprocessor
from safe_spoon.topic_modeling import LDATopicModel
from safe_spoon.visualization import render_html, save_json

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# config
INPUT_FILE = "data/high_risk_automatically_labelled_filtered.csv"
CONTENT_COL = "content"
LABEL_COL = "high_risk_label"
TEMPLATE_FILE = "static/viz_v5_template.html"
OUTPUT_FILE = "data/output/viz_v5_final.html"
OUTPUT_JSON = "data/output/viz_v5_data.json"

CATEGORIES = ['Economic and Financial', 'Health', 'Moral Values and Religion']

LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")

N_TOPICS = 30
LDA_ITERS = 500
N_REPR_QUERIES = 5
N_CUT_LEVELS = 40
LINKAGE_METHOD = "average"


if __name__ == "__main__":
    log.info("Loading data from %s", INPUT_FILE)
    df = pd.read_csv(INPUT_FILE, encoding="latin-1")
    df = df[df.factual_analytical_label == "Analytical"].reset_index(drop=True)
    queries = df[CONTENT_COL].fillna("").tolist()
    labels = df[LABEL_COL].fillna("unknown").tolist()

    all_categories = sorted(set(labels))
    categories = [c for c in CATEGORIES if c in all_categories] if CATEGORIES else all_categories
    log.info("Loaded %d queries · %d categories: %s", len(queries), len(categories), categories)

    log.info("Initialising spaCy preprocessor (model=en_core_web_lg)")
    preprocessor = SimpleTMPreprocessor(spacy_model="en_core_web_lg", min_df=10, max_df=0.6)

    data_by_category = {}
    trees_by_category = {}
    t_total = time.time()

    for cat_idx, cat in enumerate(categories, 1):
        cat_queries = [q for q, l in zip(queries, labels) if l == cat]
        n = len(cat_queries)
        log.info("── Category %d/%d: '%s' (%d queries)", cat_idx, len(categories), cat, n)

        if n < 2:
            log.warning("  Skipping '%s' — fewer than 2 queries", cat)
            continue

        # train LDA
        log.info("  [%s] Training LDA (%d topics, %d iters)…", cat, N_TOPICS, LDA_ITERS)
        lda = LDATopicModel(
            f"./data/models/{cat.replace(' ', '_')}",
            num_topics=N_TOPICS,
            num_iters=LDA_ITERS,
            alpha=0.1,
            eta=0.01,
            preprocessor=preprocessor,
            do_labeller=True,          
            do_summarizer=False,        
            llm_provider="openai",      
            llm_model_type="gpt-5-nano",    
            #llm_server="http://kumo.tsc.uc3m.es:11434",           
            llm_api_key=LLM_API_KEY,
            
        )
        tm, elapsed = lda.train([{"id": i, "text": t} for i, t in enumerate(cat_queries)])
        log.info("  [%s] LDA done in %.1f min", cat, elapsed / 60)

        X_cat = lda.get_thetas()
        topic_keys = lda.get_topic_keys()
        llm_labels = getattr(tm, "_tpc_labels", None)
        if llm_labels and not all(l.startswith("Topic ") for l in llm_labels):
            topic_labels = llm_labels
            log.info("  [%s] Using LLM-generated topic labels", cat)
        else:
            topic_labels = [" · ".join(kw[:3]) for kw in topic_keys]
            log.info("  [%s] Using keyword-based topic labels (LLM labels unavailable)", cat)
        log.info("  [%s] Thetas: %s · vocab: %d words", cat, X_cat.shape, len(set(w for kw in topic_keys for w in kw)))

        # Bhattacharyya distance matrix 
        log.info("  [%s] Computing Bhattacharyya distance matrix (%dx%d)…", cat, n, n)
        t0 = time.time()
        local_indices = list(range(n))
        D = bhattacharyya_matrix(X_cat)
        D_cond = squareform(D, checks=False)
        log.info("  [%s] Distance matrix done in %.1fs", cat, time.time() - t0)

        # hierarchical clustering 
        log.info("  [%s] Agglomerative clustering (method=%s)…", cat, LINKAGE_METHOD)
        Z = linkage(D_cond, method=LINKAGE_METHOD)
        min_d = float(Z[:, 2].min())
        max_d = float(Z[:, 2].max())
        log.info("  [%s] Linkage done · dist range [%.4f, %.4f]", cat, min_d, max_d)

        # cut levels
        log.info("  [%s] Computing %d cut levels…", cat, N_CUT_LEVELS)
        cuts = []
        for d in np.linspace(min_d * 0.99, max_d * 1.01, N_CUT_LEVELS):
            assignment = fcluster(Z, t=d, criterion="distance").tolist()
            cuts.append({
                "distance": round(float(d), 4),
                "n_clusters": len(set(assignment)),
                "assignment": assignment,
            })
        log.info("  [%s] Cuts: %d→%d clusters across levels", cat, cuts[0]["n_clusters"], cuts[-1]["n_clusters"])

        # tree building
        log.info("  [%s] Building dendrogram tree…", cat)
        root_node, _ = to_tree(Z, rd=True)
        tree = build_tree(root_node, local_indices, X_cat, cat_queries, n_repr=N_REPR_QUERIES)
        flat_nodes, root_id = flatten_tree(tree)
        log.info("  [%s] Tree built · %d nodes", cat, len(flat_nodes))

        data_by_category[cat] = {
            "queries": cat_queries,
            "topic_keys": topic_keys,
            "topic_labels": topic_labels,
            "thetas": [[round(float(v), 4) for v in row] for row in X_cat],
            "n": n,
        }
        trees_by_category[cat] = {
            "nodes": flat_nodes,
            "root_id": root_id,
            "indices": local_indices,
            "cuts": cuts,
            "min_dist": round(min_d, 4),
            "max_dist": round(max_d, 4),
            "n": n,
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
    log.info("Rendering HTML to %s", OUTPUT_FILE)
    render_html(payload, TEMPLATE_FILE, OUTPUT_FILE)
    log.info("Done · %s + %s", OUTPUT_FILE, OUTPUT_JSON)
