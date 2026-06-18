import json
import hashlib
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
import numpy as np
import pandas as pd
import spacy
import tomotopy as tp
from scipy import sparse
from scipy.cluster.hierarchy import linkage, fcluster, to_tree
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

# config
INPUT_FILE = "data/en_WildChat_history.csv"
CONTENT_COL = "content"
LABEL_COL = "high_risk_label"
TEMPLATE_FILE = "static/viz_v5_template.html"
OUTPUT_FILE = "data/output/viz_v5_final.html"
OUTPUT_JSON = "data/output/viz_v5_data.json"

SPACY_MODEL = "en_core_web_sm"
LINKAGE_METHOD = "average"
N_CUT_LEVELS = 40
N_REPR_QUERIES = 5
MAX_MEDOID = 200

N_TOPICS = 30
LDA_ALPHA = 0.1
LDA_ETA = 0.01
LDA_ITERS = 500
LDA_INTERVAL = 10
THETAS_THR = 0.01
MIN_DOC_WORDS = 3
PREPROCESS_MIN_DF = 2
PREPROCESS_MAX_DF = 0.95
FORCE_RETRAIN = True

CACHE_DIR = Path("data/output/cache")
MODEL_CACHE_FILE = CACHE_DIR / "lda_model.bin"
LEMMAS_CACHE_FILE = CACHE_DIR / "lemmas.json"
TOPIC_KEYS_CACHE_FILE = CACHE_DIR / "topic_keys.json"
THETAS_CACHE_FILE = CACHE_DIR / "thetas.npy"
CACHE_META_FILE = CACHE_DIR / "meta.json"


# for saving lda models and preproc
def _dataset_signature(queries: List[str], labels: List[str]) -> str:
    payload = json.dumps(
        {"queries": queries, "labels": labels},
        ensure_ascii=False, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_params() -> Dict:
    return {
        "input_file": INPUT_FILE, "spacy_model": SPACY_MODEL,
        "n_topics": N_TOPICS, "lda_alpha": LDA_ALPHA, "lda_eta": LDA_ETA,
        "lda_iters": LDA_ITERS, "lda_interval": LDA_INTERVAL,
        "thetas_thr": THETAS_THR, "min_doc_words": MIN_DOC_WORDS,
        "preprocess_min_df": PREPROCESS_MIN_DF,
        "preprocess_max_df": PREPROCESS_MAX_DF,
    }

# preprocessing


class SimpleTMPreprocessor:
    def __init__(
        self, *, spacy_model="en_core_web_sm", spacy_disable=None,
        valid_pos=None, stopword_files=None, equivalents_files=None,
        min_df=2, max_df=0.95, max_features=None, logger=None,
    ):
        self._logger = logger or logging.getLogger(__name__)
        self._logger.info("Loading spaCy model %r", spacy_model)
        disable = spacy_disable or []
        try:
            self.nlp = spacy.load(spacy_model, disable=disable)
        except OSError as e:
            raise OSError(
                f"spaCy model '{spacy_model}' not found. "
                f"Install: python -m spacy download {spacy_model}"
            ) from e
        self.valid_pos = set(valid_pos) if valid_pos else {
            "NOUN", "VERB", "ADJ", "PROPN"}
        self.stopwords = {w.lower() for w in self.nlp.Defaults.stop_words}
        stopword_files = [os.path.join("data/stops", f) for f in os.listdir(
            "data/stops")] if stopword_files is None else stopword_files
        self.stopwords |= self._load_stopwords(stopword_files or [])
        self.equivalents = self._load_equivalents(equivalents_files or [])
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self._cv = None
        self._tfidf = None

    @staticmethod
    def _load_stopwords(files):
        out = set()
        for f in files:
            p = Path(f)

            if not p.exists():
                continue
            this_file_words = [w.strip().lower()
                               for w in p.open("r", encoding="utf8").readlines()]
            out |= set(this_file_words)
        return out

    @staticmethod
    def _load_equivalents(files):
        eq = {}
        for f in files:
            p = Path(f)
            if not p.exists():
                continue
            lines = json.load(p.open("r", encoding="utf8")).get("wordlist", []) if p.suffix.lower() == ".json" \
                else p.open("r", encoding="utf8").readlines()
            for line in lines:
                line = str(line).strip()
                if ":" in line:
                    a, b = line.split(":", 1)
                    a, b = a.strip(), b.strip()
                    if a and b:
                        eq[a] = b
        return eq

    def _lemmatize(self, text):
        toks = [self.equivalents.get(t, t) for t in text.lower().split()]
        doc = self.nlp(" ".join(toks))
        sw, vp = self.stopwords, self.valid_pos
        return [t.lemma_.lower() for t in doc
                if t.is_alpha and t.pos_ in vp
                and not t.is_stop and t.lemma_.lower() not in sw and len(t.lemma_) > 3]

    def fit_transform(self, df, text_col="text", id_col="id",
                      compute_bow=True, compute_tfidf=True):
        out = pd.DataFrame()
        out["id"] = df[id_col].values if id_col in df.columns else range(
            len(df))
        out["text"] = df[text_col].fillna("").astype(str)
        out["lemmas"] = [self._lemmatize(t) for t in out["text"].tolist()]
        if compute_bow:
            self._cv = CountVectorizer(
                tokenizer=lambda x: x, preprocessor=lambda x: x, lowercase=False,
                min_df=self.min_df, max_df=self.max_df,
                max_features=self.max_features, token_pattern=None,
            )
            X_bow = self._cv.fit_transform(out["lemmas"].tolist())
            out["bow"] = [X_bow.getrow(i) for i in range(X_bow.shape[0])]
        if compute_tfidf:
            if self._cv is None:
                raise RuntimeError(
                    "compute_bow must be True before compute_tfidf.")
            X_bow2 = sparse.vstack(out["bow"].tolist())
            self._tfidf = TfidfTransformer(norm="l2", use_idf=True).fit(X_bow2)
            X_tfidf = self._tfidf.transform(X_bow2)
            out["tfidf"] = [X_tfidf.getrow(i) for i in range(X_tfidf.shape[0])]
        return out


def bhattacharyya_matrix(X: np.ndarray) -> np.ndarray:
    """
    Compute full n×n Bhattacharyya distance matrix in one matrix multiply.
    BC(p,q) = -ln( sum(sqrt(p_i * q_i)) )
    X rows must be non-negative (LDA thetas already satisfy this).
    """
    eps = 1e-10
    X_sqrt = np.sqrt(X + eps)
    BC = X_sqrt @ X_sqrt.T          # BC coefficient for every pair
    BC = np.clip(BC, eps, 1.0)
    D = -np.log(BC)                # Bhattacharyya distance
    np.fill_diagonal(D, 0.0)
    return D

# tree helpers


def most_representative(indices, X, n=N_REPR_QUERIES):
    """Return the indices of the n most representative queries among the given indices, based on:

    1) if the cluster has less than or equal to n queries, return all of them;
    2) otherwise, return the n medoids: the actual cluster members with the lowest mean Bhattacharyya distance to the rest of the cluster.
    """

    if len(indices) <= n:
        return list(indices)
    eps = 1e-10
    vecs = X[indices]
    if len(indices) <= MAX_MEDOID:
        sq = np.sqrt(vecs + eps)
        D = -np.log(np.clip(sq @ sq.T, eps, 1.0))
        scores = D.mean(axis=1)
    else:
        centroid = vecs.mean(axis=0)
        scores = -np.log(np.clip(np.sqrt(vecs + eps) @ np.sqrt(centroid + eps), eps, 1.0))
    return [indices[i] for i in np.argsort(scores)[:n]]


def build_tree(root_node, global_indices, X, queries):
    """
    Build a nested dict tree from a scipy ClusterNode, enriched with metadata for visualization.
    """

    stack = [(root_node, None, False)]
    ordered = []   # nodos in post-order (leaves primero)
    node_dict = {}  # scipy node id -> dict

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
            stack.append((node.left,  node.id, False))

    for scipy_id in reversed(ordered):
        d = node_dict[scipy_id]

        if not d["children"] and "idx" in d:
            #  leave is comlete
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
        d["repr"] = most_representative(all_idxs, X)

    # limpiar campos internos de navegación
    for d in node_dict.values():
        for k in ["_parent_id", "_is_right", "_scipy_id", "_left_id", "_right_id"]:
            d.pop(k, None)

    return node_dict[root_node.id]


def _gather_leaf_indices(node):
    """Recursively gather global indices of all leaf nodes under this node."""
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
    """Convert the nested tree dict to flat node list."""
    flat = []
    stack = [root]
    while stack:
        node = stack.pop()
        flat_node = {k: v for k, v in node.items() if k != "children"}
        flat_node["children_ids"] = [c["id"] for c in node.get("children", [])]
        flat.append(flat_node)
        stack.extend(node.get("children", []))
    return flat, root["id"]


# load data
logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s %(name)s: %(message)s")

df = pd.read_csv(INPUT_FILE)
df = df[df.factual_analytical_label == "Analytical"].reset_index(drop=True)
queries = df[CONTENT_COL].fillna("").tolist()
labels = df[LABEL_COL].fillna("unknown").tolist()
categories = sorted(set(labels))
n = len(queries)
print(f"Loaded {n} queries · {len(categories)} categories: {categories}")

# lda
cache_signature = _dataset_signature(queries, labels)
cache_params = _cache_params()
lemma_lists: List[List[str]] = []
topic_keys:  List[List[str]] = []
X_all: Optional[np.ndarray] = None

cache_files = [MODEL_CACHE_FILE, LEMMAS_CACHE_FILE,
               TOPIC_KEYS_CACHE_FILE, THETAS_CACHE_FILE, CACHE_META_FILE]

if not FORCE_RETRAIN and all(p.exists() for p in cache_files):
    try:
        meta = json.loads(CACHE_META_FILE.read_text("utf-8"))
        if meta.get("dataset_signature") == cache_signature \
                and meta.get("params") == cache_params:
            lemma_lists = json.loads(LEMMAS_CACHE_FILE.read_text("utf-8"))
            topic_keys = json.loads(TOPIC_KEYS_CACHE_FILE.read_text("utf-8"))
            X_all = np.load(THETAS_CACHE_FILE)
            assert len(lemma_lists) == n, "cached lemmas length mismatch"
            assert X_all.shape == (n, N_TOPICS), "cached theta shape mismatch"
            print("Loaded cached LDA artifacts.")
        else:
            print("Cache incompatible with current config — recomputing.")
    except Exception as exc:
        print(f"Cache load failed ({exc}) — recomputing.")

if X_all is None:
    preprocessor = SimpleTMPreprocessor(
        spacy_model=SPACY_MODEL,
        min_df=PREPROCESS_MIN_DF, max_df=PREPROCESS_MAX_DF,
    )
    proc_df = preprocessor.fit_transform(df, text_col=CONTENT_COL,
                                         compute_bow=True, compute_tfidf=False)
    lemma_lists = proc_df["lemmas"].tolist()

    trainable_mask = np.array([len(l) >= MIN_DOC_WORDS for l in lemma_lists])
    train_idx = np.where(trainable_mask)[0]
    short_idx = np.where(~trainable_mask)[0]
    print(f"Trainable: {len(train_idx)}  Short: {len(short_idx)}")

    print("Training LDA…")
    lda = tp.LDAModel(k=N_TOPICS, tw=tp.TermWeight.ONE,
                      alpha=LDA_ALPHA, eta=LDA_ETA)
    for i in train_idx:
        lda.add_doc(lemma_lists[i])
    for i in range(0, LDA_ITERS, LDA_INTERVAL):
        lda.train(LDA_INTERVAL)
        if (i // LDA_INTERVAL) % 10 == 0:
            print(
                f"  iter {i:4d}/{LDA_ITERS}  ll={lda.ll_per_word:.4f}  perp={lda.perplexity:.1f}")

    thetas_train = np.array([d.get_topic_dist() for d in lda.docs])
    thetas_train[thetas_train < THETAS_THR] = 0
    thetas_train = normalize(thetas_train, axis=1, norm="l1")

    topic_keys = [[w for w, _ in lda.get_topic_words(
        k, 10)] for k in range(N_TOPICS)]
    print(f"LDA done. Thetas: {thetas_train.shape}")

    def _kw_theta(tokens, keys, k):
        scores = np.array([len(set(tokens) & set(kw))
                          for kw in keys], dtype=float)
        if scores.sum() == 0:
            scores = np.ones(k)
        return scores / scores.sum()

    X_all = np.empty((n, N_TOPICS))
    X_all[train_idx] = thetas_train
    if len(short_idx):
        X_all[short_idx] = np.vstack([_kw_theta(lemma_lists[i], topic_keys, N_TOPICS)
                                      for i in short_idx])

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    lda.save(str(MODEL_CACHE_FILE))
    LEMMAS_CACHE_FILE.write_text(json.dumps(
        lemma_lists, ensure_ascii=False), "utf-8")
    TOPIC_KEYS_CACHE_FILE.write_text(json.dumps(
        topic_keys, ensure_ascii=False), "utf-8")
    np.save(THETAS_CACHE_FILE, X_all)
    CACHE_META_FILE.write_text(json.dumps({
        "dataset_signature": cache_signature,
        "params": cache_params,
        "n_samples": n,
    }, indent=2), "utf-8")
    print(f"Cache saved to {CACHE_DIR}")

# agglomerative clustering + tree building
trees_by_category = {}

for cat in categories:
    cat_indices = [i for i, l in enumerate(labels) if l == cat]
    cat_n = len(cat_indices)
    print(f"\n  [{cat}] {cat_n} queries")
    if cat_n < 2:
        print("    Skipping — fewer than 2 queries")
        continue

    X_cat = X_all[cat_indices]

    D_full = bhattacharyya_matrix(X_cat)
    D_cond = squareform(D_full, checks=False)
    Z = linkage(D_cond, method=LINKAGE_METHOD)

    min_d = float(Z[:, 2].min())
    max_d = float(Z[:, 2].max())

    cuts = []
    for d in np.linspace(min_d * 0.99, max_d * 1.01, N_CUT_LEVELS):
        assignment = fcluster(Z, t=d, criterion="distance").tolist()
        cuts.append({
            "distance":   round(float(d), 4),
            "n_clusters": len(set(assignment)),
            "assignment": assignment,
        })

    root_node, _ = to_tree(Z, rd=True)
    tree = build_tree(root_node, cat_indices, X_all, queries)

    #  flatten before storing
    flat_nodes, root_id = flatten_tree(tree)

    trees_by_category[cat] = {
        "nodes": flat_nodes,
        "root_id": root_id,
        "indices": cat_indices,
        "cuts": cuts,
        "min_dist": round(min_d, 4),
        "max_dist": round(max_d, 4),
        "n":  cat_n,
    }
    print(f"    Done · dist {min_d:.3f}–{max_d:.3f} · "
          f"{flat_nodes} nodes · "
          f"cuts: {cuts[-1]['n_clusters']}–{cuts[0]['n_clusters']} clusters")

# save
os.makedirs("data/output", exist_ok=True)

payload = {
    "categories": categories,
    "trees_by_category": trees_by_category,
    "queries": queries,
    "n_repr": N_REPR_QUERIES,
    "topic_keys": topic_keys,
    "thetas": [[round(float(v), 4) for v in row] for row in X_all],
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(payload, f, separators=(",", ":"))

with open(TEMPLATE_FILE, encoding="utf-8") as f:
    html = f.read()

payload_json = json.dumps(payload, separators=(",", ":"))
html = html.replace("PAYLOAD_PLACEHOLDER", payload_json)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(html)

size_mb = os.path.getsize(OUTPUT_JSON) / 1e6
print(f"\nSaved {OUTPUT_FILE} + {OUTPUT_JSON}  ({size_mb:.2f} MB)")
