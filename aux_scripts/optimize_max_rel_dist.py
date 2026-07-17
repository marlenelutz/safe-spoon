"""Calibration script for max_rel_dist.

Quality metric combines three signals:
  - intra-unit cohesion:   how similar queries are within each unit
  - inter-unit separation: how distinct units are from each other
  - size fitness:          how well unit sizes fall in the target range

All three are in [0, 1] and combined with configurable weights.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from safe_spoon.clustering.annotation_units import build_unit_tree, compute_leaf_indices

# config
DATA_FILE   = "data/output/viz_v5_data.json"
CATEGORY    = "Economic and Financial"
EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE  = 64
MIN_SIZE    = 20
TARGET_MIN  = 30
TARGET_MAX  = 150

# Quality weights
W_INTRA  = 0.35   # cohesion within units
W_INTER  = 0.25   # separation between units
W_SIZE   = 0.40   # fraction of units in the target size range

# Range of max_rel_dist values to evaluate
MRD_MIN  = 0.25
MRD_MAX  = 0.95
MRD_STEP = 0.025

# data
print(f"Loading {DATA_FILE}...")
with open(DATA_FILE) as f:
    data = json.load(f)

cat_data  = data["data_by_category"][CATEGORY]
cat_tree  = data["trees_by_category"][CATEGORY]

nodes    = cat_tree["nodes"]
root_id  = str(cat_tree["root_id"])
max_dist = cat_tree["max_dist"]
thetas   = np.array(cat_data["thetas"])
queries  = cat_data["queries"]

nodes_by_id  = {str(n["id"]): n for n in nodes}
leaf_indices = compute_leaf_indices(nodes_by_id, root_id)

# queries
print(f"Encoding {len(queries)} queries with {EMBED_MODEL}...")
model = SentenceTransformer(EMBED_MODEL)
E = model.encode(
    queries,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
).astype(np.float32)
print(f"Embeddings ready: {E.shape}")

# stop reason distribution and rel_dist range
print(f"\n--- Diagnostic at max_rel_dist=0.40 ---")
tree_diag, _ = build_unit_tree(
    nodes_by_id, root_id, leaf_indices, thetas,
    min_size=MIN_SIZE, max_rel_dist=0.40,
)

def collect_units(node, acc=None):
    if acc is None:
        acc = []
    if node.get("is_unit"):
        acc.append(node)
    for c in node.get("children", []):
        collect_units(c, acc)
    return acc

units_diag = collect_units(tree_diag)
reasons = {}
for u in units_diag:
    r = u.get("stop_reason", "?")
    reasons[r] = reasons.get(r, 0) + 1
print(f"Stop reasons: {reasons}")

inner = [n for n in nodes if n.get("children_ids") and n["size"] > MIN_SIZE]
dists = np.array([n["dist"] / max_dist for n in inner])
print(f"\nrel_dist for nodes with size>{MIN_SIZE} (n={len(inner)}):")
for p in [10, 25, 50, 75, 90, 99]:
    print(f"  p{p}: {np.percentile(dists, p):.3f}")
print(f"  fraction <= 0.40: {(dists <= 0.40).mean():.1%}")
print(f"  fraction <= 0.60: {(dists <= 0.60).mean():.1%}")
print(f"  fraction <= 0.80: {(dists <= 0.80).mean():.1%}")

def unit_quality(max_rel_dist: float, min_size: int = MIN_SIZE):
    unit_tree, n_units = build_unit_tree(
        nodes_by_id, root_id, leaf_indices, thetas,
        min_size=min_size, max_rel_dist=max_rel_dist,
    )
    if not unit_tree or n_units < 2:
        return None, n_units, []

    units = collect_units(unit_tree)
    sizes = []
    centroids = []

    for u in units:
        ids = leaf_indices[u["node_id"]]
        sizes.append(len(ids))
        centroids.append(E[ids].mean(axis=0))

    sizes     = np.array(sizes)
    centroids = np.array(centroids)

    # intra-unit cohesion
    intra_scores = []
    for u, ids_u in zip(units, [leaf_indices[u["node_id"]] for u in units]):
        s = u.get("intra_sim")
        if s is not None:
            intra_scores.append(s)
        else:
            vecs = E[ids_u].astype(np.float32)
            sim  = cos_sim(vecs, vecs).numpy()
            n    = len(ids_u)
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            intra_scores.append(float(sim[mask].mean()) if mask.any() else 1.0)

    mean_intra = float(np.mean(intra_scores))

    # inter-unit separation
    norms  = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10
    c_norm = centroids / norms
    sim_matrix = c_norm @ c_norm.T
    n_c  = len(centroids)
    mask = np.triu(np.ones((n_c, n_c), dtype=bool), k=1)
    mean_inter_dist = float(1.0 - sim_matrix[mask].mean()) if mask.any() else 0.0

    # size fitness
    def size_score(s):
        if TARGET_MIN <= s <= TARGET_MAX:
            return 1.0
        elif s < TARGET_MIN:
            # linearly from 1.0 at TARGET_MIN to 0.0 at TARGET_MIN/2
            floor = TARGET_MIN / 2
            return max(0.0, (s - floor) / (TARGET_MIN - floor))
        else:
            # linearly from 1.0 at TARGET_MAX to 0.0 at TARGET_MAX*2
            ceil = TARGET_MAX * 2
            return max(0.0, (ceil - s) / (TARGET_MAX))

    mean_size_fitness = float(np.mean([size_score(s) for s in sizes]))

    quality = (
        W_INTRA * mean_intra
        + W_INTER  * mean_inter_dist
        + W_SIZE   * mean_size_fitness
    )

    return quality, n_units, sizes.tolist()


# optimization loop
print(f"\n{'max_rel_dist':>14} {'n_units':>8} {'quality':>9} "
      f"{'median_size':>12} {'max_size':>9} {'pct_in_range':>13}")
print("-" * 72)

best_mrd, best_q = None, -1.0

for mrd in np.arange(MRD_MIN, MRD_MAX + MRD_STEP / 2, MRD_STEP):
    mrd = round(float(mrd), 3)
    q, n, sizes = unit_quality(mrd)

    if q is None:
        print(f"{mrd:>14.3f} {n:>8}  {'(degenerate)':>21}")
        continue

    sizes_arr= np.array(sizes)
    median_s = int(np.median(sizes_arr))
    max_s = int(sizes_arr.max())
    pct_in_range = float(np.mean(
        (sizes_arr >= TARGET_MIN) & (sizes_arr <= TARGET_MAX)
    ))

    marker = " ←" if q > best_q else ""
    print(f"{mrd:>14.3f} {n:>8} {q:>9.4f} {median_s:>12} {max_s:>9} "
          f"{pct_in_range:>12.1%}{marker}")

    if q > best_q:
        best_q = q
        best_mrd = mrd

print(f"\nBest max_rel_dist: {best_mrd}  (quality={best_q:.4f})")
print(f"Suggested config:  max_rel_dist: {best_mrd}")