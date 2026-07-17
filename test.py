import json, numpy as np
from safe_spoon.clustering.annotation_units import build_unit_tree, compute_leaf_indices

with open("data/output/viz_v5_data.json") as f:
    d = json.load(f)

cat = "Economic and Financial"
nodes = d["trees_by_category"][cat]["nodes"]
root_id = str(d["trees_by_category"][cat]["root_id"])
thetas = np.array(d["data_by_category"][cat]["thetas"])
queries = d["data_by_category"][cat]["queries"]

nodes_by_id = {str(n["id"]): n for n in nodes}
leaf_indices = compute_leaf_indices(nodes_by_id, root_id)

tree, _ = build_unit_tree(
    nodes_by_id, root_id, leaf_indices, thetas,
    min_size=20, max_rel_dist=0.80,
)

def collect_units(node, acc=None):
    if acc is None: acc = []
    if node.get("is_unit"): acc.append(node)
    for c in node.get("children", []): collect_units(c, acc)
    return acc

units = collect_units(tree)
sizes = np.array([u["size"] for u in units])

# Find the unit with ~195 queries
target = [u for u in units if 190 <= u["size"] <= 200]
for u in target:
    print(f"node_id: {u['node_id']}, size: {u['size']}, "
          f"stop_reason: {u['stop_reason']}, "
          f"rel_dist: {u.get('rel_dist')}, "
          f"intra_sim: {u.get('intra_sim')}")
    print(f"repr queries:")
    for i in u["repr"][:5]:
        print(f"  {queries[i][:80]}")
        
for ms in [5, 10, 15, 20]:
    tree, n = build_unit_tree(
        nodes_by_id, root_id, leaf_indices, thetas,
        min_size=ms, max_rel_dist=0.80,
    )
    units = collect_units(tree)
    sizes = np.array([u["size"] for u in units])
    # Find what inner_26347 splits into
    career = [u for u in units if 50 <= u["size"] <= 200 
              and any("career" in queries[i].lower() or "freelan" in queries[i].lower() 
                      for i in u["repr"][:3])]
    print(f"min_size={ms}: {n} total units, "
          f"{len(career)} career-related units of size "
          f"{[u['size'] for u in career]}")
    
    tree, _ = build_unit_tree(
    nodes_by_id, root_id, leaf_indices, thetas,
    min_size=10, max_rel_dist=0.80,
)
units = collect_units(tree)
career = [u for u in units if u["size"] in [74, 109]]
for u in career:
    print(f"\n--- size={u['size']}, intra_sim={u.get('intra_sim'):.3f} ---")
    for i in u["repr"][:6]:
        print(f"  {queries[i][:90]}")