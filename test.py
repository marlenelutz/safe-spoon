import json, numpy as np

with open("data/output/viz_v5_data.json") as f:
    data = json.load(f)

cat = "Economic and Financial"
nodes = data["trees_by_category"][cat]["nodes"]
root  = data["trees_by_category"][cat]["root_id"]
max_d = data["trees_by_category"][cat]["max_dist"]

inner = [n for n in nodes if n.get("children_ids")]
dists = np.array([n["dist"] / max_d for n in inner])
sizes = np.array([n["size"] for n in inner])

print("Relative distance distribution:")
for p in [10, 25, 50, 75, 90, 99]:
    print(f"  p{p}: {np.percentile(dists, p):.3f}")

# Balance distribution
balances = []
nodes_by_id = {str(n["id"]): n for n in nodes}
for n in inner:
    cids = n["children_ids"]
    if len(cids) == 2:
        s0 = nodes_by_id[str(cids[0])]["size"]
        s1 = nodes_by_id[str(cids[1])]["size"]
        balances.append(min(s0,s1)/max(s0,s1))
balances = np.array(balances)
print("\nMerge balance distribution:")
for p in [10, 25, 50, 75, 90]:
    print(f"  p{p}: {np.percentile(balances, p):.3f}")
print(f"  Fraction with balance < 0.1 (caterpillar): {(balances < 0.1).mean():.1%}")