import re

import pandas as pd

from safe_spoon.utils.common import load_yaml_config_file
from safe_spoon.utils.data_utils import (
    DEMO_PATTERNS,
    DEMO_REGEX,
    NAME_PATTERN,
    near_duplicate_pairs,
    remove_contained_in,
    remove_empties,
    remove_exact_duplicates,
    remove_name_pattern,
    remove_near_duplicates,
    #scan_near_duplicate_thresholds,
)

_cfg = load_yaml_config_file()

INPUT_FILE = "data/dataset/automatically-labeled-data/high_risk_automatically_labelled_filtered.csv"
OUTPUT_FILE = _cfg["input_file"]
OUTPUT_DUPLICATES = "data/near_duplicate_pairs.csv"
OUTPUT_THRESHOLD_SCAN = "data/near_duplicate_threshold_scan.csv"
CONTENT_COL = _cfg["content_col"]
LABEL_COL = _cfg["label_col"]
CATEGORIES = _cfg["categories"]

SIMILARITY_THRESHOLD = _cfg["similarity_threshold"]
THRESHOLD_SCAN_VALUES = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98]

N_PAIR_EXAMPLES = 30

def _print_pairs(pairs: list, n: int = N_PAIR_EXAMPLES) -> None:
    """Print the first n pairs of kept and removed rows, along with their similarity scores if available."""
    for kept_id, removed_id, kept_text, removed_text, *rest in pairs[:n]:
        sim = rest[0] if rest else None
        sim_str = f"  sim={sim:.3f}" if sim is not None else ""
        print(f"    KEPT    [id={kept_id}]: {kept_text[:120].replace(chr(10), ' ')!r}")
        print(f"    REMOVED [id={removed_id}]{sim_str}: {removed_text[:120].replace(chr(10), ' ')!r}")
        print()


def print_filter_stats(stage: str, df_before: pd.DataFrame, df_after: pd.DataFrame) -> None:
    """Print statistics about the filtering stage, including how many rows were removed and their label distribution."""
    removed = df_before[~df_before.index.isin(df_after.index)]
    counts = removed.groupby(LABEL_COL).size()
    total = len(df_before) - len(df_after)
    print(f"\n[{stage}] Removed {total} rows:")
    if counts.empty:
        print("  (none)")
    else:
        for label, n in counts.items():
            print(f"  {label}: {n}")

df = pd.read_csv(INPUT_FILE, encoding="latin-1")
df = df[df.factual_analytical_label == "Analytical"].reset_index(drop=True)

# keep only the rows with labels in CATEGORIES 
df = df[df[LABEL_COL].isin(CATEGORIES)].reset_index(drop=True)

df[CONTENT_COL] = df[CONTENT_COL].fillna("")
df[LABEL_COL] = df[LABEL_COL].fillna("unknown")

print(f"Initial rows: {len(df)}")
print("Label distribution:")
print(df[LABEL_COL].value_counts().to_string())

# Remove empties
df1 = remove_empties(df, CONTENT_COL)
print_filter_stats("Remove empties", df, df1)

# Remove exact duplicates (case-insensitive)
df1r = df1.reset_index(drop=True)
df2, exact_pairs = remove_exact_duplicates(df1r, CONTENT_COL)
print_filter_stats("Remove exact duplicates", df1r, df2)
print(f"  Examples of exact duplicate pairs:")
_print_pairs(exact_pairs)

# Remove NAME_X pattern
df3 = remove_name_pattern(df2.reset_index(drop=True), CONTENT_COL, NAME_PATTERN)
print_filter_stats("Remove NAME_X pattern", df2.reset_index(drop=True), df3)

# Remove rows with explicit demographic self-disclosure (age, gender, religion, ethnicity, etc.)
df2_before_demo = df3.reset_index(drop=True)
df3 = remove_name_pattern(df2_before_demo, CONTENT_COL, DEMO_REGEX)
print_filter_stats("Remove demographic self-disclosure", df2_before_demo, df3)
removed_demo = df2_before_demo[~df2_before_demo.index.isin(df3.index)]
print(f"  Examples of removed demographic rows:")
for idx, row in removed_demo.head(N_PAIR_EXAMPLES).iterrows():
    text = row[CONTENT_COL]
    categories = ", ".join(name for name, pat in DEMO_PATTERNS.items() if re.search(pat, text, re.IGNORECASE))
    print(f"    REMOVED [id={idx}] ({categories}): {text[:120].replace(chr(10), ' ')!r}")

# remove contained in rows
df3_before_contained = df3.reset_index(drop=True)
df3, contained_pairs = remove_contained_in(df3_before_contained, CONTENT_COL)
print_filter_stats("Remove contained-in rows", df3_before_contained, df3)
print(f"  Examples of contained-in pairs:")
_print_pairs(contained_pairs)

# Scan candidate thresholds to help pick SIMILARITY_THRESHOLD
df3r = df3.reset_index(drop=True)
all_pairs_df = near_duplicate_pairs(df3r, CONTENT_COL, min_similarity=min(THRESHOLD_SCAN_VALUES))
#scan_df = scan_near_duplicate_thresholds(all_pairs_df, THRESHOLD_SCAN_VALUES)
#print(f"\n[Threshold scan] Near-duplicate pair counts by candidate threshold:")
#print(scan_df.to_string(index=False))
#all_pairs_df.to_csv(OUTPUT_THRESHOLD_SCAN, index=False)
#print(f"  Saved all pairwise similarities to {OUTPUT_THRESHOLD_SCAN} for manual inspection")

# Remove near-duplicates
df4, near_pairs = remove_near_duplicates(df3r, CONTENT_COL, SIMILARITY_THRESHOLD)
print_filter_stats(f"Remove near-duplicates (>= {SIMILARITY_THRESHOLD * 100:.0f}%)", df3r, df4)
print(f"  Examples of near-duplicate pairs:")
_print_pairs(near_pairs)

# save the near-duplicate pairs with their Ids and similarity scores to a CSV file
pairs_df = pd.DataFrame(near_pairs, columns=["kept_id", "removed_id", "kept_text", "removed_text", "similarity"])
pairs_df.to_csv(OUTPUT_DUPLICATES, index=False)
print(f"\nSaved near-duplicate pairs to {OUTPUT_DUPLICATES}")

print(f"\nFinal rows: {len(df4)}")
print("Final label distribution:")
print(df4[LABEL_COL].value_counts().to_string())

df4.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved to {OUTPUT_FILE}")