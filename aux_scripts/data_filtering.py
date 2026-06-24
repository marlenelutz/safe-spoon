import re

import pandas as pd

from safe_spoon.utils.data_utils import remove_empties, remove_exact_duplicates, remove_near_duplicates


INPUT_FILE = "data/high_risk_automatically_labelled_filtered.csv"
OUTPUT_FILE = "data/high_risk_automatically_labelled_filtered_cleaned.csv"
OUTPUT_DUPLICATES = "data/near_duplicate_pairs.csv"
CONTENT_COL = "content"
LABEL_COL = "high_risk_label"
CATEGORIES = ['Economic and Financial', 'Health', 'Moral Values and Religion']

SIMILARITY_THRESHOLD = 0.95
NAME_PATTERN = re.compile(r"\bNAME_\w+\b", re.IGNORECASE)

N_PAIR_EXAMPLES = 3

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
#df3 = remove_name_pattern(df2.reset_index(drop=True), CONTENT_COL, NAME_PATTERN)
#print_filter_stats("Remove NAME_X pattern", df2.reset_index(drop=True), df3)

# Remove near-duplicates
df2r = df2.reset_index(drop=True)
df4, near_pairs = remove_near_duplicates(df2r, CONTENT_COL, SIMILARITY_THRESHOLD)
print_filter_stats(f"Remove near-duplicates (>= {SIMILARITY_THRESHOLD * 100:.0f}%)", df2r, df4)
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
