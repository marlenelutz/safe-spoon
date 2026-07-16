import re
import numpy as np
import pandas as pd
from rapidfuzz import process as rf_process  # type: ignore
from rapidfuzz.distance import Indel  # type: ignore

################################################################################
#  REGEX PATTERNS
################################################################################
NAME_PATTERN = re.compile(r"\bNAME_\w+\b", re.IGNORECASE)

DEMO_PATTERNS = {
    "age_explicit": r"\bI(?:'?m| am)\s+(?:a\s+)?\d{1,3}(?:\s*(?:years?\s*old|y\.?o\.?))",
    "age_yo": r"\bI(?:'?m| am)\s+a\s+\d{1,3}[\s-]*year[\s-]*old\b",
    "age_asa": r"\bas\s+a\s+\d{1,3}[\s-]*year[\s-]*old\b",
    "gender_identity": r"\bI(?:'?m| am)\s+(?:a\s+)?(?:woman|man|boy|girl|male|female|non-?binary|transgender|trans\s?(?:man|woman))\b",
    "self_asa_person": r"\bI(?:'?m| am)\s+a\s+person\s+who\b",
    "self_asa_demo": r"\bas\s+a\s+\d{0,3}\s*(?:year[\s-]*old\s+)?(?:woman|man|boy|girl|male|female|muslim|christian|jewish|hindu|buddhist|atheist|catholic|immigrant|refugee|disabled|black|white|asian|latino|latina|hispanic)\b",
    "nationality_from": r"\bI(?:'?m| am)\s+from\s+[A-Z][a-zA-Z]+\b",
    "living_in": r"\bI\s+live\s+in\s+[A-Z][a-zA-Z]+\b",
    "religion": r"\bI(?:'?m| am)\s+(?:a\s+)?(?:muslim|christian|jewish|hindu|buddhist|atheist|catholic)\b",
    "ethnicity": r"\bI(?:'?m| am)\s+(?:a\s+)?(?:black|white|asian|latino|latina|hispanic|african[\s-]american)\b",
    "marital": r"\bI(?:'?m| am)\s+(?:married|single|divorced|widowed)\b",
    "disability": r"\bI(?:'?m| am)\s+(?:disabled|blind|deaf)\b|\bI\s+have\s+a\s+disability\b",
    "socioeconomic": r"\bI(?:'?m| am)\s+(?:unemployed|homeless|low[\s-]income)\b|\bmy\s+family\s+is\s+poor\b",
    "orientation": r"\bI(?:'?m| am)\s+(?:gay|lesbian|bisexual|straight|heterosexual|homosexual)\b",
    "age_gender_3rd": r"\b(?:a\s+)?\d{1,3}[\s-]*year[\s-]*old\s+(?:man|woman|male|female|boy|girl|guy|lady)\b",
    "person_from_3rd": r"\b(?:a\s+|for\s+a\s+)?person\s+(?:living\s+in|from)\s+[A-Z][a-zA-Z]+\b",
}
DEMO_REGEX = re.compile(
    "|".join(f"(?:{p})" for p in DEMO_PATTERNS.values()), re.IGNORECASE)


def load_corpus_df(
    input_file: str,
    *,
    content_col: str = "content",
    label_col: str = "high_risk_label"
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Read the input CSV, keep only Analytical rows, and return raw lists.

    Parameters
    ----------
    input_file : str
        Path to the input CSV file.
    content_col : str, optional
        Name of the column containing the content strings, by default "content".
    label_col : str, optional
        Name of the column containing the label strings, by default "high_risk_label".

    Returns
    -------
    df : pd.DataFrame
        Full filtered DataFrame (may be useful for additional columns).
    queries : list[str]
        Content strings (empty strings kept; callers filter as needed).
    labels : list[str]
        Label strings aligned with queries.
    """
    df = pd.read_csv(input_file, encoding="latin-1")
    df = df[df.factual_analytical_label == "Analytical"].reset_index(drop=True)
    queries = df[content_col].fillna("").tolist()
    labels = df[label_col].fillna("unknown").tolist()
    return df, queries, labels


def corpus_for_category(
    queries: list[str],
    labels: list[str],
    category: str
) -> pd.DataFrame:
    """Return a DataFrame(id, text) for one category with empty strings removed.
    """
    pairs = [
        (i, q)
        for i, (q, l) in enumerate(zip(queries, labels))
        if l == category and q.strip()
    ]
    return pd.DataFrame({"id": [i for i, _ in pairs], "text": [q for _, q in pairs]})


def remove_empties(df: pd.DataFrame, content_col: str) -> pd.DataFrame:
    """Remove rows with empty or whitespace-only content."""
    return df[df[content_col].str.strip().astype(bool)].copy()


def remove_exact_duplicates(df: pd.DataFrame, content_col: str) -> tuple[pd.DataFrame, list]:
    """Remove exact duplicates (case-insensitive) and return the cleaned DataFrame along with pairs of kept and removed rows."""
    df = df.copy()
    df["_lower"] = df[content_col].str.lower().str.strip()
    pairs = []
    for _, group in df.groupby("_lower", sort=False):
        if len(group) > 1:
            kept = group.iloc[0]
            for _, row in group.iloc[1:].iterrows():
                pairs.append(
                    (kept.name, row.name, kept[content_col], row[content_col]))
    df = df.drop_duplicates(subset="_lower").drop(columns="_lower")
    return df, pairs


def remove_name_pattern(df: pd.DataFrame, content_col: str, name_pattern: str) -> pd.DataFrame:
    """Remove rows containing the NAME_X pattern"""
    mask = df[content_col].str.contains(name_pattern, regex=True, na=False)
    return df[~mask].copy()


def remove_contained_in(df: pd.DataFrame, content_col: str) -> tuple[pd.DataFrame, list]:
    """Remove rows whose content is contained in a longer row's content."""
    df = df.copy()
    normalized_texts = df[content_col].fillna(
        "").str.lower().str.strip().tolist()
    original_texts = df[content_col].fillna("").tolist()

    keep_mask = [True] * len(df)
    removed_pairs = []

    for i, text in enumerate(normalized_texts):
        # Keep this row unless it is contained by a strictly longer text.
        containing_candidates = [
            j for j, other in enumerate(normalized_texts)
            if i != j and text in other and len(other) > len(text)
        ]

        if containing_candidates:
            # Preserve the longest containing text; tie-break by first occurrence.
            keeper_pos = max(containing_candidates,
                             key=lambda j: len(normalized_texts[j]))
            keep_mask[i] = False
            removed_pairs.append(
                (
                    df.index[keeper_pos],
                    df.index[i],
                    original_texts[keeper_pos],
                    original_texts[i],
                )
            )

    return df.loc[keep_mask].copy(), removed_pairs


def remove_near_duplicates(df: pd.DataFrame, content_col: str, threshold: float) -> tuple[pd.DataFrame, list]:
    """Remove near-duplicate rows based on normalized similarity and return the cleaned DataFrame along with pairs of kept and removed rows."""
    texts = df[content_col].str.lower().str.strip().tolist()
    keep = []
    kept_texts: list[str] = []
    kept_indices: list[int] = []
    pairs = []
    for i, text in enumerate(texts):
        is_near_dup = False
        for j, kept in enumerate(kept_texts):
            sim = Indel.normalized_similarity(text, kept)
            if sim >= threshold:
                is_near_dup = True
                pairs.append((df.index[kept_indices[j]],
                             df.index[i], kept, text, sim))
                break
        if not is_near_dup:
            keep.append(df.index[i])
            kept_texts.append(text)
            kept_indices.append(i)
    return df.loc[keep].copy(), pairs


def near_duplicate_pairs(
    df: pd.DataFrame,
    content_col: str,
    min_similarity: float = 0.6,
    chunk_size: int = 2000,
) -> pd.DataFrame:
    """Compute normalized similarity for every pair of rows (upper triangle only) that scores at or above min_similarity."""

    texts = df[content_col].str.lower().str.strip().tolist()
    n = len(texts)
    index = df.index.to_numpy()
    records = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sim_chunk = rf_process.cdist(
            texts[start:end], texts,
            scorer=Indel.normalized_similarity,
            score_cutoff=min_similarity,
            workers=-1,
            dtype=np.float32,
        )
        for local_i, global_i in enumerate(range(start, end)):
            row = sim_chunk[local_i]
            hits = np.nonzero(row[global_i + 1:] >=
                              min_similarity)[0] + (global_i + 1)
            for global_j in hits:
                records.append((index[global_i], index[global_j],
                               texts[global_i], texts[global_j], float(row[global_j])))
    pairs_df = pd.DataFrame(
        records, columns=["idx_a", "idx_b", "text_a", "text_b", "similarity"])
    return pairs_df.sort_values("similarity", ascending=False).reset_index(drop=True)


def scan_near_duplicate_thresholds(pairs_df: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    """Count how many pairs would be flagged as near-duplicates at each candidate threshold."""
    counts = [(t, int((pairs_df["similarity"] >= t).sum()))
              for t in thresholds]
    return pd.DataFrame(counts, columns=["threshold", "num_pairs"])
