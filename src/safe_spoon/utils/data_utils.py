"""Shared data-loading utilities for pipeline scripts."""

import pandas as pd
from rapidfuzz.distance import Indel # type: ignore

def load_corpus_df(input_file: str, *, content_col: str = "content", label_col: str = "high_risk_label"):
    """Read the input CSV, keep only Analytical rows, and return raw lists.

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
    labels  = df[label_col].fillna("unknown").tolist()
    return df, queries, labels


def corpus_for_category(queries, labels, category: str) -> pd.DataFrame:
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
                pairs.append((kept.name, row.name, kept[content_col], row[content_col]))
    df = df.drop_duplicates(subset="_lower").drop(columns="_lower")
    return df, pairs


def remove_name_pattern(df: pd.DataFrame, content_col: str, name_pattern: str) -> pd.DataFrame:
    """Remove rows containing the NAME_X pattern"""
    mask = df[content_col].str.contains(name_pattern, regex=True, na=False)
    return df[~mask].copy()


def remove_near_duplicates(df: pd.DataFrame, content_col: str, threshold: float ) -> tuple[pd.DataFrame, list]:
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
                pairs.append((df.index[kept_indices[j]], df.index[i], kept, text, sim))
                break
        if not is_near_dup:
            keep.append(df.index[i])
            kept_texts.append(text)
            kept_indices.append(i)
    return df.loc[keep].copy(), pairs