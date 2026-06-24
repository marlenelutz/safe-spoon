"""Shared data-loading utilities for pipeline scripts."""

import pandas as pd


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
