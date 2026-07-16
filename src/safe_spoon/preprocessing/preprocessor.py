import json
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class SimpleTMPreprocessor:
    """Simple text preprocessor for topic modeling.
    Building upon spacy for tokenization and lemmatization, and supports filtering by part-of-speech, stopword removal, and handling of equivalent words. It also provides methods to compute bag-of-words and TF-IDF.
    """

    def __init__(
        self,
        *,
        spacy_model: str = "en_core_web_sm",
        spacy_disable: Optional[List[str]] = None,
        valid_pos: Optional[List[str]] = None,
        stopword_files: Optional[List[str]] = None,
        equivalents_files: Optional[List[str]] = None,
        min_df: int = 2,
        max_df: float = 0.95,
        min_len: int = 4,
        max_features: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self._logger = logger or logging.getLogger(__name__)
        self._logger.info("Loading spaCy model %r", spacy_model)
        disable = spacy_disable or []
        try:
            import spacy
            self.nlp = spacy.load(spacy_model, disable=disable)
        except OSError as e:
            raise OSError(
                f"spaCy model '{spacy_model}' not found. "
                f"Install: python -m spacy download {spacy_model}"
            ) from e

        self.valid_pos = set(valid_pos) if valid_pos else {
            "NOUN", "VERB", "ADJ", "PROPN"}
        self.stopwords = {w.lower() for w in self.nlp.Defaults.stop_words}
        self.stopwords |= self._load_stopwords(stopword_files or [])
        self.equivalents = self._load_equivalents(equivalents_files or [])
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.min_len = min_len
        self._cv = None
        self._tfidf = None

    @staticmethod
    def _load_stopwords(files: List[str]) -> set:
        out = set()
        for f in files:
            p = Path(f)
            if not p.exists():
                continue
            out |= {w.strip().lower()
                    for w in p.open("r", encoding="utf8").readlines()}
        return out

    @staticmethod
    def _load_equivalents(files: List[str]) -> dict:
        eq = {}
        for f in files:
            p = Path(f)
            if not p.exists():
                continue
            if p.suffix.lower() == ".json":
                lines = json.load(p.open("r", encoding="utf8")
                                  ).get("wordlist", [])
            else:
                lines = p.open("r", encoding="utf8").readlines()
            for line in lines:
                line = str(line).strip()
                if ":" in line:
                    a, b = line.split(":", 1)
                    a, b = a.strip(), b.strip()
                    if a and b:
                        eq[a] = b
        return eq

    def _lemmatize(self, text: str, min_len: int = 4) -> List[str]:
        toks = [self.equivalents.get(t, t) for t in text.lower().split()]
        doc = self.nlp(" ".join(toks))
        sw, vp = self.stopwords, self.valid_pos
        return [
            t.lemma_.lower() for t in doc
            if t.is_alpha and t.pos_ in vp
            and not t.is_stop and t.lemma_.lower() not in sw and len(t.lemma_) >= min_len
        ]

    def fit_transform(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        id_col: str = "id",
        compute_bow: bool = True,
        compute_tfidf: bool = True,
    ) -> pd.DataFrame:
        out = pd.DataFrame()
        out["id"] = df[id_col].values if id_col in df.columns else range(
            len(df))
        out["text"] = df[text_col].fillna("").astype(str)
        out["lemmas"] = [self._lemmatize(t, min_len=self.min_df) for t in out["text"].tolist()]

        if compute_bow:
            self._cv = CountVectorizer(
                tokenizer=lambda x: x,
                preprocessor=lambda x: x,
                lowercase=False,
                min_df=self.min_df,
                max_df=self.max_df,
                max_features=self.max_features,
                token_pattern=None,
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
