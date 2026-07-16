import logging
import time
from typing import List, Optional

import numpy as np

from safe_spoon.utils.common import load_yaml_config_file


class SentenceEncoder:
    """Encode texts into dense vectors with a sentence-transformers model."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ):
        if model_name is None or batch_size is None:
            cfg = load_yaml_config_file()
            model_name = model_name or cfg["embedding_model"]
            batch_size = batch_size or cfg["embedding_batch_size"]

        self.model_name = model_name
        self.batch_size = batch_size
        self._logger = logger or logging.getLogger(__name__)

        self._model = None

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts with the configured sentence transformer.

        Parameters
        ----------
        texts:
            Raw strings to embed.

        Returns
        -------
        E : np.ndarray of shape (n_texts, model_dim)
            One row per text, aligned with the texts list.
        """
        if self._model is None:
            self._model = self._load_model()

        self._logger.info("  Encoding %d texts with '%s'...",
                          len(texts), self.model_name)
        t0 = time.time()
        E = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        self._logger.info("  Encoded in %.1fs  shape=%s",
                          time.time() - t0, E.shape)
        return E.astype(np.float32)

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            message = (
                "sentence-transformers is required for embedding-based clustering. "
                "Install it with: pip install sentence-transformers"
            )
            self._logger.error(
                message
            )
            raise ImportError(
                message
            )
        return SentenceTransformer(self.model_name)
