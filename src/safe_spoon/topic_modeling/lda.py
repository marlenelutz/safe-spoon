"""LDA topic model using the Tomotopy library.

Collapsed from TOVA's BaseTMModel + TradTMmodel + TomotopyLDATMmodel into a
single flat class that requires no YAML configuration file.
"""

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sparse
import tomotopy as tp
from sklearn.preprocessing import normalize
from tqdm import tqdm

from safe_spoon.topic_modeling.tm_model import TMmodel


class LDATopicModel:
    """Train and use an LDA topic model backed by Tomotopy.

    Parameters
    ----------
    model_path:
        Directory where the model and all derived artifacts are stored.
        If it already exists when train() is called, it is renamed with an
        _old suffix so that a fresh model is always placed at the given path.
    num_topics:
        Number of latent topics.
    num_iters:
        Total training iterations.
    alpha:
        Dirichlet prior on document-topic distributions.
    eta:
        Dirichlet prior on topic-word distributions.
    iter_interval:
        Number of iterations between progress log messages.
    topn:
        Number of top words used to describe each topic.
    thetas_thr:
        Values in the theta matrix below this threshold are set to zero
        (sparsification) before creating the TMmodel.
    min_doc_words:
        Documents with fewer tokens than this value are skipped during
        training and receive a keyword-overlap-based theta fallback.
    do_labeller:
        If True, call an LLM to generate topic labels after training.
    do_summarizer:
        If True, call an LLM to generate topic summaries after training.
    llm_model_type / llm_provider / llm_server / llm_api_key:
        LLM connection parameters forwarded to the Prompter.
    labeller_prompt / summarizer_prompt:
        Paths to prompt template files.  Defaults to the bundled templates.
    preprocessor:
        An optional SimpleTMPreprocessor instance.  When provided,
        train() and infer() call its fit_transform() method to
        tokenise raw text.  When absent, data dicts must already contain
        a "lemmas" key with a list of tokens.
    logger:
        External logger.  Defaults to a basic logging logger.
    """

    def __init__(
        self,
        model_path,
        *,
        num_topics: int = 50,
        num_iters: int = 1000,
        alpha: float = 0.1,
        eta: float = 0.01,
        iter_interval: int = 10,
        topn: int = 15,
        thetas_thr: float = 3e-3,
        min_doc_words: int = 3,
        do_labeller: bool = False,
        do_summarizer: bool = False,
        llm_model_type: Optional[str] = None,
        llm_provider: Optional[str] = None,
        llm_server: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        labeller_prompt: Optional[str] = None,
        summarizer_prompt: Optional[str] = None,
        preprocessor=None,
        logger: Optional[logging.Logger] = None,
    ):
        self.model_path = Path(model_path)
        self.num_topics = num_topics
        self.num_iters = num_iters
        self.alpha = alpha
        self.eta = eta
        self.iter_interval = iter_interval
        self.topn = topn
        self.thetas_thr = thetas_thr
        self.min_doc_words = min_doc_words
        self.do_labeller = do_labeller
        self.do_summarizer = do_summarizer
        self.llm_model_type = llm_model_type
        self.llm_provider = llm_provider
        self.llm_server = llm_server
        self.llm_api_key = llm_api_key
        self.labeller_prompt = labeller_prompt
        self.summarizer_prompt = summarizer_prompt
        self._preprocessor = preprocessor
        self._logger = logger or logging.getLogger(__name__)

        self._lda_model: Optional[tp.LDAModel] = None
        self._tm: Optional[TMmodel] = None
        self._thetas: Optional[np.ndarray] = None
        self._topic_keys: Optional[List[List[str]]] = None
        self._vocab: Optional[List[str]] = None
        self._df: Optional[pd.DataFrame] = None


    def train(self, data: List[Dict]) -> Tuple[TMmodel, float]:
        """Train the LDA model on *data* and return the TMmodel + elapsed seconds.

        Parameters
        ----------
        data:
            List of dicts with keys "id" and "text".  If no preprocessor
            was configured, each dict must also contain "lemmas"
            (a list of tokens).

        Returns
        -------
        tm:
            The populated :class:TMmodel instance.
        elapsed:
            Wall-clock seconds for the full train call.
        """
        t_start = time.time()

        # Prepare model directory
        if self.model_path.exists():
            old_path = self.model_path.with_name(self.model_path.name + "_old")
            if old_path.exists():
                shutil.rmtree(old_path)
            shutil.move(str(self.model_path), str(old_path))
            self._logger.info(f"Old model moved to {old_path}")
        self.model_path.mkdir(parents=True, exist_ok=True)

        # 1. Preprocessing
        df = pd.DataFrame(data)
        if self._preprocessor is not None:
            self._logger.info("Preprocessing training data.")
            df_proc = self._preprocessor.fit_transform(
                df, text_col="text", id_col="id",
                compute_bow=False, compute_tfidf=False,
            )
        else:
            if "lemmas" not in df.columns:
                raise ValueError(
                    "No preprocessor configured. Each data dict must contain 'lemmas'."
                )
            df_proc = df.copy()
            if "id" not in df_proc.columns:
                df_proc["id"] = range(len(df_proc))
            if "text" not in df_proc.columns:
                df_proc["text"] = ""

        self._df = df_proc
        lemma_lists: List[List[str]] = df_proc["lemmas"].tolist()
        n = len(lemma_lists)

        # 2. Build trainable mask (short-doc fallback)
        trainable_mask = np.array([len(l) >= self.min_doc_words for l in lemma_lists])
        train_idx = np.where(trainable_mask)[0]
        short_idx = np.where(~trainable_mask)[0]
        self._logger.info(f"Trainable: {len(train_idx)}  Short (fallback): {len(short_idx)}")

        # 3. Train tomotopy LDA
        self._logger.info("Training LDA...")
        self._lda_model = tp.LDAModel(
            k=self.num_topics, tw=tp.TermWeight.ONE,
            alpha=self.alpha, eta=self.eta,
        )
        for i in train_idx:
            self._lda_model.add_doc(lemma_lists[i])

        pbar = tqdm(total=self.num_iters, desc='LDA Training')
        for i in range(0, self.num_iters, self.iter_interval):
            self._lda_model.train(self.iter_interval)
            pbar.update(self.iter_interval)
            if (i // self.iter_interval) % 10 == 0:
                self._logger.info(
                    f"  iter {i:4d}/{self.num_iters}  "
                    f"ll={self._lda_model.ll_per_word:.4f}  "
                    f"perp={self._lda_model.perplexity:.1f}"
                )
        pbar.close()

        # 4. Extract distributions
        thetas_train = np.array([d.get_topic_dist() for d in self._lda_model.docs])
        self._logger.info(f"Thetas shape: {thetas_train.shape}")

        betas = np.array([self._lda_model.get_topic_word_dist(k) for k in range(self.num_topics)])
        self._logger.info(f"Betas shape: {betas.shape}")

        self._topic_keys = [[w for w, _ in self._lda_model.get_topic_words(k, self.topn)]
                            for k in range(self.num_topics)]
        self._vocab = list(self._lda_model.used_vocabs)

        # Save human-readable topic descriptions
        with self.model_path.joinpath('orig_tpc_descriptions.txt').open('w', encoding='utf8') as fout:
            fout.write('\n'.join([' '.join(kw) for kw in self._topic_keys]))

        # 5. Sparsify thetas for trainable docs
        thetas_train[thetas_train < self.thetas_thr] = 0
        thetas_train = normalize(thetas_train, axis=1, norm='l1')

        # 6. Build full X_all, handling short docs with keyword-overlap fallback
        X_all = np.empty((n, self.num_topics))
        X_all[train_idx] = thetas_train
        if len(short_idx):
            X_all[short_idx] = np.vstack([
                self._kw_theta(lemma_lists[i], self._topic_keys, self.num_topics)
                for i in short_idx
            ])

        self._thetas = X_all  # dense, for downstream clustering

        # 7. Create TMmodel (uses sparsified thetas; also sorts topics internally)
        self._create_tm_model(X_all, betas, self._vocab)

        # 8. Re-derive topic_keys from TMmodel sorted betas so topic_keys,
        # topic_labels, alphas and tpc_coords all share the same sorted order.
        betas_sorted = np.load(str(self.model_path / "TMmodel" / "betas.npy"))
        vocab_sorted = (self.model_path / "TMmodel" / "vocab.txt").read_text(encoding="utf-8").strip().split("\n")
        top_idx = np.argsort(betas_sorted, axis=1)[:, ::-1][:, :self.topn]
        self._topic_keys = [[vocab_sorted[i] for i in row] for row in top_idx]
        # Also update thetas to sorted column order
        self._thetas = np.array(sparse.load_npz(str(self.model_path / "TMmodel" / "thetas.npz")).todense())

        # 9. Persist
        self.save()

        elapsed = time.time() - t_start
        self._logger.info(f"Training complete in {elapsed/60:.2f} min")
        return self._tm, elapsed

    def get_thetas(self) -> np.ndarray:
        """Return the full dense theta matrix (n_docs × n_topics), including short-doc fallback rows."""
        if self._thetas is None:
            raise RuntimeError("Model not trained yet. Call train() first.")
        return self._thetas

    def get_topic_keys(self) -> List[List[str]]:
        """Return the top-word lists for each topic."""
        if self._topic_keys is None:
            raise RuntimeError("Model not trained yet. Call train() first.")
        return self._topic_keys

    def infer(self, data: List[Dict]) -> np.ndarray:
        """Infer topic distributions for unseen documents.

        Parameters
        ----------
        data:
            List of dicts with keys "id" and "text".  If no preprocessor
            was configured, each dict must also contain "lemmas".

        Returns
        -------
        thetas : np.ndarray
            Document-topic matrix (n_docs × n_topics), sparsified and normalised.
        """
        if self._lda_model is None:
            raise RuntimeError("Model not trained or loaded. Call train() or load() first.")

        df = pd.DataFrame(data)
        if self._preprocessor is not None:
            df_proc = self._preprocessor.fit_transform(
                df, text_col="text", id_col="id",
                compute_bow=False, compute_tfidf=False,
            )
            lemma_lists = df_proc["lemmas"].tolist()
        else:
            if "lemmas" not in df.columns:
                raise ValueError("No preprocessor configured. Each data dict must contain 'lemmas'.")
            lemma_lists = df["lemmas"].tolist()

        doc_inst = [self._lda_model.make_doc(tokens) for tokens in lemma_lists]
        topic_prob, _ = self._lda_model.infer(doc_inst)
        thetas = np.array(topic_prob)
        thetas[thetas < self.thetas_thr] = 0
        thetas = normalize(thetas, axis=1, norm='l1')
        return thetas

    def save(self) -> None:
        """Save the tomotopy model binary and metadata JSON."""
        bin_path = self.model_path / "model.bin"
        self._logger.info(f"Saving model to {bin_path}")
        self._lda_model.save(str(bin_path))

        meta = {
            "num_topics": self.num_topics,
            "num_iters": self.num_iters,
            "alpha": self.alpha,
            "eta": self.eta,
            "iter_interval": self.iter_interval,
            "topn": self.topn,
            "thetas_thr": self.thetas_thr,
            "min_doc_words": self.min_doc_words,
            "do_labeller": self.do_labeller,
            "do_summarizer": self.do_summarizer,
            "llm_model_type": self.llm_model_type,
            "llm_provider": self.llm_provider,
            "llm_server": self.llm_server,
            # llm_api_key intentionally omitted
            "labeller_prompt": str(self.labeller_prompt) if self.labeller_prompt else None,
            "summarizer_prompt": str(self.summarizer_prompt) if self.summarizer_prompt else None,
        }
        with (self.model_path / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        self._logger.info("Model saved successfully.")

    @classmethod
    def load(cls, model_path) -> "LDATopicModel":
        """Load a previously saved model from *model_path*.

        The returned instance has _lda_model ready for inference.
        The TMmodel is accessible via the tm property (loaded lazily).
        A preprocessor must be re-injected manually if needed for inference.
        """
        model_path = Path(model_path)
        meta_path = model_path / "metadata.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"metadata.json not found in {model_path}")

        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        obj = cls(model_path=model_path, **meta)
        bin_path = model_path / "model.bin"
        obj._logger.info(f"Loading Tomotopy model from {bin_path}")
        obj._lda_model = tp.LDAModel.load(str(bin_path))

        # Restore thetas and topic_keys from TMmodel artifacts so they use
        # the same sorted topic ordering that TMmodel._sort_topics() applied.
        thetas_path = model_path / "TMmodel" / "thetas.npz"
        betas_path  = model_path / "TMmodel" / "betas.npy"
        vocab_path  = model_path / "TMmodel" / "vocab.txt"

        if thetas_path.is_file():
            obj._thetas = np.array(sparse.load_npz(str(thetas_path)).todense())
            obj._logger.info(f"Thetas restored from disk: {obj._thetas.shape}")

        if betas_path.is_file() and vocab_path.is_file():
            betas = np.load(str(betas_path))
            vocab = vocab_path.read_text(encoding="utf-8").strip().split("\n")
            top_idx = np.argsort(betas, axis=1)[:, ::-1][:, :obj.topn]
            obj._topic_keys = [[vocab[i] for i in row] for row in top_idx]
            obj._logger.info(f"Topic keys restored: {len(obj._topic_keys)} topics")

        obj._logger.info("Model loaded successfully.")
        return obj

    # Alias for compatibility
    from_saved_model = load

    @property
    def tm(self) -> TMmodel:
        """Lazily load and return the :class:TMmodel from disk."""
        if self._tm is None:
            tm_folder = self.model_path / "TMmodel"
            if not tm_folder.is_dir():
                raise FileNotFoundError(
                    f"TMmodel folder not found at {tm_folder}. "
                    "Train the model first or ensure the folder exists."
                )
            self._tm = TMmodel(
                TMfolder=tm_folder,
                do_labeller=self.do_labeller,
                do_summarizer=self.do_summarizer,
                llm_model_type=self.llm_model_type,
                llm_server=self.llm_server,
                llm_provider=self.llm_provider,
                llm_api_key=self.llm_api_key,
                labeller_prompt=self.labeller_prompt,
                summarizer_prompt=self.summarizer_prompt,
            )
        return self._tm

    def _create_tm_model(
        self,
        X_all: np.ndarray,
        betas: np.ndarray,
        vocab: List[str],
    ) -> None:
        """Create and populate the TMmodel from raw training outputs."""
        # Sparsify for TMmodel (separate from self._thetas which stays dense)
        thetas_sp = X_all.copy()
        thetas_sp[thetas_sp < self.thetas_thr] = 0
        thetas_sp = normalize(thetas_sp, axis=1, norm='l1')
        thetas_sp = sparse.csr_matrix(thetas_sp, copy=True)

        alphas = np.asarray(np.mean(thetas_sp, axis=0)).ravel()

        tm = TMmodel(
            TMfolder=self.model_path / "TMmodel",
            df_corpus_train=self._df,
            do_labeller=self.do_labeller,
            do_summarizer=self.do_summarizer,
            llm_model_type=self.llm_model_type,
            llm_server=self.llm_server,
            llm_provider=self.llm_provider,
            llm_api_key=self.llm_api_key,
            labeller_prompt=self.labeller_prompt,
            summarizer_prompt=self.summarizer_prompt,
            logger=self._logger,
        )
        tm.create(betas=betas, thetas=thetas_sp, alphas=alphas, vocab=vocab)
        self._tm = tm

    @staticmethod
    def _kw_theta(tokens: List[str], keys: List[List[str]], k: int) -> np.ndarray:
        """Keyword-overlap fallback distribution for short documents."""
        scores = np.array([len(set(tokens) & set(kw)) for kw in keys], dtype=float)
        if scores.sum() == 0:
            scores = np.ones(k)
        return scores / scores.sum()
