"""Generic topic model representation for curation and visualization.

Adapted from the topicmodeler project and TOVA.
Authors: Jerónimo Arenas-García, J.A. Espinosa-Melchor, Lorena Calvo-Bartolomé
"""

import concurrent.futures
import itertools
import json
import logging
import shutil
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import rbo
import scipy.sparse as sparse
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from scipy.spatial.distance import jensenshannon

from safe_spoon.prompting import Prompter, _default_prompt_path


class TMmodel(object):
    """Represents a topic model (LDA-style) and provides curation operations.

    The model is characterised by:
      _alphas  - topic weights vector
      _betas   - topic-word matrix  (n_topics x n_vocab)
      _thetas  - document-topic matrix (n_docs x n_topics, sparse)

    All matrices and derived quantities are persisted to TMfolder so that
    the object can be reconstructed from disk without retraining.
    """

    _TMfolder = None

    _betas_orig = None
    _thetas_orig = None
    _alphas_orig = None

    _betas = None
    _thetas = None
    _alphas = None
    _edits = None
    _ntopics = None
    _betas_ds = None
    _coords = None
    _topic_entropy = None
    _topic_coherence = None
    _ndocs_active = None
    _tpc_descriptions = None
    _tpc_labels = None
    _tpc_summaries = None
    _tpc_add_info = None
    _vocab_w2id = None
    _vocab_id2w = None
    _vocab = None
    _size_vocab = None
    _most_representative_docs = None
    _tpc_clusters = None

    def __init__(
        self,
        TMfolder: Path,
        df_corpus_train: pd.DataFrame = None,
        do_labeller: bool = False,
        do_summarizer: bool = False,
        llm_model_type: str = None,
        llm_server: Optional[str] = None,
        llm_provider: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        labeller_prompt: str = None,
        summarizer_prompt: str = None,
        logger: logging.Logger = None,
    ):
        if logger:
            self._logger = logger
        else:
            logging.basicConfig(level='INFO')
            self._logger = logging.getLogger('TMmodel')

        self._TMfolder = Path(TMfolder)

        if not self._TMfolder.is_dir():
            try:
                self._TMfolder.mkdir(parents=True)
            except Exception:
                self._logger.error(
                    '-- -- Topic model object (TMmodel) could not be created')

        self._df_corpus_train = df_corpus_train
        self._do_labeller = do_labeller
        self._do_summarizer = do_summarizer
        self.llm_model_type = llm_model_type
        self.llm_server = llm_server
        self.llm_provider = llm_provider
        self.llm_api_key = llm_api_key
        self._labeller_prompt = labeller_prompt or _default_prompt_path("labelling_dft.txt")
        self._summarizer_prompt = summarizer_prompt or _default_prompt_path("summarization_dft.txt")
        self._training_warnings: List[str] = []

        self._logger.info(
            '-- -- -- Topic model object (TMmodel) successfully created')

    def create(self, betas=None, thetas=None, alphas=None, vocab=None,
               tpc_labels=None, tpc_summaries=None, add_info=None):
        """Initialise the topic model from raw matrices and persist all derived quantities."""

        if not self._TMfolder.is_dir():
            self._logger.error('-- -- Topic model object (TMmodel) folder not ready')
            return

        self._alphas_orig = alphas
        self._betas_orig = betas
        self._thetas_orig = thetas
        self._tpc_labels = tpc_labels
        self._tpc_summaries = tpc_summaries
        self._alphas = alphas
        self._betas = betas
        self._thetas = thetas
        self._vocab = vocab
        self._size_vocab = len(vocab)
        self._ntopics = thetas.shape[1]
        self._edits = []

        np.save(self._TMfolder.joinpath('alphas_orig.npy'), alphas)
        np.save(self._TMfolder.joinpath('betas_orig.npy'), betas)
        sparse.save_npz(self._TMfolder.joinpath('thetas_orig.npz'), thetas)
        with self._TMfolder.joinpath('vocab.txt').open('w', encoding='utf8') as fout:
            fout.write('\n'.join(vocab))

        self._sort_topics()
        self._logger.info("-- -- Sorted")
        self._calculate_beta_ds()
        self._logger.info("-- -- betas ds")
        self._calculate_topic_entropy()
        self._logger.info("-- -- entropy")
        self._ndocs_active = np.array((self._thetas != 0).sum(0).tolist()[0])
        self._logger.info("-- -- active")
        self._tpc_descriptions = [el[1] for el in self.get_tpc_word_descriptions()]
        self._logger.info("-- -- descriptions")
        self.calculate_topic_coherence()

        self._load_vocab_dicts()

        if self._do_labeller:
            try:
                self._tpc_labels = [el[1] for el in self.get_tpc_labels()]
            except Exception as e:
                self._logger.warning(f"Error in labeller: {e}")
                self._tpc_labels = ["Topic " + str(i) for i in range(self._ntopics)]
        elif not self._do_labeller and (self._tpc_labels is None):
            self._tpc_labels = ["Topic " + str(i) for i in range(self._ntopics)]

        if self._do_summarizer:
            try:
                self._tpc_summaries = [el[1] for el in self.get_tpc_summaries()]
            except Exception as e:
                self._logger.warning(f"Error in summarizer: {e}")
                self._tpc_summaries = ["Placeholder for summary from Topic " + str(i) for i in range(self._ntopics)]
        elif not self._do_summarizer and (self._tpc_summaries is None):
            self._tpc_summaries = ["Placeholder for summary from Topic " + str(i) for i in range(self._ntopics)]

        self._tpc_add_info = add_info

        self.get_most_representative_per_tpc(self._thetas)
        self.get_topic_clusters()

        self.get_thetas_representation()

        try:
            self.calculate_rbo()
            self.calculate_topic_diversity()
        except Exception as e:
            self._logger.warning(f"Error in rbo or topic diversity: {e}")

        self._save_all()

        self._logger.info(
            '-- -- Topic model variables were computed and saved to file')
        return

    def _save_all(self):
        np.save(self._TMfolder.joinpath('alphas.npy'), self._alphas)
        np.save(self._TMfolder.joinpath('betas.npy'), self._betas)
        sparse.save_npz(self._TMfolder.joinpath('thetas.npz'), self._thetas)

        with self._TMfolder.joinpath('edits.txt').open('w', encoding='utf8') as fout:
            fout.write('\n'.join(self._edits))
        np.save(self._TMfolder.joinpath('betas_ds.npy'), self._betas_ds)
        np.save(self._TMfolder.joinpath('topic_entropy.npy'), self._topic_entropy)
        np.save(self._TMfolder.joinpath('topic_coherence.npy'), self._topic_coherence)
        np.save(self._TMfolder.joinpath('ndocs_active.npy'), self._ndocs_active)
        with self._TMfolder.joinpath('tpc_descriptions.txt').open('w', encoding='utf8') as fout:
            fout.write('\n'.join(self._tpc_descriptions))

        self.save_topic_documents(mode="most_representative")
        self.save_topic_documents(mode="clusters")

        self.save_thetas_representation()

        with self._TMfolder.joinpath('tpc_labels.txt').open('w', encoding='utf8') as fout:
            fout.write('\n'.join(self._tpc_labels))

        with self._TMfolder.joinpath('tpc_summaries.txt').open('w', encoding='utf8') as fout:
            fout.write('\n'.join(self._tpc_summaries))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            try:
                import pyLDAvis
            except ImportError:
                msg = "pyLDAvis not installed; visualization skipped. Install with: pip install safe-spoon[viz]"
                self._logger.warning(msg)
                self._training_warnings.append(msg)
                return

        if self._ntopics < 2:
            msg = f"The model has only {self._ntopics} topic(s); pyLDAvis visualization could not be generated."
            self._logger.warning(msg)
            self._training_warnings.append(msg)
            return

        try:
            ndocs = 10000
            validDocs = np.sum(self._thetas.toarray(), axis=1) > 0
            nValidDocs = np.sum(validDocs)
            if ndocs > nValidDocs:
                ndocs = nValidDocs
            perm = np.sort(np.random.permutation(nValidDocs)[:ndocs])
            doc_len = ndocs * [1]
            vocabfreq = np.round(ndocs * (self._alphas.dot(self._betas))).astype(int)
            vis_data = pyLDAvis.prepare(
                self._betas,
                self._thetas[validDocs, ][perm, ].toarray(),
                doc_len,
                self._vocab,
                vocabfreq,
                lambda_step=0.05,
                sort_topics=False,
                n_jobs=-1)

            with self._TMfolder.joinpath("pyLDAvis.html").open("w") as f:
                pyLDAvis.save_html(vis_data, f)

            vis_data_dict = vis_data.to_dict()
            self._coords = list(
                zip(*[vis_data_dict['mdsDat']['x'], vis_data_dict['mdsDat']['y']]))

            with self._TMfolder.joinpath('tpc_coords.txt').open('w', encoding='utf8') as fout:
                for item in self._coords:
                    fout.write(str(item) + "\n")
        except Exception as e:
            msg = f"pyLDAvis visualization could not be generated: {e}"
            self._logger.warning(msg)
            self._training_warnings.append(msg)
        return

    def _save_cohr(self):
        np.save(self._TMfolder.joinpath('topic_coherence.npy'), self._topic_coherence)

    def _sort_topics(self):
        self._load_alphas()
        self._load_betas()
        self._load_thetas()
        self._load_edits()

        idx = np.argsort(self._alphas)[::-1]
        self._edits.append('s ' + ' '.join([str(el) for el in idx]))

        self._alphas = self._alphas[idx]
        self._betas = self._betas[idx, :]
        self._thetas = self._thetas[:, idx]

        return

    def _load_alphas(self):
        if self._alphas is None:
            self._alphas = np.load(self._TMfolder.joinpath('alphas.npy'))
            self._ntopics = self._alphas.shape[0]

    def _load_betas(self):
        if self._betas is None:
            self._betas = np.load(self._TMfolder.joinpath('betas.npy'))
            self._ntopics = self._betas.shape[0]
            self._size_vocab = self._betas.shape[1]

    def _load_thetas(self):
        if self._thetas is None:
            self._thetas = sparse.load_npz(self._TMfolder.joinpath('thetas.npz'))
            self._ntopics = self._thetas.shape[1]

    def _load_ndocs_active(self):
        if self._ndocs_active is None:
            self._ndocs_active = np.load(self._TMfolder.joinpath('ndocs_active.npy'))
            self._ntopics = self._ndocs_active.shape[0]

    def _load_edits(self):
        if self._edits is None:
            with self._TMfolder.joinpath('edits.txt').open('r', encoding='utf8') as fin:
                self._edits = fin.readlines()

    def _calculate_beta_ds(self):
        self._load_betas()

        self._betas_ds = np.copy(self._betas)
        if np.min(self._betas_ds) < 1e-12:
            self._betas_ds += 1e-12
        deno = np.reshape((sum(np.log(self._betas_ds)) / self._ntopics), (self._size_vocab, 1))
        deno = np.ones((self._ntopics, 1)).dot(deno.T)
        self._betas_ds = self._betas_ds * (np.log(self._betas_ds) - deno)

    def _load_betas_ds(self):
        if self._betas_ds is None:
            self._betas_ds = np.load(self._TMfolder.joinpath('betas_ds.npy'))
            self._ntopics = self._betas_ds.shape[0]
            self._size_vocab = self._betas_ds.shape[1]

    def _load_vocab(self):
        if self._vocab is None:
            with self._TMfolder.joinpath('vocab.txt').open('r', encoding='utf8') as fin:
                self._vocab = [el.strip() for el in fin.readlines()]

    def _load_vocab_dicts(self):
        if self._vocab_w2id is None and self._vocab_id2w is None:
            self._vocab_w2id = {}
            self._vocab_id2w = {}
            with self._TMfolder.joinpath('vocab.txt').open('r', encoding='utf8') as fin:
                for i, line in enumerate(fin):
                    wd = line.strip()
                    self._vocab_w2id[wd] = i
                    self._vocab_id2w[str(i)] = wd

    def _calculate_topic_entropy(self):
        self._load_betas()

        if np.min(self._betas) < 1e-12:
            self._betas += 1e-12
        self._topic_entropy = -np.sum(self._betas * np.log(self._betas), axis=1)
        self._topic_entropy = self._topic_entropy / np.log(self._size_vocab)

    def _load_topic_entropy(self):
        if self._topic_entropy is None:
            self._topic_entropy = np.load(self._TMfolder.joinpath('topic_entropy.npy'))

    def calculate_rbo(self, weight: float = 1.0, n_words: int = 15) -> float:
        if self._tpc_descriptions is None:
            self._tpc_descriptions = [el[1] for el in self.get_tpc_word_descriptions(n_words)]

        collect = []
        for list1, list2 in itertools.combinations(self._tpc_descriptions, 2):
            rbo_val = rbo.RankingSimilarity(
                list1.split(", "), list2.split(", ")).rbo(p=weight)
            collect.append(rbo_val)

        irbo = 1 - np.mean(collect)

        try:
            with self._TMfolder.joinpath('rbo.txt').open('w', encoding='utf8') as fout:
                fout.write(str(irbo))
        except Exception:
            self._logger.warning("Rank-biased overlap could not be saved to file")
        return irbo

    def calculate_topic_diversity(self, n_words: int = 15) -> float:
        if self._tpc_descriptions is None:
            self._tpc_descriptions = [el[1] for el in self.get_tpc_word_descriptions(n_words)]

        unique_words = set()
        for topic in self._tpc_descriptions:
            unique_words = unique_words.union(set(topic.split(", ")))
        td = len(unique_words) / (n_words * len(self._tpc_descriptions))

        try:
            with self._TMfolder.joinpath('topic_diversity.txt').open('w', encoding='utf8') as fout:
                fout.write(str(td))
        except Exception:
            self._logger.warning("Topic diversity could not be saved to file")
        return td

    def calculate_topic_coherence(
        self,
        metrics: List[str] = ["c_npmi", "c_v"],
        n_words: int = 15,
        reference_text: Optional[List[List[str]]] = None,
        only_one: bool = True,
        aggregated: bool = False
    ) -> list:
        if self._tpc_descriptions is None:
            self._tpc_descriptions = [el[1] for el in self.get_tpc_word_descriptions()]

        tpc_descriptions_ = [tpc.split(', ') for tpc in self._tpc_descriptions]

        if reference_text is None:
            corpus = [el.split() for el in self._df_corpus_train["text"].values.tolist()]
        else:
            corpus = reference_text

        dictionary = None
        if self._TMfolder.parent.joinpath('dictionary.gensim').is_file():
            try:
                dictionary = Dictionary.load_from_text(
                    self._TMfolder.parent.joinpath('dictionary.gensim').as_posix())
            except Exception:
                self._logger.warning("Gensim dictionary could not be loaded from file.")
        if dictionary is None:
            dictionary = Dictionary(corpus)

        if n_words > len(tpc_descriptions_[0]):
            self._logger.error(
                '-- -- -- Coherence calculation failed: n_words exceeds topic word count.')
            return None

        if only_one:
            metric = metrics[0]
            self._logger.info(f"Calculating just coherence {metric}.")
            if metric in ["c_npmi", "u_mass", "c_v", "c_uci"]:
                cm = CoherenceModel(
                    topics=tpc_descriptions_, texts=corpus,
                    dictionary=dictionary, coherence=metric, topn=n_words)
                self._topic_coherence = cm.get_coherence_per_topic()
                if aggregated:
                    return cm.aggregate_measures(self._topic_coherence)
                return self._topic_coherence
            else:
                self._logger.error('-- -- -- Coherence metric not available.')
                return None
        else:
            cohrs_aux = []
            for metric in metrics:
                self._logger.info(f"Calculating coherence {metric}.")
                if metric in ["c_npmi", "u_mass", "c_v", "c_uci"]:
                    cm = CoherenceModel(
                        topics=tpc_descriptions_, texts=corpus,
                        dictionary=dictionary, coherence=metric, topn=n_words)
                    aux = cm.get_coherence_per_topic()
                    cohrs_aux.extend(aux)
                else:
                    self._logger.error('-- -- -- Coherence metric not available.')
                    return None
            self._topic_coherence = cohrs_aux

        return self._topic_coherence

    def _load_topic_coherence(self):
        if self._topic_coherence is None:
            coherence_path = self._TMfolder.joinpath('topic_coherence.npy')
            if not coherence_path.is_file():
                self._logger.warning("topic_coherence.npy not found; using zeros.")
                n = self._ntopics if self._ntopics else 0
                self._topic_coherence = np.zeros(n)
                return
            self._topic_coherence = np.load(coherence_path)

    def _largest_indices(self, ary, n):
        flat = ary.flatten()
        indices = np.argpartition(flat, -n)[-n:]
        indices = indices[np.argsort(-flat[indices])]
        idx0, idx1 = np.unravel_index(indices, ary.shape)
        idx0 = idx0.tolist()
        idx1 = idx1.tolist()
        selected_idx = []
        for id0, id1 in zip(idx0, idx1):
            if id0 < id1:
                selected_idx.append((id0, id1, ary[id0, id1]))
        return selected_idx

    def get_model_info_for_hierarchical(self):
        self._load_betas()
        self._load_thetas()
        self._load_vocab_dicts()
        return self._betas, self._thetas, self._vocab_w2id, self._vocab_id2w

    def get_model_info_for_vis(self):
        self._load_alphas()
        self._load_betas()
        self._load_thetas()
        self._load_vocab()
        self.load_tpc_coords()
        return self._alphas, self._betas, self._thetas, self._vocab, self._coords

    def get_tpc_word_descriptions(self, n_words=15, tfidf=True, tpc=None):
        if tfidf:
            self._load_betas_ds()
        else:
            self._load_betas()
        self._load_vocab()

        if not tpc:
            tpc = range(self._ntopics)

        tpc_descs = []
        for i in tpc:
            if tfidf:
                words = [self._vocab[idx2] for idx2 in np.argsort(self._betas_ds[i])[::-1][0:n_words]]
            else:
                words = [self._vocab[idx2] for idx2 in np.argsort(self._betas[i])[::-1][0:n_words]]
            tpc_descs.append((i, ', '.join(words)))

        return tpc_descs

    def load_tpc_descriptions(self):
        if self._tpc_descriptions is None:
            with self._TMfolder.joinpath('tpc_descriptions.txt').open('r', encoding='utf8') as fin:
                self._tpc_descriptions = [el.strip() for el in fin.readlines()]

    def get_thetas_representation(self):
        if self._thetas is None:
            self._load_thetas()

        all_docs = {}
        thetas_array = self._thetas.toarray()

        for doc_id, topic_distribution in zip(self._df_corpus_train.id, thetas_array):
            non_zero_topics = [(topic_id, float(prob)) for topic_id, prob in enumerate(topic_distribution) if prob > 0]
            sorted_topics = sorted(non_zero_topics, key=lambda x: x[1], reverse=True)
            all_docs[doc_id] = sorted_topics

        return all_docs

    def save_thetas_representation(self):
        all_docs = self.get_thetas_representation()
        output_path = self._TMfolder.joinpath("thetas_representation.json")
        with output_path.open("w", encoding="utf-8") as fout:
            json.dump(all_docs, fout, indent=4)
        self._logger.info(f"Thetas representation saved to {output_path}")

    def load_thetas_representation(self):
        input_path = self._TMfolder.joinpath("thetas_representation.json")
        if not input_path.is_file():
            self._logger.error(f"Thetas representation file not found: {input_path}")
            return
        with input_path.open("r", encoding="utf-8") as fin:
            return json.load(fin)

    def get_most_representative_per_tpc(self, mat, topn=None, get_text=False):
        top_docs_per_topic = []
        aux = mat.toarray()

        if topn is None:
            topn = len(aux)

        for doc_distr in aux.T:
            sorted_docs_indices = np.argsort(doc_distr)[::-1]
            top = sorted_docs_indices[:topn].tolist()
            top_docs_per_topic.append(top)

        most_representative_docs = []

        if get_text:
            for topic_id, topic_docs in enumerate(top_docs_per_topic):
                reps = [
                    (self._df_corpus_train.iloc[doc].id, self._df_corpus_train.iloc[doc].text, aux[doc, topic_id])
                    for doc in topic_docs
                ]
                most_representative_docs.append(reps)
        else:
            for topic_id, topic_docs in enumerate(top_docs_per_topic):
                reps = [
                    (self._df_corpus_train.iloc[doc].id, "", aux[doc, topic_id])
                    for doc in topic_docs
                ]
                most_representative_docs.append(reps)

        self._most_representative_docs = most_representative_docs

    def get_topic_clusters(self, get_text: bool = False):
        if self._thetas is None:
            self._logger.warning("Thetas not loaded.")
            return []

        thetas = self._thetas.toarray()
        n_topics = thetas.shape[1]

        if not hasattr(self, "_df_corpus_train"):
            self._logger.warning("Document corpus not available.")
            return []
        elif len(self._df_corpus_train) != thetas.shape[0] and hasattr(self, "sample") and len(self.df) == thetas.shape[0]:
            df_corpus = self.df
        else:
            df_corpus = self._df_corpus_train

        clusters = [[] for _ in range(n_topics)]

        for doc_idx, topic_probs in enumerate(thetas):
            top_topic = topic_probs.argmax()
            doc_id = df_corpus.iloc[doc_idx].id
            text = df_corpus.iloc[doc_idx].text if get_text else ""
            prob = topic_probs[top_topic]
            clusters[top_topic].append((doc_id, text, prob))

        self._tpc_clusters = clusters

    def save_topic_documents(self, mode: str = "most_representative", output_file: str = None):
        if mode not in {"most_representative", "clusters"}:
            raise ValueError("Mode must be 'most_representative' or 'clusters'.")

        if mode == "most_representative":
            if self._most_representative_docs is None:
                self._logger.warning("Most representative documents not calculated yet.")
                return
            data = self._most_representative_docs
            filename = "most_representative_docs.jsonl"
        else:
            data = self._tpc_clusters
            filename = "topic_clusters.jsonl"

        doc_meta_map = {}
        extra_keys = set()

        if self._tpc_add_info:
            for topic_info in self._tpc_add_info:
                for doc in topic_info.get("docs", []):
                    d_id = doc.get("doc_id")
                    if d_id is not None:
                        meta = {k: v for k, v in doc.items() if k not in ['doc_id', 'prob']}
                        doc_meta_map[d_id] = meta
                        extra_keys.update(meta.keys())

        sorted_extra_keys = sorted(list(extra_keys))
        output_path = Path(output_file) if output_file else self._TMfolder.joinpath(filename)

        with output_path.open("w", encoding="utf-8") as fout:
            for tpc_id, docs in enumerate(data):
                enriched_docs = []
                for doc_id, _, prob in docs:
                    doc_id = doc_id.item() if hasattr(doc_id, "item") else doc_id
                    doc_obj = {"doc_id": doc_id, "prob": float(prob)}

                    if self._tpc_add_info:
                        meta_data = doc_meta_map.get(doc_id, {})
                        for key in sorted_extra_keys:
                            val = meta_data.get(key, "")
                            doc_obj[key] = val.item() if hasattr(val, "item") else val

                    enriched_docs.append(doc_obj)

                topic_entry = {"topic_id": tpc_id, "docs": enriched_docs}
                fout.write(json.dumps(topic_entry) + "\n")

        self._logger.info(f"{mode.replace('_', ' ').title()} documents saved to {output_path}")

    def load_topic_documents(self, mode: str = "most_representative", n_most: int = None, store: bool = True):
        if mode not in {"most_representative", "clusters"}:
            raise ValueError("Mode must be 'most_representative' or 'clusters'.")

        filename = "most_representative_docs.jsonl" if mode == "most_representative" else "topic_clusters.jsonl"
        jsonl_path = self._TMfolder.joinpath(filename)

        if not jsonl_path.is_file():
            self._logger.warning(f"File not found: {jsonl_path}")
            return None

        self._logger.info(f"Loading topic document assignments from {jsonl_path}")
        topic_docs_list = []

        with jsonl_path.open("r", encoding="utf-8") as fin:
            for line in fin:
                entry = json.loads(line)
                docs = sorted(entry.get("docs", []), key=lambda d: d["prob"], reverse=True)
                keep_n = n_most if n_most is not None else len(docs)
                topic_docs = [(doc["doc_id"], None, doc["prob"]) for doc in docs[:keep_n]]
                topic_docs_list.append(topic_docs)

        if store:
            if mode == "most_representative":
                self._most_representative_docs = topic_docs_list
            else:
                self._tpc_clusters = topic_docs_list
            self._logger.info(f"Loaded {mode.replace('_', ' ')} documents into internal attribute.")
        else:
            return topic_docs_list

    def generate_topic_outputs(self, task: str = "label", topn: int = 3, max_tokens: int = None, batch_size: int = None, max_retries: int = 5):
        """Generate LLM-based labels or summaries for all topics using parallel batch processing."""
        if task not in {"label", "summary"}:
            raise ValueError(f"Invalid task: {task}. Use 'label' or 'summary'.")

        self.load_tpc_descriptions()
        self.get_most_representative_per_tpc(self._thetas, topn=topn, get_text=True)

        prompt_path = self._labeller_prompt if task == "label" else self._summarizer_prompt
        template_str = Path(str(prompt_path)).read_text(encoding="utf-8")

        self._logger.info(
            f"Effective LLM provider used: {self.llm_provider}, "
            f"model: {self.llm_model_type}, server: {self.llm_server}"
        )
        prompter = Prompter(
            model_type=self.llm_model_type,
            llm_server=self.llm_server,
            llm_provider=self.llm_provider,
            api_key=self.llm_api_key,
            #max_tokens=max_tokens,
        )

        prompts = []
        for tpc_id, most_repr in enumerate(self._most_representative_docs):
            docs = "\n- " + "\n- ".join([doc_tuple[1] for doc_tuple in most_repr])
            prompt_filled = template_str.format(
                keywords=self._tpc_descriptions[tpc_id],
                docs=docs,
            )
            prompts.append((tpc_id, prompt_filled))

        def _run_prompt(args):
            tpc_id, prompt_filled = args
            output_text = None
            for attempt in range(max_retries + 1):
                # Vary temperature on retries so joblib cache doesn't return the same empty result.
                temperature = attempt * 0.1 if attempt > 0 else None
                raw, _ = prompter.prompt(
                    question=prompt_filled,
                    system_prompt_template_path=None,
                    temperature=temperature,
                )
                if raw and raw.strip():
                    output_text = raw.replace("\n", " ")
                    break
                self._logger.warning(
                    f"Topic {tpc_id}: empty output on attempt {attempt + 1}/{max_retries + 1}"
                )
            if output_text is None:
                self._logger.error(f"Topic {tpc_id}: all {max_retries + 1} attempts returned empty output.")
                output_text = ""
            return tpc_id, output_text

        max_workers = batch_size or len(prompts)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_run_prompt, prompts))

        return sorted(results, key=lambda x: x[0])

    def get_tpc_labels(self, topn=3, max_tokens=50):
        return self.generate_topic_outputs(task="label", topn=topn, max_tokens=max_tokens)

    def get_tpc_summaries(self, topn=3):
        return self.generate_topic_outputs(task="summary", topn=topn)

    def load_tpc_labels(self):
        if self._tpc_labels is None:
            with self._TMfolder.joinpath('tpc_labels.txt').open('r', encoding='utf8') as fin:
                self._tpc_labels = [el.strip() for el in fin.readlines()]

    def load_tpc_summaries(self):
        if self._tpc_summaries is None:
            with self._TMfolder.joinpath('tpc_summaries.txt').open('r', encoding='utf8') as fin:
                self._tpc_summaries = [el.strip() for el in fin.readlines()]

    def load_tpc_coords(self):
        if self._coords is None:
            coords_path = self._TMfolder.joinpath('tpc_coords.txt')
            if not coords_path.is_file():
                self._logger.warning("tpc_coords.txt not found; using placeholder coordinates.")
                n = self._ntopics if self._ntopics else 0
                if n <= 1:
                    self._coords = [(0.0, 0.0)] * n
                else:
                    rng = np.random.default_rng(seed=42)
                    self._coords = [tuple(xy) for xy in rng.uniform(-1.0, 1.0, size=(n, 2))]
                return
            with coords_path.open('r', encoding='utf8') as fin:
                self._coords = [
                    tuple(map(float, line.strip()[1:-1].split(', ')))
                    for line in fin
                ]

    def get_alphas(self):
        self._load_alphas()
        return self._alphas

    def showTopics(self):
        self._load_alphas()
        self._load_ndocs_active()
        self.load_tpc_descriptions()
        self.load_tpc_labels()
        return [
            {
                "Size": str(round(el[0], 4)),
                "Label": el[1].strip(),
                "Word Description": el[2].strip(),
                "Ndocs Active": str(el[3]),
            }
            for el in zip(self._alphas, self._tpc_labels, self._tpc_descriptions, self._ndocs_active)
        ]

    def showTopicsAdvanced(self):
        self._load_alphas()
        self._load_ndocs_active()
        self.load_tpc_descriptions()
        self.load_tpc_labels()
        self._load_topic_entropy()
        self._load_topic_coherence()
        return [
            {
                "Size": str(round(el[0], 4)),
                "Label": el[1].strip(),
                "Word Description": el[2].strip(),
                "Ndocs Active": str(el[3]),
                "Topics entropy": str(round(el[4], 4)),
                "Topics coherence": str(round(el[5], 4)),
            }
            for el in zip(
                self._alphas, self._tpc_labels, self._tpc_descriptions,
                self._ndocs_active, self._topic_entropy, self._topic_coherence,
            )
        ]

    def setTpcLabels(self, TpcLabels):
        self._tpc_labels = [el.strip() for el in TpcLabels]
        self._load_alphas()
        if len(TpcLabels) == self._ntopics:
            with self._TMfolder.joinpath('tpc_labels.txt').open('w', encoding='utf8') as fout:
                fout.write('\n'.join(self._tpc_labels))
            return 1
        else:
            return 0

    def deleteTopics(self, tpcs):
        self._load_alphas()
        self._load_betas()
        self._load_thetas()
        self._load_betas_ds()
        self._load_topic_entropy()
        self._load_topic_coherence()
        self.load_tpc_descriptions()
        self.load_tpc_labels()
        self._load_ndocs_active()
        self._load_edits()
        self._load_vocab()

        try:
            tpc_keep = [k for k in range(self._ntopics) if k not in tpcs]
            tpc_keep = [k for k in tpc_keep if k < self._ntopics]

            self._thetas = self._thetas[:, tpc_keep]
            from sklearn.preprocessing import normalize
            self._thetas = normalize(self._thetas, axis=1, norm='l1')
            self._alphas = np.asarray(np.mean(self._thetas, axis=0)).ravel()
            self._ntopics = self._thetas.shape[1]
            self._betas = self._betas[tpc_keep, :]
            self._betas_ds = self._betas_ds[tpc_keep, :]
            self._ndocs_active = self._ndocs_active[tpc_keep]
            self._topic_entropy = self._topic_entropy[tpc_keep]
            self._topic_coherence = self._topic_coherence[tpc_keep]
            self._tpc_labels = [self._tpc_labels[i] for i in tpc_keep]
            self._tpc_descriptions = [self._tpc_descriptions[i] for i in tpc_keep]
            self._edits.append('d ' + ' '.join([str(k) for k in tpcs]))

            self._save_all()
            self._logger.info('-- -- Topics deletion successful. All variables saved to file')
            return 1
        except Exception:
            self._logger.info('-- -- Topics deletion generated an error. Operation failed')
            return 0

    def getSimilarTopics(self, npairs, thr=1e-3):
        self._load_thetas()
        self._load_betas()

        med = np.asarray(np.mean(self._thetas, axis=0)).ravel()
        thetas2 = self._thetas.multiply(self._thetas)
        med2 = np.asarray(np.mean(thetas2, axis=0)).ravel()
        stds = np.sqrt(med2 - med ** 2)
        num = self._thetas.T.dot(self._thetas).toarray() / self._thetas.shape[0]
        num = num - med[..., np.newaxis].dot(med[np.newaxis, ...])
        deno = stds[..., np.newaxis].dot(stds[np.newaxis, ...])
        corrcoef = num / deno
        selected_coocur = self._largest_indices(corrcoef, self._ntopics + 2 * npairs)
        selected_coocur = [(el[0], el[1], el[2].astype(float)) for el in selected_coocur]

        betas_aux = self._betas[:, np.where(self._betas.max(axis=0) > thr)[0]]
        js_mat = np.zeros((self._ntopics, self._ntopics))
        for k in range(self._ntopics):
            for kk in range(self._ntopics):
                js_mat[k, kk] = jensenshannon(betas_aux[k, :], betas_aux[kk, :])
        JSsim = 1 - js_mat
        selected_worddesc = self._largest_indices(JSsim, self._ntopics + 2 * npairs)
        selected_worddesc = [(el[0], el[1], el[2].astype(float)) for el in selected_worddesc]

        return {'Coocurring': selected_coocur, 'Worddesc': selected_worddesc}

    def getSimilarTopicsDicts(self, nsimilar: int = 5, thr: float = 1e-3):
        self._load_thetas()
        self._load_betas()

        med = np.asarray(np.mean(self._thetas, axis=0)).ravel()
        thetas2 = self._thetas.multiply(self._thetas)
        med2 = np.asarray(np.mean(thetas2, axis=0)).ravel()
        stds = np.sqrt(med2 - med ** 2)

        num = self._thetas.T.dot(self._thetas).toarray() / self._thetas.shape[0]
        num -= med[..., np.newaxis].dot(med[np.newaxis, ...])
        deno = stds[..., np.newaxis].dot(stds[np.newaxis, ...])
        corrcoef = num / deno

        coocur_sim = {}
        for i in range(self._ntopics):
            sim_row = corrcoef[i].copy()
            sim_row[i] = -np.inf
            top_indices = np.argsort(sim_row)[-nsimilar:][::-1]
            coocur_sim[i] = [(int(j), float(sim_row[j])) for j in top_indices]

        vocab_mask = self._betas.max(axis=0) > thr
        betas_aux = self._betas[:, vocab_mask]

        worddesc_sim = {}

        if betas_aux.shape[1] == 0:
            self._logger.warning("No vocab terms passed the threshold for JS computation.")
            worddesc_sim = {i: [] for i in range(self._ntopics)}
        else:
            js_mat = np.zeros((self._ntopics, self._ntopics))
            for k in range(self._ntopics):
                for kk in range(self._ntopics):
                    js_mat[k, kk] = jensenshannon(betas_aux[k, :], betas_aux[kk, :])
            JSsim = 1 - js_mat

            for i in range(self._ntopics):
                sim_row = JSsim[i].copy()
                sim_row[i] = -np.inf
                top_indices = np.argsort(sim_row)[-nsimilar:][::-1]
                worddesc_sim[i] = [(int(j), float(sim_row[j])) for j in top_indices]

        return {"Coocurring": coocur_sim, "Worddesc": worddesc_sim}

    def fuseTopics(self, tpcs):
        self._load_alphas()
        self._load_betas()
        self._load_thetas()
        self._load_topic_coherence()
        self.load_tpc_descriptions()
        self.load_tpc_labels()
        self._load_edits()
        self._load_vocab()

        try:
            tpcs = sorted(tpcs)

            weights = self._alphas[tpcs]
            bet = weights[np.newaxis, ...].dot(self._betas[tpcs, :]) / (sum(weights))
            self._betas[tpcs[0], :] = bet
            self._betas = np.delete(self._betas, tpcs[1:], 0)

            thetas_full = self._thetas.toarray()
            thet = np.sum(thetas_full[:, tpcs], axis=1)
            thetas_full[:, tpcs[0]] = thet
            thetas_full = np.delete(thetas_full, tpcs[1:], 1)
            self._thetas = sparse.csr_matrix(thetas_full, copy=True)

            self._alphas = np.asarray(np.mean(self._thetas, axis=0)).ravel()
            self._ntopics = self._thetas.shape[1]
            self._calculate_beta_ds()
            self._calculate_topic_entropy()
            self._ndocs_active = np.array((self._thetas != 0).sum(0).tolist()[0])

            for tpc in tpcs[1:][::-1]:
                del self._tpc_descriptions[tpc]
            self._tpc_descriptions[tpcs[0]] = self.get_tpc_word_descriptions(tpc=[tpcs[0]])[0][1]
            for tpc in tpcs[1:][::-1]:
                del self._tpc_labels[tpc]

            self.calculate_topic_coherence()
            self._edits.append('f ' + ' '.join([str(el) for el in tpcs]))
            self._save_all()
            self._logger.info('-- -- Topics merging successful. All variables saved to file')
            return 1
        except Exception:
            self._logger.info('-- -- Topics merging generated an error. Operation failed')
            return 0

    def sortTopics(self):
        self._load_alphas()
        self._load_betas()
        self._load_thetas()
        self._load_betas_ds()
        self._load_topic_entropy()
        self._load_topic_coherence()
        self.load_tpc_descriptions()
        self.load_tpc_labels()
        self._load_ndocs_active()
        self._load_edits()
        self._load_vocab()

        try:
            idx = np.argsort(self._alphas)[::-1]
            self._edits.append('s ' + ' '.join([str(el) for el in idx]))

            self._thetas = self._thetas[:, idx]
            self._alphas = self._alphas[idx]
            self._betas = self._betas[idx, :]
            self._betas_ds = self._betas_ds[idx, :]
            self._ndocs_active = self._ndocs_active[idx]
            self._topic_entropy = self._topic_entropy[idx]
            self._topic_coherence = self._topic_coherence[idx]
            self._tpc_labels = [self._tpc_labels[i] for i in idx]
            self._tpc_descriptions = [self._tpc_descriptions[i] for i in idx]
            self._edits.append('s ' + ' '.join([str(el) for el in idx]))

            self._save_all()
            self._logger.info('-- -- Topics reordering successful. All variables saved to file')
            return 1
        except Exception:
            self._logger.info('-- -- Topics reordering generated an error. Operation failed')
            return 0

    def resetTM(self):
        self._alphas_orig = np.load(self._TMfolder.joinpath('alphas_orig.npy'))
        self._betas_orig = np.load(self._TMfolder.joinpath('betas_orig.npy'))
        self._thetas_orig = sparse.load_npz(self._TMfolder.joinpath('thetas_orig.npz'))
        self._load_vocab()

        try:
            self.create(betas=self._betas_orig, thetas=self._thetas_orig,
                        alphas=self._alphas_orig, vocab=self._vocab)
            return 1
        except Exception:
            return 0

    def recalculate_cohrs(self):
        self.load_tpc_descriptions()
        try:
            self.calculate_topic_coherence()
            self._save_cohr()
            self._logger.info('-- -- Topics coherence recalculation successful.')
            return 1
        except Exception:
            self._logger.info('-- -- Topics coherence recalculation failed.')
            return 0

    def get_all_model_info(self, nsimilar: int = 5, thr: float = 1e-3, n_most: int = 20):
        self._load_alphas()
        self._load_betas()
        self._load_betas_ds()
        self._load_thetas()
        self._load_topic_entropy()
        self._load_topic_coherence()
        self.load_tpc_descriptions()
        self.load_tpc_labels()
        self.load_tpc_summaries()
        self._load_ndocs_active()
        self._load_vocab()
        self._load_vocab_dicts()
        self.load_topic_documents(mode="most_representative", n_most=n_most)
        self.load_topic_documents(mode="clusters")
        self.load_tpc_coords()
        irbo = self.calculate_rbo()
        td = self.calculate_topic_diversity()
        similar = self.getSimilarTopicsDicts(nsimilar=nsimilar, thr=thr)
        thetas_rpr = self.load_thetas_representation()

        data = {
            "Size": [self._alphas],
            "Entropy": [self._topic_entropy],
            "Coherence (NPMI)": [self._topic_coherence],
            "# Docs Active": [self._ndocs_active],
            "Keywords": [self._tpc_descriptions],
            "Label": [self._tpc_labels],
            "Summary": [self._tpc_summaries],
            "Top Documents": [self._most_representative_docs],
            "Assigned Documents": [self._tpc_clusters],
            "Coordinates": [self._coords],
        }
        df = pd.DataFrame(data)
        df = df.apply(pd.Series.explode)
        df["Size"] = df["Size"].apply(lambda x: f"{x:.2%}")
        df["Top Documents"] = df["Top Documents"].apply(
            lambda x: {i[0]: float(i[2]) for i in x}
        )
        df["Assigned Documents"] = df["Assigned Documents"].apply(
            lambda x: {i[0]: float(i[2]) for i in x}
        )
        df = df.reset_index(drop=True)
        df["ID"] = df.index

        return df, self._vocab_id2w, self._vocab, irbo, td, similar, thetas_rpr
