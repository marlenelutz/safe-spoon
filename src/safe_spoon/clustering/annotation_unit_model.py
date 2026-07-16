"""Same concept as the TMmodel but for annotation units instead of topics.  Builds the unit tree and generates LLM labels/notes.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .annotation_units import (
    build_unit_tree,
    compute_leaf_indices,
    generate_unit_labels,
    make_unit_label,
)


class AnnotationUnitModel:
    def __init__(
        self,
        flat_nodes: List[dict],
        root_id: str,
        thetas: np.ndarray,
        topic_keys: List[List[str]],
        topic_labels: List[str],
        queries: List[str],
        model_path: Optional[str] = None,
        min_size: int = 20,
        max_rel_dist: float = 0.40,
        llm_provider: Optional[str] = None,
        llm_model_type: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_server: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self._flat_nodes = flat_nodes
        self._root_id = root_id
        self._thetas = np.array(thetas, dtype=np.float32)
        self._topic_keys = topic_keys or []
        self._topic_labels = topic_labels or []
        self._queries = queries
        self._model_path = Path(model_path) if model_path else None
        self._min_size    = min_size
        self._max_rel_dist = max_rel_dist
        self._llm_provider = llm_provider
        self._llm_model_type = llm_model_type
        self._llm_api_key = llm_api_key
        self._llm_server = llm_server
        self._logger = logger or logging.getLogger(__name__)

        self._unit_tree: Optional[dict] = None
        self._n_units: int = 0
        self._unit_labels: Dict[str, Dict[str, str]] = {}

    @classmethod
    def from_lda(
        cls,
        lda,
        queries: Optional[List[str]] = None,
        linkage_method: str = "average",
        n_repr: int = 5,
        **kwargs,
    ) -> "AnnotationUnitModel":
        """Recomputes the Bhattacharyya distance matrix and agglomerative clustering from a loaded LDATopicModel, and builds the annotation-unit tree.

        Parameters
        ----------
        lda:
            A loaded LDATopicModel instance (corpus already injected).
        queries:
            Raw query texts in training order.  If None, extracted from
            lda._df["text"] (requires the model to have been loaded
            with corpus=).
        **kwargs:
            Forwarded as it comes to the constructor (e.g. min_size,
            max_rel_dist, llm_provider, ...).
        """
        from safe_spoon.clustering.hierarchical import build_flat_tree

        if queries is None:
            queries, _ = lda.get_ordered_corpus()

        thetas = lda.get_thetas()
        topic_keys = lda.get_topic_keys()

        lda.tm.load_tpc_labels()
        topic_labels = getattr(lda.tm, "_tpc_labels", None)
        if not topic_labels or all(l.startswith("Topic ") for l in topic_labels):
            topic_labels = [" · ".join(kw[:3]) for kw in topic_keys]

        flat_nodes, root_id, _ = build_flat_tree(
            thetas, list(range(len(queries))), queries,
            linkage_method=linkage_method, n_repr=n_repr,
        )

        model_path = kwargs.pop("model_path", str(lda.model_path))
        return cls(
            flat_nodes=flat_nodes,
            root_id=root_id,
            thetas=thetas,
            topic_keys=topic_keys,
            topic_labels=topic_labels,
            queries=queries,
            model_path=model_path,
            llm_provider=kwargs.pop("llm_provider", lda.llm_provider),
            llm_model_type=kwargs.pop("llm_model_type", lda.llm_model_type),
            llm_api_key=kwargs.pop("llm_api_key", lda.llm_api_key),
            llm_server=kwargs.pop("llm_server", lda.llm_server),
            **kwargs,
        )

    def build(self) -> "AnnotationUnitModel":
        """Builds the annotation-unit tree from the stored flat nodes."""
        nodes_by_id = {n["id"]: n for n in self._flat_nodes}
        leaf_id = compute_leaf_indices(nodes_by_id, self._root_id)
        self._unit_tree, self._n_units = build_unit_tree(
            nodes_by_id,
            self._root_id,
            leaf_id,
            self._thetas,
            min_size = self._min_size,
            max_rel_dist = self._max_rel_dist,
            topic_labels = self._topic_labels,
            topic_keys = self._topic_keys,
        )
        return self
    
    def generate_unit_outputs(
        self,
        topn_docs: int = 5,
        prompt_path: Optional[str] = None,
        max_retries: int = 3,
    ) -> List[Tuple[str, Dict[str, str]]]:
        """Generate LLM labels + annotation notes for every annotation unit.

        Returns
        -------
        List[Tuple[str, Dict[str, str]]]
            [(node_id, {"label": ..., "note": ...}), ...] sorted by node_id.
        """
        if self._unit_tree is None:
            self.build()

        from safe_spoon.prompting import Prompter, _default_prompt_path

        if prompt_path is None:
            prompt_path = _default_prompt_path("unit_labelling_dft.txt")

        prompt_template = Path(str(prompt_path)).read_text(encoding="utf-8")

        self._logger.info(
            "Generating unit labels via %s / %s",
            self._llm_provider,
            self._llm_model_type,
        )
        prompter = Prompter(
            model_type  = self._llm_model_type,
            llm_provider = self._llm_provider,
            api_key = self._llm_api_key,
            llm_server = self._llm_server,
        )

        self._unit_labels = generate_unit_labels(
            self._unit_tree,
            self._queries,
            topic_keys = self._topic_keys,
            topic_labels = self._topic_labels,
            prompter = prompter,
            prompt_template = prompt_template,
            topn_docs = topn_docs,
            max_retries = max_retries,
        )

        return sorted(self._unit_labels.items())

    @property
    def _labels_path(self) -> Optional[Path]:
        if self._model_path is None:
            return None
        return self._model_path / "TMmodel" / "unit_labels.json"

    @property
    def _params_path(self) -> Optional[Path]:
        if self._model_path is None:
            return None
        return self._model_path / "TMmodel" / "unit_tree_params.json"

    def save_params(self) -> None:
        """Persist the min_size/max_rel_dist thresholds used to build this unit tree, alongside the labels they were generated for.
        """
        p = self._params_path
        if p is None:
            self._logger.warning("No model_path set; unit tree params not saved.")
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "min_size": self._min_size,
                    "max_rel_dist": self._max_rel_dist,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self._logger.info("Unit tree params saved to %s", p)

    @staticmethod
    def load_saved_params(model_path: Optional[str]) -> Optional[Dict[str, float]]:
        """Load the min_size/max_rel_dist thresholds a unit tree
        was last built with"""
        
        if not model_path:
            return None
        p = Path(model_path) / "TMmodel" / "unit_tree_params.json"
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def load_unit_labels(self) -> bool:
        """Load labels from {model_path}/TMmodel/unit_labels.json."""
        p = self._labels_path
        if p is None or not p.exists():
            return False
        self._unit_labels = json.loads(p.read_text(encoding="utf-8"))
        self._logger.info("Unit labels loaded from %s (%d entries)", p, len(self._unit_labels))
        return True

    def save_unit_labels(self) -> None:
        """Persist current unit labels to {model_path}/TMmodel/unit_labels.json."""
        p = self._labels_path
        if p is None:
            self._logger.warning("No model_path set; unit labels not saved.")
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self._unit_labels, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._logger.info("Unit labels saved to %s", p)

    @property
    def unit_tree(self) -> Optional[dict]:
        return self._unit_tree

    @property
    def n_units(self) -> int:
        return self._n_units

    @property
    def unit_labels(self) -> Dict[str, Dict[str, str]]:
        return dict(self._unit_labels)

    def get_unit_label(self, node_id: str) -> str:
        """Return the stored LLM label for node_id, falling back to the rule-based label."""
        entry = self._unit_labels.get(node_id)
        if entry and entry.get("label"):
            return entry["label"]

        # if it does not work, then derive from mean_theta via the rule-based function
        nodes_by_id = {n["id"]: n for n in self._flat_nodes}
        if node_id not in nodes_by_id:
            return ""
        leaf_id = compute_leaf_indices(nodes_by_id, self._root_id)
        ids = leaf_id.get(node_id, [])
        if not ids:
            return ""
        mean_theta = self._thetas[ids].mean(axis=0).tolist()
        return make_unit_label(mean_theta, self._topic_labels, self._topic_keys)

    def get_unit_note(self, node_id: str) -> str:
        """Return the stored annotation note for node_id, or "" if none was generated."""
        entry = self._unit_labels.get(node_id)
        return entry.get("note", "") if entry else ""