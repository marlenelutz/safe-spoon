"""Shared state for the Flask app"""

import csv
import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

from flask import current_app

from safe_spoon.annotation import store as annotation_store

log = logging.getLogger(__name__)


class AppState:
    # CSV column names used before configurable labels were introduced —
    # kept so old labels.csv files still load correctly.
    _LEGACY_CSV_COLS = {"trash": "TRASH_PRESENT",
                        "demo": "DEMOGRAPHICS_DESCRIBED"}

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.data_file = cfg["output_json"]
        self.labels_file = cfg["labels_csv"]
        self.annotation_db = cfg["annotation_db"]
        self.query_labels = cfg["query_labels"]
        self.ui_display = cfg["ui_display"]

        self._data: Optional[dict] = None
        self._data_mtime: Optional[float] = None
        self.labels: dict = {}
        self.unit_cache: dict = {}     # cat -> serialised annotation-unit payload
        self.safety_cache: dict = {}   # cat -> {unit_node_id: float}
        self.safety_status: dict = {}  # cat -> 'pending' | 'running' | 'done' | 'disabled'
        self.safety_evaluator = None   # set by webapp/safety_routes.build_safety_evaluator()
        # cat -> {node_id: [global_leaf_idx, ...]}, see rubric_routes
        self.leaf_index_cache: dict = {}

        self.load_labels()

        # Create the schema once at startup; individual requests each open
        # their own short-lived connection (see annotation_conn()) rather
        # than sharing one across threads.
        conn = annotation_store.get_connection(self.annotation_db)
        annotation_store.init_schema(conn)
        conn.close()

    def annotation_conn(self) -> sqlite3.Connection:
        """A fresh connection to the annotation DB, for one request's use."""
        return annotation_store.get_connection(self.annotation_db)

    def get_data(self) -> dict:
        mtime = Path(self.data_file).stat().st_mtime
        if self._data is None or mtime != self._data_mtime:
            log.info("Loading data from %s", self.data_file)
            self._data = json.load(open(self.data_file, encoding="utf-8"))
            self._data_mtime = mtime
            # Pipeline reran and rewrote output_json: any cached, tree-derived
            # state (unit ids, safety scores, leaf indices) is now stale.
            self.unit_cache = {}
            self.safety_cache = {}
            self.safety_status = {}
            self.leaf_index_cache = {}
            log.info("Data loaded.")
        return self._data

    def _write_labels_csv(self, writer, data_by_category) -> None:
        field_cols = [lf["field"].upper() for lf in self.query_labels]
        writer.writerow(
            ["query_id", "category", "query_index", "query"] + field_cols)
        for key, state in self.labels.items():
            cat, gi_str = key.rsplit(":", 1)
            gi = int(gi_str)
            info = data_by_category.get(cat, {})
            ids = info.get("query_ids") or []
            qs = info.get("queries") or []
            qid = ids[gi] if gi < len(ids) else gi
            text = qs[gi] if gi < len(qs) else ""
            vals = [state.get(lf["field"], False) for lf in self.query_labels]
            writer.writerow([qid, cat, gi, text] + vals)

    def write_labels_csv(self, writer) -> None:
        self._write_labels_csv(writer, self.get_data()["data_by_category"])

    def load_labels(self) -> None:
        self.labels = {}
        p = Path(self.labels_file)
        if not p.exists():
            return
        with p.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = f"{row['category']}:{row['query_index']}"
                state = {}
                for lf in self.query_labels:
                    field = lf["field"]
                    col = field.upper()
                    legacy = self._LEGACY_CSV_COLS.get(field)
                    if col in row:
                        state[field] = row[col] == "True"
                    elif legacy and legacy in row:
                        state[field] = row[legacy] == "True"
                    else:
                        state[field] = False
                self.labels[key] = state

    def save_labels(self) -> None:
        p = Path(self.labels_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="", encoding="utf-8") as f:
            self.write_labels_csv(csv.writer(f))


def get_state() -> AppState:
    return current_app.extensions["safespoon"]
