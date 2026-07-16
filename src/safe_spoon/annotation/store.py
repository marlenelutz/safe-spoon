"""SQLite storage for the annotation/rubric system.

Rubrics have real relational structure (rubric -> criteria -> cells) and are
queried by category/unit/annotator. Risk profiles are a fixed set of rows per
category, and are referenced by rubric cells. LLM rubric suggestions are
stored as JSON blobs, keyed by category/unit/annotator/candidate_index.
"""

import sqlite3
from pathlib import Path

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS risk_profiles (
    id            INTEGER PRIMARY KEY,
    category      TEXT NOT NULL,
    name          TEXT NOT NULL,
    description   TEXT NOT NULL DEFAULT '',
    severity_rank INTEGER NOT NULL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    UNIQUE(category, name)
);

CREATE TABLE IF NOT EXISTS rubrics (
    id                    INTEGER PRIMARY KEY,
    category              TEXT NOT NULL,
    unit_stable_id        TEXT NOT NULL,
    annotator             TEXT NOT NULL,
    status                TEXT NOT NULL CHECK(status IN ('draft','submitted','confirmed')),
    source                TEXT NOT NULL CHECK(source IN ('llm_suggestion','manual')),
    created_at            TEXT NOT NULL,
    updated_at            TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_rubrics_lookup
    ON rubrics(category, unit_stable_id, annotator);

CREATE TABLE IF NOT EXISTS rubric_criteria (
    id          INTEGER PRIMARY KEY,
    rubric_id   INTEGER NOT NULL REFERENCES rubrics(id) ON DELETE CASCADE,
    order_index INTEGER NOT NULL,
    title       TEXT NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS rubric_cells (
    id                     INTEGER PRIMARY KEY,
    rubric_id              INTEGER NOT NULL REFERENCES rubrics(id) ON DELETE CASCADE,
    criterion_id           INTEGER NOT NULL REFERENCES rubric_criteria(id) ON DELETE CASCADE,
    risk_profile_id        INTEGER NOT NULL REFERENCES risk_profiles(id),
    expected_behavior      TEXT,
    risk_signals           TEXT,
    inherited_from_cell_id INTEGER REFERENCES rubric_cells(id),
    is_override            INTEGER NOT NULL DEFAULT 0,
    UNIQUE(rubric_id, criterion_id, risk_profile_id)
);

CREATE TABLE IF NOT EXISTS llm_rubric_suggestions (
    id              INTEGER PRIMARY KEY,
    category        TEXT NOT NULL,
    unit_stable_id  TEXT NOT NULL,
    annotator       TEXT NOT NULL,
    candidate_index INTEGER NOT NULL,
    criteria_json   TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
"""


def get_connection(db_path: str = "data/output/annotation.db") -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()
