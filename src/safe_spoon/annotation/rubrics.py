"""Database operations for rubrics and their criteria/cells."""

import json
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from safe_spoon.annotation.models import Rubric, RubricCell, RubricCriterion


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_rubric(conn: sqlite3.Connection, rubric_id: int) -> Optional[Rubric]:
    row = conn.execute("SELECT * FROM rubrics WHERE id = ?", (rubric_id,)).fetchone()
    if row is None:
        return None
    rubric = Rubric(
        id=row["id"],
        category=row["category"],
        unit_stable_id=row["unit_stable_id"],
        annotator=row["annotator"],
        status=row["status"],
        source=row["source"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
    criteria_rows = conn.execute(
        "SELECT * FROM rubric_criteria WHERE rubric_id = ? ORDER BY order_index", (rubric_id,)
    ).fetchall()
    for crow in criteria_rows:
        cells = conn.execute(
            "SELECT * FROM rubric_cells WHERE rubric_id = ? AND criterion_id = ?",
            (rubric_id, crow["id"]),
        ).fetchall()
        rubric.criteria.append(RubricCriterion(
            id=crow["id"],
            order_index=crow["order_index"],
            title=crow["title"],
            description=crow["description"] or "",
            cells=[
                RubricCell(
                    id=cell["id"],
                    criterion_id=cell["criterion_id"],
                    risk_profile_id=cell["risk_profile_id"],
                    expected_behavior=cell["expected_behavior"] or "",
                    risk_signals=cell["risk_signals"] or "",
                    inherited_from_cell_id=cell["inherited_from_cell_id"],
                    is_override=bool(cell["is_override"]),
                )
                for cell in cells
            ],
        ))
    return rubric


def get_rubric(conn: sqlite3.Connection, category: str, unit_stable_id: str, annotator: str) -> Optional[Rubric]:
    """Return the most recent (draft or submitted) rubric for this unit/annotator, if any."""
    row = conn.execute(
        """SELECT id FROM rubrics
           WHERE category = ? AND unit_stable_id = ? AND annotator = ?
           ORDER BY updated_at DESC LIMIT 1""",
        (category, unit_stable_id, annotator),
    ).fetchone()
    return _load_rubric(conn, row["id"]) if row else None


def get_or_create_draft(
    conn: sqlite3.Connection,
    category: str,
    unit_stable_id: str,
    annotator: str,
) -> Rubric:
    existing = get_rubric(conn, category, unit_stable_id, annotator)
    if existing is not None:
        return existing

    now = _now()
    cur = conn.execute(
        """INSERT INTO rubrics
           (category, unit_stable_id, annotator, status, source, created_at, updated_at)
           VALUES (?, ?, ?, 'draft', 'manual', ?, ?)""",
        (category, unit_stable_id, annotator, now, now),
    )
    conn.commit()
    return _load_rubric(conn, cur.lastrowid)


def _cell_text(v) -> str:
    """Coerce a cell field to a string for sqlite binding.

    Client data ultimately traces back to LLM JSON output (see
    suggestions.py), which occasionally returns a list for a field the
    prompt asked for as prose — sqlite3 can't bind a list, so normalize here
    at the DB boundary regardless of what the client already does.
    """
    if v is None:
        return ""
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    return str(v)


def submit_rubric(
    conn: sqlite3.Connection,
    rubric_id: int,
    criteria: List[dict],
    source: str = "manual",
) -> Rubric:
    """Overwrite this rubric's criteria/cells and mark it 'submitted'.

    criteria: [{"title": str, "description": str, "cells": {risk_profile_id: {
        "expected_behavior": str, "risk_signals": str,
        "inherited_from_cell_id": int | None, "is_override": bool
    }}}]
    """
    now = _now()
    conn.execute("DELETE FROM rubric_criteria WHERE rubric_id = ?", (rubric_id,))
    # ON DELETE CASCADE on rubric_cells.criterion_id handles cell cleanup.

    for order_index, crit in enumerate(criteria):
        cur = conn.execute(
            "INSERT INTO rubric_criteria (rubric_id, order_index, title, description) VALUES (?, ?, ?, ?)",
            (rubric_id, order_index, crit["title"], crit.get("description", "")),
        )
        criterion_id = cur.lastrowid
        for risk_profile_id, cell in crit.get("cells", {}).items():
            conn.execute(
                """INSERT INTO rubric_cells
                   (rubric_id, criterion_id, risk_profile_id, expected_behavior, risk_signals,
                    inherited_from_cell_id, is_override)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    rubric_id, criterion_id, int(risk_profile_id),
                    _cell_text(cell.get("expected_behavior")), _cell_text(cell.get("risk_signals")),
                    cell.get("inherited_from_cell_id"), int(bool(cell.get("is_override", False))),
                ),
            )

    conn.execute(
        "UPDATE rubrics SET status = 'submitted', source = ?, updated_at = ? WHERE id = ?",
        (source, now, rubric_id),
    )
    conn.commit()
    return _load_rubric(conn, rubric_id)


def confirm_rubric(conn: sqlite3.Connection, rubric_id: int) -> Rubric:
    conn.execute(
        "UPDATE rubrics SET status = 'confirmed', updated_at = ? WHERE id = ?",
        (_now(), rubric_id),
    )
    conn.commit()
    return _load_rubric(conn, rubric_id)


def find_similar_confirmed_rubrics(
    conn: sqlite3.Connection,
    category: str,
    mean_theta: List[float],
    unit_thetas: Dict[str, List[float]],
    top_k: int = 3,
) -> List[tuple]:
    """Find confirmed rubrics in this category whose unit's mean_theta is
    close to mean_theta, for the "use this as a starting point / merge"
    suggestion.

    unit_thetas: {unit_stable_id: mean_theta} for every unit that might have
    a confirmed rubric — supplied by the caller (the webapp layer), since
    mean_theta lives in the pipeline's JSON artifact, not in this DB.

    Returns [(Rubric, cosine_similarity), ...] sorted by similarity descending.
    """
    rows = conn.execute(
        "SELECT DISTINCT unit_stable_id, id FROM rubrics WHERE category = ? AND status = 'confirmed'",
        (category,),
    ).fetchall()
    if not rows:
        return []

    query_vec = np.array(mean_theta, dtype=np.float32)
    query_norm = np.linalg.norm(query_vec) + 1e-10

    scored = []
    for row in rows:
        theta = unit_thetas.get(row["unit_stable_id"])
        if theta is None:
            continue
        vec = np.array(theta, dtype=np.float32)
        sim = float(np.dot(query_vec, vec) / (query_norm * (np.linalg.norm(vec) + 1e-10)))
        scored.append((row["id"], sim))

    scored.sort(key=lambda x: -x[1])
    return [(_load_rubric(conn, rubric_id), sim) for rubric_id, sim in scored[:top_k]]
