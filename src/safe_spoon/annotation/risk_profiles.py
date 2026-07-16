"""Database operations for risk profiles."""

import sqlite3
from datetime import datetime, timezone
from typing import List, Optional

from safe_spoon.annotation.models import RiskProfile


class RiskProfileInUseError(Exception):
    """Raised when deleting a risk profile still referenced by rubric cells."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_profile(row: sqlite3.Row) -> RiskProfile:
    return RiskProfile(
        id=row["id"],
        category=row["category"],
        name=row["name"],
        description=row["description"],
        severity_rank=row["severity_rank"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def list_risk_profiles(conn: sqlite3.Connection, category: str) -> List[RiskProfile]:
    rows = conn.execute(
        "SELECT * FROM risk_profiles WHERE category = ? ORDER BY severity_rank",
        (category,),
    ).fetchall()
    return [_row_to_profile(r) for r in rows]


def create_risk_profile(
    conn: sqlite3.Connection,
    category: str,
    name: str,
    description: str,
    severity_rank: int,
) -> RiskProfile:
    now = _now()
    cur = conn.execute(
        """INSERT INTO risk_profiles (category, name, description, severity_rank, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (category, name, description, severity_rank, now, now),
    )
    conn.commit()
    return get_risk_profile(conn, cur.lastrowid)


def get_risk_profile(conn: sqlite3.Connection, profile_id: int) -> Optional[RiskProfile]:
    row = conn.execute("SELECT * FROM risk_profiles WHERE id = ?", (profile_id,)).fetchone()
    return _row_to_profile(row) if row else None


def update_risk_profile(conn: sqlite3.Connection, profile_id: int, **fields) -> Optional[RiskProfile]:
    if not fields:
        return get_risk_profile(conn, profile_id)
    allowed = {"name", "description", "severity_rank"}
    set_clause = ", ".join(f"{k} = ?" for k in fields if k in allowed)
    values = [v for k, v in fields.items() if k in allowed]
    if not set_clause:
        return get_risk_profile(conn, profile_id)
    values.extend([_now(), profile_id])
    conn.execute(
        f"UPDATE risk_profiles SET {set_clause}, updated_at = ? WHERE id = ?",
        values,
    )
    conn.commit()
    return get_risk_profile(conn, profile_id)


def delete_risk_profile(conn: sqlite3.Connection, profile_id: int) -> None:
    in_use = conn.execute(
        "SELECT 1 FROM rubric_cells WHERE risk_profile_id = ? LIMIT 1", (profile_id,)
    ).fetchone()
    if in_use:
        raise RiskProfileInUseError(
            f"Risk profile {profile_id} is referenced by existing rubric cells and cannot be deleted."
        )
    conn.execute("DELETE FROM risk_profiles WHERE id = ?", (profile_id,))
    conn.commit()
