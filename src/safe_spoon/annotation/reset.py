"""Manual reset of the annotation database."""

import sqlite3
from typing import Dict, List, Optional

import click

from safe_spoon.annotation.store import get_connection, init_schema
from safe_spoon.utils.common import load_yaml_config_file

# Deletion order matters
_CATEGORY_SCOPED_TABLES = ("llm_rubric_suggestions", "rubrics", "risk_profiles")

def reset_db(conn: sqlite3.Connection, categories: Optional[List[str]] = None) -> Dict[str, int]:
    """Delete rubrics (+ their criteria/cells via cascade), risk profiles,
    and LLM rubric suggestions.
    """
    counts: Dict[str, int] = {}
    for table in _CATEGORY_SCOPED_TABLES:
        if categories:
            placeholders = ",".join("?" for _ in categories)
            cur = conn.execute(f"DELETE FROM {table} WHERE category IN ({placeholders})", categories)
        else:
            cur = conn.execute(f"DELETE FROM {table}")
        counts[table] = cur.rowcount
    conn.commit()
    return counts


@click.command()
@click.option("--config", "config_path", default="config/config.yaml", help="Path to config.yaml (used to resolve the DB path)")
@click.option("--db-path", default=None, help="Path to annotation.db (overrides config.yaml's paths.annotation_db)")
@click.option("--category", "categories", multiple=True, help="Only reset these categories (default: reset everything)")
@click.option("--yes", is_flag=True, default=False, help="Skip the confirmation prompt")
def main(config_path, db_path, categories, yes):
    """Wipe rubrics, risk profiles and LLM suggestions from the annotation database."""
    if db_path is None:
        db_path = load_yaml_config_file(config_path)["annotation_db"]

    scope = f"categories {list(categories)}" if categories else "ALL categories"
    if not yes:
        click.confirm(
            f"This will permanently delete rubrics, risk profiles, LLM suggestions "
            f"for {scope} in {db_path}. Continue?",
            abort=True,
        )

    conn = get_connection(db_path)
    init_schema(conn)
    counts = reset_db(conn, list(categories) if categories else None)
    conn.close()

    for table, n in counts.items():
        click.echo(f"  {table}: {n} rows deleted")
    click.echo("Done.")


if __name__ == "__main__":
    main()
