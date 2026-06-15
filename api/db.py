"""
DuckDB query layer.

DuckDB is embedded — one connection per process is the recommended pattern
for read-heavy workloads. We open in read-only mode so the API never
accidentally mutates the database.

Usage:
    from db import fetch_all, fetch_one, fetch_scalar

    rows = fetch_all("SELECT * FROM horses WHERE horse_id = ?", [horse_id])
"""

import duckdb
import threading
from contextlib import contextmanager
from typing import Any

from config import settings

# ---------------------------------------------------------------------------
# Connection pool (thread-local read-only connections)
# DuckDB supports multiple read-only connections from the same process.
# ---------------------------------------------------------------------------

_local = threading.local()


def _get_conn() -> duckdb.DuckDBPyConnection:
    """Return a thread-local read-only connection, creating it if needed."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = duckdb.connect(settings.db_path, read_only=True)
    return _local.conn


@contextmanager
def get_cursor():
    """Context manager that yields a cursor and handles cleanup."""
    conn = _get_conn()
    cursor = conn.cursor()
    try:
        yield cursor
    finally:
        cursor.close()


# ---------------------------------------------------------------------------
# Public query helpers — always return plain dicts (JSON-serialisable)
# ---------------------------------------------------------------------------

def _row_to_dict(cursor: duckdb.DuckDBPyConnection, row: tuple) -> dict:
    """Convert a result row to a dict keyed by column name."""
    cols = [desc[0] for desc in cursor.description]
    return dict(zip(cols, row))


def fetch_all(
    sql: str,
    params: list[Any] | None = None,
) -> list[dict]:
    """Execute *sql* and return all rows as a list of dicts."""
    with get_cursor() as cur:
        cur.execute(sql, params or [])
        rows = cur.fetchall()
        return [_row_to_dict(cur, r) for r in rows]


def fetch_one(
    sql: str,
    params: list[Any] | None = None,
) -> dict | None:
    """Execute *sql* and return the first row as a dict, or None."""
    with get_cursor() as cur:
        cur.execute(sql, params or [])
        row = cur.fetchone()
        if row is None:
            return None
        return _row_to_dict(cur, row)


def fetch_scalar(
    sql: str,
    params: list[Any] | None = None,
) -> Any:
    """Execute *sql* and return the first column of the first row."""
    with get_cursor() as cur:
        cur.execute(sql, params or [])
        row = cur.fetchone()
        return row[0] if row else None


def paginate(
    base_sql: str,
    params: list[Any] | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """
    Wrap *base_sql* in a count + paginated fetch.

    Returns:
        {
            "total": <int>,
            "limit": <int>,
            "offset": <int>,
            "data": [<dict>, ...]
        }
    """
    count_sql = f"SELECT COUNT(*) FROM ({base_sql}) _q"
    total = fetch_scalar(count_sql, params)

    paged_sql = f"{base_sql} LIMIT {int(limit)} OFFSET {int(offset)}"
    data = fetch_all(paged_sql, params)

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "data": data,
    }
