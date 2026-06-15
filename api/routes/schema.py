"""
Schema introspection endpoints.
Useful for dashboard tooling and dynamic query builders.
"""

from fastapi import APIRouter, HTTPException, Query
from api.db import fetch_all, fetch_one, fetch_scalar, paginate
from config import settings

router = APIRouter(prefix="/schema", tags=["Schema & Stats"])


@router.get("/tables", summary="List all tables and views")
def list_tables():
    """Return every user table and view with row counts for tables."""
    tables = fetch_all(
        """
        SELECT table_name, table_type
        FROM information_schema.tables
        WHERE table_schema = 'main'
        ORDER BY table_type DESC, table_name
        """
    )
    return {"data": tables, "total": len(tables)}


@router.get("/tables/{table_name}", summary="Columns for a table or view")
def table_columns(table_name: str):
    """Return column names, types and nullability for the given table/view."""
    rows = fetch_all(
        """
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_schema = 'main'
          AND table_name = ?
        ORDER BY ordinal_position
        """,
        [table_name],
    )
    if not rows:
        raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found")
    return {"table_name": table_name, "columns": rows}


@router.get("/stats", summary="High-level database statistics")
def db_stats():
    """Return row counts for every core table — useful for a dashboard header."""
    counts = {}
    tables = [
        "venues", "horses", "horse_aliases", "jockeys", "trainers",
        "meetings", "races", "runners",
        "runner_acceptances", "runner_declarations",
        "ratings_changes", "penalties", "exotics", "race_dividends",
        "horse_medical", "horse_treadmill", "horse_remarks", "horse_actions",
        "jockey_changes", "equipment_changes",
    ]
    for tbl in tables:
        counts[tbl] = fetch_scalar(f"SELECT COUNT(*) FROM {tbl}")
    return counts


@router.get("/recent-activity", summary="Last N meetings with race counts")
def recent_activity(limit: int = Query(10, le=50)):
    rows = fetch_all(
        """
        SELECT
            m.meet_date,
            v.venue_name,
            m.track_condition,
            m.weather,
            COUNT(r.race_id) AS race_count
        FROM meetings m
        JOIN venues v ON v.venue_id = m.venue_id
        LEFT JOIN races r ON r.meet_date = m.meet_date AND r.venue_id = m.venue_id
        GROUP BY m.meet_date, v.venue_name, m.track_condition, m.weather
        ORDER BY m.meet_date DESC
        LIMIT ?
        """,
        [limit],
    )
    return {"data": rows}
