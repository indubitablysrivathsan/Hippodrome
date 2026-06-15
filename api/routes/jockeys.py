from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from api.db import fetch_all, fetch_one, paginate
from config import settings

router = APIRouter(prefix="/jockeys", tags=["Jockeys"])


@router.get("/", summary="List / search jockeys")
def list_jockeys(
    name: Optional[str] = Query(None, description="Partial name search"),
    limit: int = Query(settings.default_limit, le=settings.max_limit),
    offset: int = Query(0, ge=0),
):
    filters = []
    params: list = []
    if name:
        filters.append("LOWER(jockey_name) LIKE LOWER(?)")
        params.append(f"%{name}%")
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    sql = f"SELECT * FROM jockeys {where} ORDER BY jockey_name"
    return paginate(sql, params=params, limit=limit, offset=offset)


@router.get("/{jockey_id}", summary="Get jockey by ID")
def get_jockey(jockey_id: int):
    row = fetch_one("SELECT * FROM jockeys WHERE jockey_id = ?", [jockey_id])
    if not row:
        raise HTTPException(status_code=404, detail="Jockey not found")
    return row


@router.get("/{jockey_id}/rides", summary="Race history for a jockey")
def jockey_rides(
    jockey_id: int,
    limit: int = Query(settings.default_limit, le=settings.max_limit),
    offset: int = Query(0, ge=0),
):
    _check_jockey(jockey_id)
    sql = """
        SELECT *
        FROM v_runner_full
        WHERE jockey_id = ?
        ORDER BY meet_date DESC
    """
    return paginate(sql, params=[jockey_id], limit=limit, offset=offset)


@router.get("/{jockey_id}/penalties", summary="Penalties issued to a jockey")
def jockey_penalties(jockey_id: int):
    _check_jockey(jockey_id)
    rows = fetch_all(
        "SELECT * FROM penalties WHERE jockey_id = ? ORDER BY race_id DESC",
        [jockey_id],
    )
    return {"jockey_id": jockey_id, "data": rows}


@router.get("/{jockey_id}/stats", summary="Win/place stats for a jockey")
def jockey_stats(jockey_id: int):
    _check_jockey(jockey_id)
    row = fetch_one(
        """
        SELECT
            vrf.jockey_id,
            vrf.jockey_name,
            COUNT(*)                                                         AS total_rides,
            COUNT(*) FILTER (WHERE vrf.finish_position = 1)                  AS wins,
            COUNT(*) FILTER (WHERE vrf.finish_position <= 3)                 AS places,
            ROUND(
                COUNT(*) FILTER (WHERE vrf.finish_position = 1) * 100.0
                / NULLIF(COUNT(*), 0), 2
            )                                                                AS win_pct
        FROM v_runner_full vrf
        WHERE vrf.jockey_id = ?
        GROUP BY vrf.jockey_id, vrf.jockey_name
        """,
        [jockey_id],
    )
    return row or {"jockey_id": jockey_id, "total_rides": 0}


def _check_jockey(jockey_id: int):
    if not fetch_one("SELECT jockey_id FROM jockeys WHERE jockey_id = ?", [jockey_id]):
        raise HTTPException(status_code=404, detail="Jockey not found")
