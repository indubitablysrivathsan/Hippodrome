from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from api.db import fetch_all, fetch_one, paginate
from config import settings

router = APIRouter(prefix="/horses", tags=["Horses"])


@router.get("/", summary="List / search horses")
def list_horses(
    name: Optional[str] = Query(None, description="Partial horse name search"),
    sire: Optional[str] = Query(None, description="Partial sire name search"),
    dam: Optional[str] = Query(None, description="Partial dam name search"),
    limit: int = Query(settings.default_limit, le=settings.max_limit),
    offset: int = Query(0, ge=0),
):
    filters = []
    params: list = []

    if name:
        filters.append("LOWER(horse_name) LIKE LOWER(?)")
        params.append(f"%{name}%")
    if sire:
        filters.append("LOWER(sire) LIKE LOWER(?)")
        params.append(f"%{sire}%")
    if dam:
        filters.append("LOWER(dam) LIKE LOWER(?)")
        params.append(f"%{dam}%")

    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    sql = f"SELECT * FROM horses {where} ORDER BY horse_name"
    return paginate(sql, params=params, limit=limit, offset=offset)


@router.get("/search", summary="Quick name lookup (includes aliases)")
def search_horse(
    q: str = Query(..., description="Horse name or alias"),
    limit: int = Query(10, le=50),
):
    """
    Search both `horses.horse_name` and `horse_aliases.alias_name`.
    Returns matching horses with their current name and any alias hit.
    """
    rows = fetch_all(
        """
        SELECT h.horse_id, h.horse_name, NULL AS matched_alias
        FROM horses h
        WHERE LOWER(h.horse_name) LIKE LOWER(?)

        UNION ALL

        SELECT h.horse_id, h.horse_name, a.alias_name AS matched_alias
        FROM horse_aliases a
        JOIN horses h ON h.horse_id = a.horse_id
        WHERE LOWER(a.alias_name) LIKE LOWER(?)

        ORDER BY horse_name
        LIMIT ?
        """,
        [f"%{q}%", f"%{q}%", limit],
    )
    return {"query": q, "data": rows}


@router.get("/{horse_id}", summary="Get a horse by ID")
def get_horse(horse_id: int):
    row = fetch_one("SELECT * FROM horses WHERE horse_id = ?", [horse_id])
    if not row:
        raise HTTPException(status_code=404, detail="Horse not found")
    return row


@router.get("/{horse_id}/aliases", summary="Name change history for a horse")
def horse_aliases(horse_id: int):
    _check_horse(horse_id)
    rows = fetch_all(
        "SELECT * FROM horse_aliases WHERE horse_id = ? ORDER BY effective_from",
        [horse_id],
    )
    return {"horse_id": horse_id, "data": rows}


@router.get("/{horse_id}/ratings", summary="Rating change history for a horse")
def horse_ratings(horse_id: int):
    _check_horse(horse_id)
    rows = fetch_all(
        "SELECT * FROM ratings_changes WHERE horse_id = ? ORDER BY meet_date DESC",
        [horse_id],
    )
    return {"horse_id": horse_id, "data": rows}


@router.get("/{horse_id}/current-rating", summary="Most recent official rating")
def horse_current_rating(horse_id: int):
    _check_horse(horse_id)
    row = fetch_one(
        "SELECT * FROM v_horse_current_rating WHERE horse_id = ?",
        [horse_id],
    )
    return row or {"horse_id": horse_id, "current_rating": None}


@router.get("/{horse_id}/races", summary="Race history for a horse")
def horse_races(
    horse_id: int,
    limit: int = Query(settings.default_limit, le=settings.max_limit),
    offset: int = Query(0, ge=0),
):
    _check_horse(horse_id)
    sql = """
        SELECT *
        FROM v_runner_full
        WHERE horse_id = ?
        ORDER BY meet_date DESC
    """
    return paginate(sql, params=[horse_id], limit=limit, offset=offset)


@router.get("/{horse_id}/medical", summary="Medical history disclosed at acceptance")
def horse_medical(horse_id: int):
    _check_horse(horse_id)
    rows = fetch_all(
        "SELECT * FROM horse_medical WHERE horse_id = ? ORDER BY condition_date DESC",
        [horse_id],
    )
    return {"horse_id": horse_id, "data": rows}


@router.get("/{horse_id}/treadmill", summary="Treadmill rehab sessions")
def horse_treadmill(horse_id: int):
    _check_horse(horse_id)
    rows = fetch_all(
        "SELECT * FROM horse_treadmill WHERE horse_id = ? ORDER BY session_date DESC, segment",
        [horse_id],
    )
    return {"horse_id": horse_id, "data": rows}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_horse(horse_id: int):
    row = fetch_one("SELECT horse_id FROM horses WHERE horse_id = ?", [horse_id])
    if not row:
        raise HTTPException(status_code=404, detail="Horse not found")
