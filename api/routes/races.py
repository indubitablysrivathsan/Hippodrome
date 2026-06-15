from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from api.db import fetch_all, fetch_one, paginate
from config import settings

router = APIRouter(prefix="/races", tags=["Races"])


@router.get("/", summary="List races")
def list_races(
    venue_id: Optional[int] = Query(None),
    date_from: Optional[str] = Query(None, description="YYYY-MM-DD"),
    date_to: Optional[str] = Query(None, description="YYYY-MM-DD"),
    distance_min: Optional[int] = Query(None, description="Minimum distance in metres"),
    distance_max: Optional[int] = Query(None, description="Maximum distance in metres"),
    limit: int = Query(settings.default_limit, le=settings.max_limit),
    offset: int = Query(0, ge=0),
):
    filters = []
    params: list = []

    if venue_id is not None:
        filters.append("r.venue_id = ?")
        params.append(venue_id)
    if date_from:
        filters.append("r.meet_date >= ?")
        params.append(date_from)
    if date_to:
        filters.append("r.meet_date <= ?")
        params.append(date_to)
    if distance_min is not None:
        filters.append("r.distance_meters >= ?")
        params.append(distance_min)
    if distance_max is not None:
        filters.append("r.distance_meters <= ?")
        params.append(distance_max)

    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    sql = f"""
        SELECT r.*, v.venue_name, m.track_condition, m.weather
        FROM races r
        JOIN venues v ON v.venue_id = r.venue_id
        JOIN meetings m ON m.meet_date = r.meet_date AND m.venue_id = r.venue_id
        {where}
        ORDER BY r.meet_date DESC, r.race_no
    """
    return paginate(sql, params=params, limit=limit, offset=offset)


@router.get("/{race_id}", summary="Get race by ID")
def get_race(race_id: int):
    row = fetch_one(
        """
        SELECT r.*, v.venue_name, m.track_condition, m.weather, m.penetrometer
        FROM races r
        JOIN venues v ON v.venue_id = r.venue_id
        JOIN meetings m ON m.meet_date = r.meet_date AND m.venue_id = r.venue_id
        WHERE r.race_id = ?
        """,
        [race_id],
    )
    if not row:
        raise HTTPException(status_code=404, detail="Race not found")
    return row


@router.get("/{race_id}/runners", summary="All runners in a race (full profile)")
def race_runners(race_id: int):
    # Verify the race exists first
    race = fetch_one("SELECT race_id FROM races WHERE race_id = ?", [race_id])
    if not race:
        raise HTTPException(status_code=404, detail="Race not found")

    rows = fetch_all(
        """
        SELECT *
        FROM v_runner_full
        WHERE race_id = ?
        ORDER BY COALESCE(finish_position, 999), draw
        """,
        [race_id],
    )
    return {"race_id": race_id, "data": rows, "total": len(rows)}


@router.get("/{race_id}/dividends", summary="Tote dividends for a race")
def race_dividends(race_id: int):
    rows = fetch_all(
        "SELECT * FROM race_dividends WHERE race_id = ? ORDER BY div_type",
        [race_id],
    )
    return {"race_id": race_id, "data": rows}


@router.get("/{race_id}/exotics", summary="Exotic pool results for a race")
def race_exotics(race_id: int):
    rows = fetch_all(
        "SELECT * FROM exotics WHERE race_id = ? ORDER BY pool_type",
        [race_id],
    )
    return {"race_id": race_id, "data": rows}


@router.get("/{race_id}/remarks", summary="Steward/starter remarks for a race")
def race_remarks(race_id: int):
    rows = fetch_all(
        """
        SELECT hr.*, h.horse_name
        FROM horse_remarks hr
        JOIN horses h ON h.horse_id = hr.horse_id
        WHERE hr.race_id = ?
        """,
        [race_id],
    )
    return {"race_id": race_id, "data": rows}


@router.get("/{race_id}/penalties", summary="Penalties issued at a race")
def race_penalties(race_id: int):
    rows = fetch_all(
        "SELECT * FROM penalties WHERE race_id = ? ORDER BY penalty_id",
        [race_id],
    )
    return {"race_id": race_id, "data": rows}
