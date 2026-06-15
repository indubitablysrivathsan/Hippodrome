from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from api.db import fetch_all, fetch_one, paginate
from config import settings

router = APIRouter(prefix="/meetings", tags=["Meetings"])


@router.get("/", summary="List meetings")
def list_meetings(
    venue_id: Optional[int] = Query(None, description="Filter by venue"),
    season: Optional[str] = Query(None, description="Filter by season label"),
    track_condition: Optional[str] = Query(None, description="e.g. Good, Soft, Heavy"),
    date_from: Optional[str] = Query(None, description="YYYY-MM-DD"),
    date_to: Optional[str] = Query(None, description="YYYY-MM-DD"),
    limit: int = Query(settings.default_limit, le=settings.max_limit),
    offset: int = Query(0, ge=0),
):
    filters = []
    params: list = []

    if venue_id is not None:
        filters.append("m.venue_id = ?")
        params.append(venue_id)
    if season:
        filters.append("m.season = ?")
        params.append(season)
    if track_condition:
        filters.append("LOWER(m.track_condition) LIKE LOWER(?)")
        params.append(f"%{track_condition}%")
    if date_from:
        filters.append("m.meet_date >= ?")
        params.append(date_from)
    if date_to:
        filters.append("m.meet_date <= ?")
        params.append(date_to)

    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    sql = f"""
        SELECT m.*, v.venue_name
        FROM meetings m
        JOIN venues v ON v.venue_id = m.venue_id
        {where}
        ORDER BY m.meet_date DESC
    """
    return paginate(sql, params=params, limit=limit, offset=offset)


@router.get("/{meet_date}/{venue_id}", summary="Get a specific meeting")
def get_meeting(meet_date: str, venue_id: int):
    row = fetch_one(
        """
        SELECT m.*, v.venue_name
        FROM meetings m
        JOIN venues v ON v.venue_id = m.venue_id
        WHERE m.meet_date = ? AND m.venue_id = ?
        """,
        [meet_date, venue_id],
    )
    if not row:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return row


@router.get("/{meet_date}/{venue_id}/races", summary="All races for a meeting")
def meeting_races(meet_date: str, venue_id: int):
    rows = fetch_all(
        """
        SELECT r.*, v.venue_name
        FROM races r
        JOIN venues v ON v.venue_id = r.venue_id
        WHERE r.meet_date = ? AND r.venue_id = ?
        ORDER BY r.race_no
        """,
        [meet_date, venue_id],
    )
    return {"data": rows, "total": len(rows)}
