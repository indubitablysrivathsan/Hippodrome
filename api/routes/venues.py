from fastapi import APIRouter, HTTPException, Query
from api.db import fetch_all, fetch_one, paginate
from config import settings

router = APIRouter(prefix="/venues", tags=["Venues"])


@router.get("/", summary="List all venues")
def list_venues(
    limit: int = Query(settings.default_limit, le=settings.max_limit),
    offset: int = Query(0, ge=0),
):
    """Return all venues with optional pagination."""
    return paginate("SELECT * FROM venues ORDER BY venue_name", limit=limit, offset=offset)


@router.get("/{venue_id}", summary="Get a venue by ID")
def get_venue(venue_id: int):
    row = fetch_one("SELECT * FROM venues WHERE venue_id = ?", [venue_id])
    if not row:
        raise HTTPException(status_code=404, detail="Venue not found")
    return row


@router.get("/{venue_id}/meetings", summary="List meetings at a venue")
def venue_meetings(
    venue_id: int,
    limit: int = Query(settings.default_limit, le=settings.max_limit),
    offset: int = Query(0, ge=0),
):
    sql = """
        SELECT m.*, v.venue_name
        FROM meetings m
        JOIN venues v ON v.venue_id = m.venue_id
        WHERE m.venue_id = ?
        ORDER BY m.meet_date DESC
    """
    return paginate(sql, params=[venue_id], limit=limit, offset=offset)
