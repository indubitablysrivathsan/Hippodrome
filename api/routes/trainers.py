from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from api.db import fetch_all, fetch_one, paginate
from config import settings

router = APIRouter(prefix="/trainers", tags=["Trainers"])


@router.get("/", summary="List / search trainers")
def list_trainers(
    name: Optional[str] = Query(None, description="Partial name search"),
    limit: int = Query(settings.default_limit, le=settings.max_limit),
    offset: int = Query(0, ge=0),
):
    filters = []
    params: list = []
    if name:
        filters.append("LOWER(trainer_name) LIKE LOWER(?)")
        params.append(f"%{name}%")
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    sql = f"SELECT * FROM trainers {where} ORDER BY trainer_name"
    return paginate(sql, params=params, limit=limit, offset=offset)


@router.get("/{trainer_id}", summary="Get trainer by ID")
def get_trainer(trainer_id: int):
    row = fetch_one("SELECT * FROM trainers WHERE trainer_id = ?", [trainer_id])
    if not row:
        raise HTTPException(status_code=404, detail="Trainer not found")
    return row


@router.get("/{trainer_id}/runners", summary="Race history for a trainer")
def trainer_runners(
    trainer_id: int,
    limit: int = Query(settings.default_limit, le=settings.max_limit),
    offset: int = Query(0, ge=0),
):
    _check_trainer(trainer_id)
    sql = """
        SELECT *
        FROM v_runner_full
        WHERE trainer_id = ?
        ORDER BY meet_date DESC
    """
    return paginate(sql, params=[trainer_id], limit=limit, offset=offset)


@router.get("/{trainer_id}/stats", summary="Win/place stats for a trainer")
def trainer_stats(trainer_id: int):
    _check_trainer(trainer_id)
    row = fetch_one(
        """
        SELECT
            vrf.trainer_id,
            vrf.trainer_name,
            COUNT(*)                                                          AS total_runners,
            COUNT(*) FILTER (WHERE vrf.finish_position = 1)                  AS wins,
            COUNT(*) FILTER (WHERE vrf.finish_position <= 3)                 AS places,
            ROUND(
                COUNT(*) FILTER (WHERE vrf.finish_position = 1) * 100.0
                / NULLIF(COUNT(*), 0), 2
            )                                                                 AS win_pct
        FROM v_runner_full vrf
        WHERE vrf.trainer_id = ?
        GROUP BY vrf.trainer_id, vrf.trainer_name
        """,
        [trainer_id],
    )
    return row or {"trainer_id": trainer_id, "total_runners": 0}


@router.get("/{trainer_id}/penalties", summary="Penalties issued to a trainer")
def trainer_penalties(trainer_id: int):
    _check_trainer(trainer_id)
    rows = fetch_all(
        "SELECT * FROM penalties WHERE trainer_id = ? ORDER BY race_id DESC",
        [trainer_id],
    )
    return {"trainer_id": trainer_id, "data": rows}


def _check_trainer(trainer_id: int):
    if not fetch_one("SELECT trainer_id FROM trainers WHERE trainer_id = ?", [trainer_id]):
        raise HTTPException(status_code=404, detail="Trainer not found")
