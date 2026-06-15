from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from api.db import fetch_all, fetch_one, paginate
from config import settings

router = APIRouter(prefix="/runners", tags=["Runners"])


@router.get("/{race_id}/{horse_id}", summary="Get full runner profile")
def get_runner(race_id: int, horse_id: int):
    row = fetch_one(
        "SELECT * FROM v_runner_full WHERE race_id = ? AND horse_id = ?",
        [race_id, horse_id],
    )
    if not row:
        raise HTTPException(status_code=404, detail="Runner not found")
    return row


@router.get("/{race_id}/{horse_id}/acceptance", summary="Acceptance data for a runner")
def runner_acceptance(race_id: int, horse_id: int):
    row = fetch_one(
        "SELECT * FROM runner_acceptances WHERE race_id = ? AND horse_id = ?",
        [race_id, horse_id],
    )
    if not row:
        raise HTTPException(status_code=404, detail="Acceptance record not found")
    return row


@router.get("/{race_id}/{horse_id}/declaration", summary="Declaration data for a runner")
def runner_declaration(race_id: int, horse_id: int):
    row = fetch_one(
        """
        SELECT rd.*, j.jockey_name, t.trainer_name
        FROM runner_declarations rd
        LEFT JOIN jockeys j ON j.jockey_id = rd.jockey_id
        LEFT JOIN trainers t ON t.trainer_id = rd.trainer_id
        WHERE rd.race_id = ? AND rd.horse_id = ?
        """,
        [race_id, horse_id],
    )
    if not row:
        raise HTTPException(status_code=404, detail="Declaration record not found")
    return row


@router.get("/{race_id}/{horse_id}/equipment", summary="Equipment at declaration")
def runner_equipment(race_id: int, horse_id: int):
    row = fetch_one(
        "SELECT * FROM runner_equipment WHERE race_id = ? AND horse_id = ?",
        [race_id, horse_id],
    )
    return row or {}


@router.get("/{race_id}/{horse_id}/equipment-changes", summary="Equipment changes vs previous race")
def runner_equipment_changes(race_id: int, horse_id: int):
    row = fetch_one(
        "SELECT * FROM equipment_changes WHERE race_id = ? AND horse_id = ?",
        [race_id, horse_id],
    )
    return row or {}


@router.get("/{race_id}/{horse_id}/jockey-change", summary="Jockey substitution for a runner")
def runner_jockey_change(race_id: int, horse_id: int):
    row = fetch_one(
        """
        SELECT jc.*,
               j1.jockey_name AS original_jockey_name,
               j2.jockey_name AS replacement_jockey_name
        FROM jockey_changes jc
        LEFT JOIN jockeys j1 ON j1.jockey_id = jc.original_jockey_id
        LEFT JOIN jockeys j2 ON j2.jockey_id = jc.replacement_jockey_id
        WHERE jc.race_id = ? AND jc.horse_id = ?
        """,
        [race_id, horse_id],
    )
    return row or {"race_id": race_id, "horse_id": horse_id, "change": None}
