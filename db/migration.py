"""
RWITC Horse Racing — ETL Pipeline
==================================
Loads all CSV/XLSX source files into DuckDB using the rwitc_schema.sql schema.

Usage:
    python etl.py --db rwitc.db --schema rwitc_schema.sql

Add your file paths in the DATASETS block at the bottom of this file.
Every loader follows the same contract:
    loader(df, con, log) -> None
    where df is the merged/repaired DataFrame, con is duckdb connection, log is ImportLog
"""

import re
import sys
import argparse
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import duckdb


# ─────────────────────────────────────────────────────────────
# IMPORT LOG
# ─────────────────────────────────────────────────────────────

@dataclass
class LogEntry:
    dataset:    str
    row:        Optional[int]
    column:     Optional[str]
    xlsx_value: object
    csv_value:  object
    action:     str          # 'repaired', 'warning', 'error'
    message:    str = ''

class ImportLog:
    def __init__(self):
        self.entries: list[LogEntry] = []

    def repaired(self, dataset, row, column, xlsx_val, csv_val):
        self.entries.append(LogEntry(dataset, row, column, xlsx_val, csv_val, 'repaired'))

    def warning(self, dataset, row, column, xlsx_val, csv_val, msg=''):
        self.entries.append(LogEntry(dataset, row, column, xlsx_val, csv_val, 'warning', msg))

    def error(self, dataset, row, column, xlsx_val, csv_val, msg=''):
        self.entries.append(LogEntry(dataset, row, column, xlsx_val, csv_val, 'error', msg))

    def summary(self, dataset: str):
        subset = [e for e in self.entries if e.dataset == dataset]
        repairs  = sum(1 for e in subset if e.action == 'repaired')
        warnings = sum(1 for e in subset if e.action == 'warning')
        errors   = sum(1 for e in subset if e.action == 'error')
        print(f"\n{'─'*50}")
        print(f"  Dataset : {dataset}")
        print(f"  Repairs : {repairs}")
        print(f"  Warnings: {warnings}")
        print(f"  Errors  : {errors}")
        print(f"{'─'*50}")

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([vars(e) for e in self.entries])

    def save(self, path: str):
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        print(f"\nImport log saved to: {path}")


# ─────────────────────────────────────────────────────────────
# GENERIC SOURCE READER
# ─────────────────────────────────────────────────────────────

# Columns known to be mangled by Excel date auto-conversion
# Maps column name -> repair function
FALLBACK_COLUMNS = {
    'odds':        'odds',
    'finish_time': 'finish_time',
}

def _is_mangled_odds(value) -> bool:
    """
    Detect Excel-mangled odds.
    '20-Jan' -> True  (was 20/1)
    '40/1'   -> False (survived correctly)
    '11-Apr' -> True  (was 11/4)
    """
    if pd.isna(value):
        return False
    s = str(value).strip()
    # Excel converts fractions like 20/1 to dates like "20-Jan"
    # Pattern: digit(s) + '-' + month abbreviation
    return bool(re.match(r'^\d{1,2}-[A-Za-z]{3}$', s))

def _is_mangled_time(value) -> bool:
    """
    Detect Excel-mangled finish times.
    0.045497685 -> True  (was 1:05:012, stored as Excel time fraction)
    '1:11:090'  -> False (survived correctly)
    '-'         -> False (legitimate DNF marker)
    """
    if pd.isna(value):
        return False
    s = str(value).strip()
    if s == '-':
        return False
    try:
        f = float(s)
        # Excel time fractions are between 0 and 1 (or just over 1 for > 24h)
        return 0 < f < 1
    except (ValueError, TypeError):
        return False

MANGLE_DETECTORS = {
    'odds':        _is_mangled_odds,
    'finish_time': _is_mangled_time,
}

def read_source(
    dataset_name: str,
    xlsx_path: Optional[str],
    csv_path: Optional[str],
    fallback_columns: Optional[list[str]],
    log: ImportLog,
    merge_key: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Load a dataset from xlsx and/or csv, repairing known-bad columns
    using csv as fallback where xlsx values are mangled.

    merge_key: list of column names to use for row-matching between xlsx and csv
               e.g. ['meet_date', 'venue', 'race_no', 'horse_name']
               If None, falls back to positional row index (less robust).
    """

    if xlsx_path is None and csv_path is None:
        raise ValueError(f"[{dataset_name}] No source files provided.")

    # ── Case 1: CSV only ──────────────────────────────────────
    if xlsx_path is None:
        print(f"[{dataset_name}] Reading CSV only: {csv_path}")
        return pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    # ── Case 2: XLSX only ─────────────────────────────────────
    if csv_path is None:
        print(f"[{dataset_name}] Reading XLSX only (no CSV fallback): {xlsx_path}")
        if fallback_columns:
            log.warning(dataset_name, None, str(fallback_columns), None, None,
                        "Fallback columns requested but no CSV available")
        return pd.read_excel(xlsx_path, dtype=str)

    # ── Case 3: Both exist — xlsx authoritative, csv fallback ─
    print(f"[{dataset_name}] Reading XLSX (authoritative) + CSV (fallback)...")
    xlsx_df = pd.read_excel(xlsx_path, dtype=str)
    csv_df  = pd.read_csv(csv_path,   dtype=str, keep_default_na=False)

    if not fallback_columns:
        return xlsx_df  # no repair needed

    # Normalise column names for matching
    xlsx_df.columns = [c.strip().lower() for c in xlsx_df.columns]
    csv_df.columns  = [c.strip().lower() for c in csv_df.columns]

    # Build row-matching index
    if merge_key:
        key_cols = [k.lower() for k in merge_key]
        missing_xlsx = [k for k in key_cols if k not in xlsx_df.columns]
        missing_csv  = [k for k in key_cols if k not in csv_df.columns]
        if missing_xlsx or missing_csv:
            log.warning(dataset_name, None, str(key_cols), None, None,
                        f"Merge key columns missing — falling back to positional. "
                        f"XLSX missing: {missing_xlsx}, CSV missing: {missing_csv}")
            merge_key = None

    if merge_key:
        # Key-based merge: robust against sorting differences
        key_cols = [k.lower() for k in merge_key]
        csv_indexed = csv_df.set_index(key_cols)
    else:
        csv_indexed = None

    for col in fallback_columns:
        col_lower = col.lower()
        if col_lower not in xlsx_df.columns:
            log.warning(dataset_name, None, col, None, None,
                        f"Fallback column '{col}' not found in XLSX — skipping")
            continue
        if col_lower not in (csv_df.columns if csv_indexed is None else csv_indexed.columns):
            log.warning(dataset_name, None, col, None, None,
                        f"Fallback column '{col}' not found in CSV — cannot repair")
            continue

        detector = MANGLE_DETECTORS.get(col_lower)
        if detector is None:
            # No detector registered; skip
            continue

        for idx, xlsx_val in xlsx_df[col_lower].items():
            if not detector(xlsx_val):
                continue

            # Value is mangled — attempt CSV repair
            csv_val = None
            if merge_key and csv_indexed is not None:
                key_cols = [k.lower() for k in merge_key]
                try:
                    key_tuple = tuple(xlsx_df.loc[idx, k] for k in key_cols)
                    csv_val = csv_indexed.loc[key_tuple, col_lower]
                    if isinstance(csv_val, pd.Series):
                        csv_val = csv_val.iloc[0]
                except KeyError:
                    csv_val = None
            else:
                # Positional fallback
                if idx < len(csv_df):
                    csv_val = csv_df.iloc[idx][col_lower]

            if csv_val is not None and str(csv_val).strip() not in ('', 'nan'):
                xlsx_df.at[idx, col_lower] = csv_val
                log.repaired(dataset_name, idx, col_lower, xlsx_val, csv_val)
            else:
                log.error(dataset_name, idx, col_lower, xlsx_val, csv_val,
                          "CSV value unavailable or empty — could not repair")

    return xlsx_df


# ─────────────────────────────────────────────────────────────
# NAME NORMALISATION HELPERS
# ─────────────────────────────────────────────────────────────

def norm_name(s: str) -> str:
    """Uppercase, strip extra whitespace, normalise apostrophes."""
    if not s or pd.isna(s):
        return ''
    return re.sub(r'\s+', ' ', str(s).strip().upper()) \
             .replace('\u2019', "'").replace('\u2018', "'")

def norm_person(s: str) -> str:
    """Title-case, strip, normalise for jockeys/trainers."""
    if not s or pd.isna(s):
        return ''
    return re.sub(r'\s+', ' ', str(s).strip())

def parse_jockey_claim(raw: str) -> tuple[str, Optional[float]]:
    """
    'A. S. Peter - 3.5' -> ('A. S. Peter', 3.5)
    'C. S. Jodha'       -> ('C. S. Jodha', None)
    """
    if not raw or pd.isna(raw):
        return ('', None)
    m = re.match(r'^(.+?)\s*-\s*(\d+\.?\d*)$', str(raw).strip())
    if m:
        return (norm_person(m.group(1)), float(m.group(2)))
    return (norm_person(raw), None)

def parse_odds(raw: str) -> tuple[Optional[int], Optional[int]]:
    """
    '40/1'  -> (40, 1)
    '7/2'   -> (7, 2)
    'Evens' -> (1, 1)
    '-'     -> (None, None)
    """
    if not raw or pd.isna(raw):
        return (None, None)
    s = str(raw).strip()
    if s in ('-', '', 'nan'):
        return (None, None)
    if s.lower() in ('evens', 'evs'):
        return (1, 1)
    m = re.match(r'^(\d+)\s*/\s*(\d+)$', s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return (None, None)

def parse_finish_time(raw: str) -> Optional[int]:
    """
    '1:11:090' -> 71090  (milliseconds)
    '-'        -> None
    """
    if not raw or pd.isna(raw):
        return None
    s = str(raw).strip()
    if s in ('-', '', 'nan'):
        return None
    m = re.match(r'^(\d+):(\d+):(\d+)$', s)
    if m:
        mins, secs, ms = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return (mins * 60 + secs) * 1000 + ms
    return None

def safe_int(v) -> Optional[int]:
    try:
        return int(v)
    except (ValueError, TypeError):
        return None

def safe_float(v) -> Optional[float]:
    try:
        return float(v)
    except (ValueError, TypeError):
        return None

def safe_date(v) -> Optional[str]:
    """Return ISO date string or None."""
    if not v or pd.isna(v):
        return None
    try:
        return pd.to_datetime(v).date().isoformat()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# ENTITY REGISTRY (in-memory lookup + upsert helpers)
# Avoids round-tripping the DB for every row
# ─────────────────────────────────────────────────────────────

class EntityRegistry:
    """
    Caches horse/jockey/trainer/venue IDs during ETL.
    Also handles horse alias resolution for name-change linking.
    """

    def __init__(self, con: duckdb.DuckDBPyConnection):
        self.con = con
        self._venues:   dict[str, int] = {}
        self._jockeys:  dict[str, int] = {}
        self._trainers: dict[str, int] = {}
        self._horses:   dict[str, int] = {}   # current_name -> horse_id
        self._aliases:  dict[str, int] = {}   # alias_name   -> horse_id

        self._load_existing()

    def _load_existing(self):
        for row in self.con.execute("SELECT venue_name, venue_id FROM venues").fetchall():
            self._venues[row[0].upper()] = row[1]
        for row in self.con.execute("SELECT jockey_name, jockey_id FROM jockeys").fetchall():
            self._jockeys[row[0]] = row[1]
        for row in self.con.execute("SELECT trainer_name, trainer_id FROM trainers").fetchall():
            self._trainers[row[0]] = row[1]
        for row in self.con.execute("SELECT horse_name, horse_id FROM horses").fetchall():
            self._horses[row[0]] = row[1]
        for row in self.con.execute("SELECT alias_name, horse_id FROM horse_aliases").fetchall():
            self._aliases[row[0]] = row[1]

    def venue(self, name: str) -> int:
        key = name.strip().upper()
        if key not in self._venues:
            self.con.execute(
                "INSERT OR IGNORE INTO venues(venue_name) VALUES (?)", [key])
            row = self.con.execute(
                "SELECT venue_id FROM venues WHERE venue_name=?", [key]).fetchone()
            self._venues[key] = row[0]
        return self._venues[key]

    def jockey(self, name: str) -> Optional[int]:
        if not name:
            return None
        key = norm_person(name)
        if key not in self._jockeys:
            self.con.execute(
                "INSERT OR IGNORE INTO jockeys(jockey_name) VALUES (?)", [key])
            row = self.con.execute(
                "SELECT jockey_id FROM jockeys WHERE jockey_name=?", [key]).fetchone()
            self._jockeys[key] = row[0]
        return self._jockeys[key]

    def trainer(self, name: str) -> Optional[int]:
        if not name:
            return None
        key = norm_person(name)
        if key not in self._trainers:
            self.con.execute(
                "INSERT OR IGNORE INTO trainers(trainer_name) VALUES (?)", [key])
            row = self.con.execute(
                "SELECT trainer_id FROM trainers WHERE trainer_name=?", [key]).fetchone()
            self._trainers[key] = row[0]
        return self._trainers[key]

    def horse(
        self,
        name: str,
        late_name: Optional[str] = None,
        horse_seq: Optional[int] = None,
        sire: Optional[str] = None,
        sire_nat: Optional[str] = None,
        dam: Optional[str] = None,
        dam_nat: Optional[str] = None,
    ) -> int:
        """
        Resolve or create a horse record.

        Resolution order:
          1. Look up current name in horses table
          2. Look up current name in horse_aliases (old name)
          3. If late_name provided, look up late_name in horses/aliases
          4. If none found, insert new horse

        When late_name is found for a horse:
          - The horse's canonical name is updated to current `name`
          - The old name is recorded in horse_aliases
        """
        key = norm_name(name)
        if not key:
            raise ValueError(f"Empty horse name passed to registry")

        # ── 1. Direct match on current name ──────────────────
        if key in self._horses:
            horse_id = self._horses[key]
            # Register late_name alias if provided and not yet known
            if late_name:
                self._register_alias(horse_id, late_name)
            return horse_id

        # ── 2. Current name is an alias (horse was renamed) ──
        if key in self._aliases:
            horse_id = self._aliases[key]
            # This name WAS an alias; update canonical name
            self.con.execute(
                "UPDATE horses SET horse_name=? WHERE horse_id=?", [key, horse_id])
            # Old canonical becomes alias
            old_name = self.con.execute(
                "SELECT horse_name FROM horses WHERE horse_id=?", [horse_id]).fetchone()
            if old_name:
                self._register_alias(horse_id, old_name[0])
            self._horses[key] = horse_id
            if late_name:
                self._register_alias(horse_id, late_name)
            return horse_id

        # ── 3. Check if late_name matches an existing horse ──
        if late_name:
            late_key = norm_name(late_name)
            late_id  = self._horses.get(late_key) or self._aliases.get(late_key)
            if late_id:
                # Existing horse was renamed to `name`
                # Update canonical name and alias old one
                self.con.execute(
                    "UPDATE horses SET horse_name=? WHERE horse_id=?", [key, late_id])
                self._register_alias(late_id, late_key)
                # Remove old canonical from primary dict, add new
                self._horses.pop(late_key, None)
                self._horses[key] = late_id
                return late_id

        # ── 4. New horse ──────────────────────────────────────
        self.con.execute(
            """INSERT INTO horses(horse_name, horse_seq, sire, sire_nat, dam, dam_nat)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [key, horse_seq, sire, sire_nat, dam, dam_nat])
        row = self.con.execute(
            "SELECT horse_id FROM horses WHERE horse_name=?", [key]).fetchone()
        horse_id = row[0]
        self._horses[key] = horse_id
        if late_name:
            self._register_alias(horse_id, late_name)
        return horse_id

    def _register_alias(self, horse_id: int, alias: str, source: str = 'late_name'):
        key = norm_name(alias)
        if not key or key in self._aliases:
            return
        self.con.execute(
            """INSERT OR IGNORE INTO horse_aliases(horse_id, alias_name, source)
               VALUES (?, ?, ?)""",
            [horse_id, key, source])
        self._aliases[key] = horse_id

    def race_id(self, meet_date: str, venue_id: int, race_no: int) -> Optional[int]:
        row = self.con.execute(
            """SELECT race_id FROM races
               WHERE meet_date=? AND venue_id=? AND race_no=?""",
            [meet_date, venue_id, race_no]).fetchone()
        return row[0] if row else None


# ─────────────────────────────────────────────────────────────
# EQUIPMENT CHANGE PARSER
# ─────────────────────────────────────────────────────────────

# Maps abbreviation -> (column_name, on=True/False)
EQUIP_MAP = {
    'BLK':   'blinkers',
    'VISOR': 'visor',
    'TS':    'tongue_strap',
    'EP':    'earplugs',
    'CNB':   'cheek_pieces',
    'HOOD':  'hood',
    'PACI':  'pacifier',
    'RES':   'earplugs',         # alternate abbreviation for earplugs
}

def parse_equipment_changes(raw: str) -> dict:
    """
    'BLK OFF,VISOR ON,EP OFF' ->
    {
        'equip_change_raw': 'BLK OFF,VISOR ON,EP OFF',
        'blinkers_on': False, 'blinkers_off': True,
        'visor_on': True, 'visor_off': False,
        'earplugs_on': False, 'earplugs_off': True,
        ...
    }
    """
    result = {
        'equip_change_raw': raw,
        'blinkers_on': None,     'blinkers_off': None,
        'visor_on': None,        'visor_off': None,
        'tongue_strap_on': None, 'tongue_strap_off': None,
        'earplugs_on': None,     'earplugs_off': None,
        'cheek_pieces_on': None, 'cheek_pieces_off': None,
        'hood_on': None,         'hood_off': None,
        'pacifier_on': None,     'pacifier_off': None,
    }
    if not raw or pd.isna(raw):
        return result
    for token in str(raw).upper().split(','):
        token = token.strip()
        m = re.match(r'^(\w+)\s+(ON|OFF)$', token)
        if not m:
            continue
        abbr, direction = m.group(1), m.group(2)
        col_base = EQUIP_MAP.get(abbr)
        if col_base:
            result[f"{col_base}_on"]  = (direction == 'ON')
            result[f"{col_base}_off"] = (direction == 'OFF')
    return result


# ─────────────────────────────────────────────────────────────
# INDIVIDUAL LOADERS
# Each receives a clean DataFrame and writes to DuckDB.
# ─────────────────────────────────────────────────────────────

def load_results_meetings(df: pd.DataFrame, con: duckdb.DuckDBPyConnection,
                           log: ImportLog, reg: EntityRegistry):
    df.columns = [c.strip().lower() for c in df.columns]
    for _, row in df.iterrows():
        try:
            venue_id = reg.venue(row.get('venue', ''))
            meet_date = safe_date(row.get('meet_date'))
            con.execute("""
                INSERT OR IGNORE INTO meetings
                    (meet_date, venue_id, season, meeting_day_desc,
                     weather, track_condition, penetrometer, false_rails)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                meet_date, venue_id,
                row.get('season') or None,
                row.get('meeting_day_desc') or None,
                row.get('weather') or None,
                row.get('track_condition') or None,
                safe_float(row.get('penetrometer')),
                row.get('false_rails') or None,
            ])
        except Exception as e:
            log.error('results_meetings', None, None, None, None, str(e))


def load_results_races(df: pd.DataFrame, con: duckdb.DuckDBPyConnection,
                        log: ImportLog, reg: EntityRegistry):
    df.columns = [c.strip().lower() for c in df.columns]

    DIV_COLS = {
        'win_div': 'WIN', 'place_div': 'PLACE',
        'shp_div': 'SHP', 'for_div':   'FOR',
        'qnl_div': 'QNL', 'tnl_div':   'TNL',
    }

    for _, row in df.iterrows():
        try:
            venue_id  = reg.venue(row.get('venue', ''))
            meet_date = safe_date(row.get('meet_date'))
            race_no   = safe_int(row.get('race_no'))

            con.execute("""
                INSERT OR IGNORE INTO races
                    (meet_date, venue_id, race_no, card_seq, race_name,
                     class_conditions, scheduled_time, distance_meters,
                     margins, tote_favourite)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                meet_date, venue_id, race_no,
                safe_int(row.get('card_seq')),
                row.get('race_name') or None,
                row.get('class_conditions') or None,
                row.get('scheduled_time') or None,
                safe_int(row.get('distance_meters')),
                row.get('margins') or None,
                row.get('tote_favourite') or None,
            ])

            race_id = reg.race_id(meet_date, venue_id, race_no)
            if race_id is None:
                continue

            # Insert dividends
            for col, div_type in DIV_COLS.items():
                val = safe_float(row.get(col))
                if val is not None:
                    con.execute("""
                        INSERT OR IGNORE INTO race_dividends(race_id, div_type, dividend)
                        VALUES (?, ?, ?)
                    """, [race_id, div_type, val])

        except Exception as e:
            log.error('results_races', None, None, None, None, str(e))


def load_results_runners(df: pd.DataFrame, con: duckdb.DuckDBPyConnection,
                          log: ImportLog, reg: EntityRegistry):
    df.columns = [c.strip().lower() for c in df.columns]
    for _, row in df.iterrows():
        try:
            venue_id  = reg.venue(row.get('venue', ''))
            meet_date = safe_date(row.get('meet_date'))
            race_no   = safe_int(row.get('race_no'))
            race_id   = reg.race_id(meet_date, venue_id, race_no)
            if race_id is None:
                log.warning('results_runners', None, 'race_id', None, None,
                            f"Race not found: {meet_date} {row.get('venue')} R{race_no}")
                continue

            horse_id = reg.horse(
                name=row.get('horse_name', ''),
                horse_seq=safe_int(row.get('horse_seq')),
                sire=norm_name(row.get('sire', '')),
                sire_nat=row.get('sire_nat') or None,
                dam=norm_name(row.get('dam', '')),
                dam_nat=row.get('dam_nat') or None,
            )

            # Placing
            placing_raw = str(row.get('placing', '')).strip()
            placing = safe_int(placing_raw)

            # Finish time
            ft_raw = str(row.get('finish_time', '')).strip()
            ft_ms  = parse_finish_time(ft_raw)

            # Odds
            odds_raw = str(row.get('odds', '')).strip()
            odds_n, odds_d = parse_odds(odds_raw)

            # Body weight
            bwt_raw = str(row.get('horse_body_wt', '')).strip()
            bwt_nr  = (bwt_raw.upper() == 'NR')
            bwt     = None if bwt_nr else safe_int(bwt_raw)

            con.execute("""
                INSERT OR IGNORE INTO runners
                    (race_id, horse_id, placing, placing_raw,
                     finish_time_ms, finish_time_raw,
                     odds_numerator, odds_denominator, odds_raw,
                     horse_body_wt, horse_body_wt_nr,
                     weight)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                race_id, horse_id, placing, placing_raw,
                ft_ms, ft_raw,
                odds_n, odds_d, odds_raw,
                bwt, bwt_nr,
                safe_float(row.get('weight')),
            ])
        except Exception as e:
            log.error('results_runners', None, None, None, None, str(e))


def load_results_exotics(df: pd.DataFrame, con: duckdb.DuckDBPyConnection,
                          log: ImportLog, reg: EntityRegistry):
    df.columns = [c.strip().lower() for c in df.columns]
    for _, row in df.iterrows():
        try:
            venue_id  = reg.venue(row.get('venue', ''))
            meet_date = safe_date(row.get('meet_date'))
            legs      = str(row.get('legs', '')).strip()
            pool_type = str(row.get('pool_type', '')).strip().upper()

            # race_id = last leg race
            last_leg = safe_int(legs.split(',')[-1]) if legs else None
            race_id  = reg.race_id(meet_date, venue_id, last_leg) if last_leg else None
            if race_id is None:
                log.warning('results_exotics', None, 'race_id', None, None,
                            f"Last leg race not found: {meet_date} R{last_leg}")
                continue

            con.execute("""
                INSERT OR IGNORE INTO exotics
                    (race_id, pool_type, legs, winners,
                     div_70pct, tickets_70pct, div_30pct, tickets_30pct,
                     dividend, tickets, carried_forward)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                race_id, pool_type, legs,
                row.get('winners') or None,
                safe_float(row.get('div_70pct')),
                safe_int(row.get('tickets_70pct')),
                safe_float(row.get('div_30pct')),
                safe_int(row.get('tickets_30pct')),
                safe_float(row.get('dividend')),
                safe_int(row.get('tickets')),
                safe_float(row.get('carried_forward')),
            ])
        except Exception as e:
            log.error('results_exotics', None, None, None, None, str(e))


def load_acceptances(df: pd.DataFrame, con: duckdb.DuckDBPyConnection,
                      log: ImportLog, reg: EntityRegistry):
    df.columns = [c.strip().lower() for c in df.columns]
    for _, row in df.iterrows():
        try:
            venue_id  = reg.venue(row.get('venue', ''))
            meet_date = safe_date(row.get('meet_date'))
            race_no   = safe_int(row.get('race_no'))
            race_id   = reg.race_id(meet_date, venue_id, race_no)
            if race_id is None:
                log.warning('acceptances', None, 'race_id', None, None,
                            f"Race not found: {meet_date} R{race_no}")
                continue

            late_name = norm_name(row.get('late_name', '')) or None
            horse_id  = reg.horse(
                name=row.get('horse_name', ''),
                late_name=late_name,
            )

            con.execute("""
                INSERT OR IGNORE INTO runner_acceptances
                    (race_id, horse_id, rating, weight_at_acceptance,
                     weight_update_handicap, weight_update_acceptance,
                     late_name, foreign_jockeys_allowed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                race_id, horse_id,
                safe_int(row.get('rating')),
                safe_float(row.get('weight')),
                safe_float(row.get('weight_update_handicap')),
                safe_float(row.get('weight_update_acceptance')),
                late_name,
                row.get('foreign_jockeys', '').strip().upper() == 'YES',
            ])
        except Exception as e:
            log.error('acceptances', None, None, None, None, str(e))


def load_highest_ratings(df: pd.DataFrame, con: duckdb.DuckDBPyConnection,
                          log: ImportLog, reg: EntityRegistry):
    df.columns = [c.strip().lower() for c in df.columns]
    for _, row in df.iterrows():
        try:
            meet_date = safe_date(row.get('meet_date'))
            horse_id  = reg.horse(name=row.get('horse_name', ''))
            # Upsert into runner_acceptances peak_rating columns
            con.execute("""
                UPDATE runner_acceptances
                SET peak_rating=?, peak_rating_date=?
                WHERE horse_id=?
                  AND race_id IN (
                      SELECT race_id FROM races WHERE meet_date=?
                  )
            """, [
                safe_int(row.get('highest_rating')),
                safe_date(row.get('achieved_date')),
                horse_id, meet_date,
            ])
        except Exception as e:
            log.error('acceptances_highest_ratings', None, None, None, None, str(e))


def load_medical(df: pd.DataFrame, con: duckdb.DuckDBPyConnection,
                  log: ImportLog, reg: EntityRegistry):
    df.columns = [c.strip().lower() for c in df.columns]
    for _, row in df.iterrows():
        try:
            meet_date = safe_date(row.get('meet_date'))
            race_no   = safe_int(row.get('race_no'))
            venue_id  = reg.venue(row.get('venue', 'UNKNOWN'))
            horse_id  = reg.horse(name=row.get('horse_name', ''))
            race_id   = reg.race_id(meet_date, venue_id, race_no)
            con.execute("""
                INSERT OR IGNORE INTO horse_medical
                    (horse_id, condition, condition_date, disclosed_race_id)
                VALUES (?, ?, ?, ?)
            """, [
                horse_id,
                row.get('condition') or None,
                safe_date(row.get('date')),
                race_id,
            ])
        except Exception as e:
            log.error('acceptances_medical', None, None, None, None, str(e))


def load_treadmill(df: pd.DataFrame, con: duckdb.DuckDBPyConnection,
                    log: ImportLog, reg: EntityRegistry):
    df.columns = [c.strip().lower() for c in df.columns]
    for _, row in df.iterrows():
        try:
            meet_date = safe_date(row.get('meet_date'))
            race_no   = safe_int(row.get('race_no'))
            venue_id  = reg.venue(row.get('venue', 'UNKNOWN'))
            horse_id  = reg.horse(name=row.get('horse_name', ''))
            race_id   = reg.race_id(meet_date, venue_id, race_no)
            con.execute("""
                INSERT OR IGNORE INTO horse_treadmill
                    (horse_id, session_date, segment,
                     speed_kmh, duration_min, disclosed_race_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [
                horse_id,
                safe_date(row.get('date')),
                safe_int(row.get('segment')),
                safe_float(row.get('km/h')),
                safe_int(row.get('minutes')),
                race_id,
            ])
        except Exception as e:
            log.error('acceptances_treadmill', None, None, None, None, str(e))


def load_bandages(df: pd.DataFrame, con: duckdb.DuckDBPyConnection,
                   log: ImportLog, reg: EntityRegistry):
    """Merges into runner_equipment."""
    df.columns = [c.strip().lower() for c in df.columns]
    for _, row in df.iterrows():
        try:
            meet_date = safe_date(row.get('meet_date'))
            race_no   = safe_int(row.get('race_no'))
            # Bandages file doesn't have venue; query races to find it
            r = con.execute(
                "SELECT race_id, venue_id FROM races WHERE meet_date=? AND race_no=?",
                [meet_date, race_no]).fetchone()
            if not r:
                log.warning('acceptances_bandages', None, 'race_id', None, None,
                            f"Race not found: {meet_date} R{race_no}")
                continue
            race_id = r[0]
            horse_id = reg.horse(name=row.get('horse_name', ''))
            bandage  = str(row.get('bandage_type', '')).strip() or None

            con.execute("""
                INSERT INTO runner_equipment(race_id, horse_id, bandage_type)
                VALUES (?, ?, ?)
                ON CONFLICT(race_id, horse_id)
                DO UPDATE SET bandage_type=excluded.bandage_type
            """, [race_id, horse_id, bandage])
        except Exception as e:
            log.error('acceptances_bandages', None, None, None, None, str(e))


def load_equipment(df: pd.DataFrame, con: duckdb.DuckDBPyConnection,
                    log: ImportLog, reg: EntityRegistry):
    df.columns = [c.strip().lower() for c in df.columns]
    for _, row in df.iterrows():
        try:
            meet_date = safe_date(row.get('meet_date'))
            race_no   = safe_int(row.get('race_no'))
            r = con.execute(
                "SELECT race_id FROM races WHERE meet_date=? AND race_no=?",
                [meet_date, race_no]).fetchone()
            if not r:
                continue
            race_id  = r[0]
            horse_id = reg.horse(name=row.get('horse_name', ''))
            con.execute("""
                INSERT INTO runner_equipment(race_id, horse_id, shoe_type, bit, hood_other)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(race_id, horse_id) DO UPDATE SET
                    shoe_type=COALESCE(excluded.shoe_type, runner_equipment.shoe_type),
                    bit=COALESCE(excluded.bit, runner_equipment.bit),
                    hood_other=COALESCE(excluded.hood_other, runner_equipment.hood_other)
            """, [
                race_id, horse_id,
                row.get('shoe_type') or None,
                row.get('bit') or None,
                row.get('hood_other') or None,
            ])
        except Exception as e:
            log.error('acceptances_equipment', None, None, None, None, str(e))


def load_declarations(df: pd.DataFrame, con: duckdb.DuckDBPyConnection,
                       log: ImportLog, reg: EntityRegistry):
    df.columns = [c.strip().lower() for c in df.columns]
    for _, row in df.iterrows():
        try:
            venue_id  = reg.venue(row.get('venue', ''))
            meet_date = safe_date(row.get('meet_date'))
            race_no   = safe_int(row.get('race_no'))
            race_id   = reg.race_id(meet_date, venue_id, race_no)
            if race_id is None:
                log.warning('declarations', None, 'race_id', None, None,
                            f"Race not found: {meet_date} R{race_no}")
                continue

            horse_id = reg.horse(name=row.get('horse_name', ''))
            jockey_raw = str(row.get('jockey', '')).strip()
            jockey_name, claim = parse_jockey_claim(jockey_raw)
            jockey_id  = reg.jockey(jockey_name) if jockey_name else None
            trainer_id = reg.trainer(row.get('trainer', ''))

            con.execute("""
                INSERT OR IGNORE INTO runner_declarations
                    (race_id, horse_id, draw, jockey_id, jockey_claim,
                     trainer_id, shoe_type)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                race_id, horse_id,
                safe_int(row.get('draw')),
                jockey_id, claim, trainer_id,
                row.get('shoe') or None,
            ])

            # horse_weight from declarations -> update runners
            hw = safe_int(row.get('horse_weight'))
            if hw:
                con.execute("""
                    UPDATE runners SET horse_body_wt=?
                    WHERE race_id=? AND horse_id=? AND horse_body_wt IS NULL
                """, [hw, race_id, horse_id])

        except Exception as e:
            log.error('declarations', None, None, None, None, str(e))


def load_equipment_changes(df: pd.DataFrame, con: duckdb.DuckDBPyConnection,
                            log: ImportLog, reg: EntityRegistry):
    df.columns = [c.strip().lower() for c in df.columns]
    for _, row in df.iterrows():
        try:
            venue_id  = reg.venue(row.get('venue', ''))
            meet_date = safe_date(row.get('meet_date'))
            race_no   = safe_int(row.get('race_no'))
            race_id   = reg.race_id(meet_date, venue_id, race_no)
            if race_id is None:
                continue
            horse_id = reg.horse(name=row.get('horse_name', ''))
            parsed   = parse_equipment_changes(row.get('equip_change', ''))
            con.execute("""
                INSERT OR IGNORE INTO equipment_changes
                    (race_id, horse_id,
                     equip_change_raw,
                     blinkers_on, blinkers_off,
                     visor_on, visor_off,
                     tongue_strap_on, tongue_strap_off,
                     earplugs_on, earplugs_off,
                     cheek_pieces_on, cheek_pieces_off,
                     hood_on, hood_off,
                     pacifier_on, pacifier_off)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                race_id, horse_id,
                parsed['equip_change_raw'],
                parsed['blinkers_on'],     parsed['blinkers_off'],
                parsed['visor_on'],        parsed['visor_off'],
                parsed['tongue_strap_on'], parsed['tongue_strap_off'],
                parsed['earplugs_on'],     parsed['earplugs_off'],
                parsed['cheek_pieces_on'], parsed['cheek_pieces_off'],
                parsed['hood_on'],         parsed['hood_off'],
                parsed['pacifier_on'],     parsed['pacifier_off'],
            ])
        except Exception as e:
            log.error('equipment_changes', None, None, None, None, str(e))


def load_ratings_changes(df: pd.DataFrame, con: duckdb.DuckDBPyConnection,
                          log: ImportLog, reg: EntityRegistry):
    df.columns = [c.strip().lower() for c in df.columns]
    for _, row in df.iterrows():
        try:
            meet_date = safe_date(row.get('meet_date'))
            horse_id  = reg.horse(name=row.get('horse_name', ''))
            con.execute("""
                INSERT OR IGNORE INTO ratings_changes
                    (meet_date, horse_id, race_range, new_rating, old_rating)
                VALUES (?, ?, ?, ?, ?)
            """, [
                meet_date, horse_id,
                row.get('race_range') or None,
                safe_int(row.get('new_rating')),
                safe_int(row.get('old_rating')),
            ])
        except Exception as e:
            log.error('ratings_changes', None, None, None, None, str(e))


def load_remarks(df: pd.DataFrame, con: duckdb.DuckDBPyConnection,
                  log: ImportLog, reg: EntityRegistry):
    df.columns = [c.strip().lower() for c in df.columns]
    for _, row in df.iterrows():
        try:
            meet_date = safe_date(row.get('meet_date'))
            race_no   = safe_int(row.get('race_no'))
            r = con.execute(
                "SELECT race_id FROM races WHERE meet_date=? AND race_no=?",
                [meet_date, race_no]).fetchone()
            if not r:
                continue
            race_id  = r[0]
            horse_id = reg.horse(name=row.get('horse_name', ''))
            con.execute("""
                INSERT OR IGNORE INTO horse_remarks
                    (race_id, horse_id, remark, remark_source)
                VALUES (?, ?, ?, ?)
            """, [
                race_id, horse_id,
                row.get('remark') or None,
                row.get('remark_source') or None,
            ])
        except Exception as e:
            log.error('remarks', None, None, None, None, str(e))


def load_jockey_changes(df: pd.DataFrame, con: duckdb.DuckDBPyConnection,
                         log: ImportLog, reg: EntityRegistry):
    df.columns = [c.strip().lower() for c in df.columns]
    for _, row in df.iterrows():
        try:
            venue_id  = reg.venue(row.get('venue', ''))
            meet_date = safe_date(row.get('meet_date'))
            race_no   = safe_int(row.get('race_no'))
            race_id   = reg.race_id(meet_date, venue_id, race_no)
            if race_id is None:
                continue
            horse_id = reg.horse(name=row.get('horse', ''))
            con.execute("""
                INSERT OR IGNORE INTO jockey_changes
                    (race_id, horse_id,
                     original_jockey_id, replacement_jockey_id, reason)
                VALUES (?, ?, ?, ?, ?)
            """, [
                race_id, horse_id,
                reg.jockey(norm_person(row.get('original_jockey', ''))),
                reg.jockey(norm_person(row.get('replacement_jockey', ''))),
                row.get('reason') or None,
            ])
        except Exception as e:
            log.error('jockey_changes', None, None, None, None, str(e))


def load_horse_actions(df: pd.DataFrame, con: duckdb.DuckDBPyConnection,
                        log: ImportLog, reg: EntityRegistry):
    df.columns = [c.strip().lower() for c in df.columns]
    for _, row in df.iterrows():
        try:
            meet_date = safe_date(row.get('date'))
            race_no   = safe_int(row.get('race_no'))
            r = con.execute(
                "SELECT race_id FROM races WHERE meet_date=? AND race_no=?",
                [meet_date, race_no]).fetchone()
            if not r:
                continue
            race_id  = r[0]
            horse_id = reg.horse(name=row.get('horse', ''))
            con.execute("""
                INSERT OR IGNORE INTO horse_actions
                    (race_id, horse_id, action, condition)
                VALUES (?, ?, ?, ?)
            """, [
                race_id, horse_id,
                row.get('action') or None,
                row.get('condition') or None,
            ])
        except Exception as e:
            log.error('horse_actions', None, None, None, None, str(e))


def load_penalties(df: pd.DataFrame, con: duckdb.DuckDBPyConnection,
                    log: ImportLog, reg: EntityRegistry):
    df.columns = [c.strip().lower() for c in df.columns]
    for _, row in df.iterrows():
        try:
            meet_date = safe_date(row.get('date'))
            race_no   = safe_int(row.get('race_no'))
            r = con.execute(
                "SELECT race_id FROM races WHERE meet_date=? AND race_no=?",
                [meet_date, race_no]).fetchone()
            race_id = r[0] if r else None

            person_raw = str(row.get('person', '')).strip()
            role       = str(row.get('role', '')).strip().upper() or None

            # Try to parse a clean jockey/trainer name from person_raw
            # e.g. "App.P.Trevor suspended for 21 Aug 2011" -> store raw, attempt partial match
            jockey_id  = None
            trainer_id = None
            if role == 'JOCKEY':
                # Attempt: first word(s) before a verb
                m = re.match(r'^([A-Za-z\s\.]+?)(?:\s+(?:fined|suspended|disqualified))',
                             person_raw, re.IGNORECASE)
                if m:
                    jockey_id = reg.jockey(norm_person(m.group(1)))

            con.execute("""
                INSERT INTO penalties
                    (race_id, person_raw, role,
                     jockey_id, trainer_id, action_type, penalty_raw)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                race_id, person_raw, role,
                jockey_id, trainer_id,
                row.get('action_type') or None,
                row.get('penalty') or None,
            ])
        except Exception as e:
            log.error('penalties', None, None, None, None, str(e))


# ─────────────────────────────────────────────────────────────
# DATASET REGISTRY
# Fill in your file paths here before running.
# Set either xlsx or csv to None if that file doesn't exist.
# ─────────────────────────────────────────────────────────────

DATASETS = [
    # Tier 1 — load first (meetings before races before runners)
    {
        "name": "results_meetings",
        "xlsx": None,   # e.g. "data/results/meetings.xlsx"
        "csv":  "../data/cleaned/results/meetings.csv",   # e.g. "data/results/meetings.csv"
        "fallback_columns": [],
        "merge_key": ["meet_date", "venue"],
        "loader": load_results_meetings,
    },
    {
        "name": "results_races",
        "xlsx": "../data/cleaned/results/races.xlsx",
        "csv":  "../data/cleaned/results/races.csv",
        "fallback_columns": [],
        "merge_key": ["meet_date", "venue", "race_no"],
        "loader": load_results_races,
    },
    {
        "name": "results_runners",
        "xlsx": "../data/cleaned/results/runners.xlsx",
        "csv":  "../data/cleaned/results/runners.csv",
        "fallback_columns": ["odds", "finish_time"],   # known Excel-mangled columns
        "merge_key": ["meet_date", "venue", "race_no", "horse_name"],
        "loader": load_results_runners,
    },
    {
        "name": "results_exotics",
        "xlsx": None,
        "csv":  "../data/cleaned/results/exotics.csv",
        "fallback_columns": [],
        "merge_key": ["meet_date", "venue", "pool_type"],
        "loader": load_results_exotics,
    },
    # Tier 2 — acceptances (load before declarations)
    {
        "name": "acceptances",
        "xlsx": "../data/cleaned/acceptances_cleaned/acceptances.xlsx",
        "csv":  None,
        "fallback_columns": [],
        "merge_key": ["meet_date", "venue", "race_no", "horse_name"],
        "loader": load_acceptances,
    },
    {
        "name": "acceptances_highest_ratings",
        "xlsx": "../data/cleaned/acceptances_cleaned/highest_ratings.xlsx",
        "csv":  None,
        "fallback_columns": [],
        "merge_key": ["meet_date", "horse_name"],
        "loader": load_highest_ratings,
    },
    {
        "name": "acceptances_medical",
        "xlsx": "../data/cleaned/acceptances_cleaned/medical.xlsx",
        "csv":  None,
        "fallback_columns": [],
        "merge_key": ["meet_date", "horse_name", "condition"],
        "loader": load_medical,
    },
    {
        "name": "acceptances_treadmill",
        "xlsx": "../data/cleaned/acceptances_cleaned/treadmill.xlsx",
        "csv":  None,
        "fallback_columns": [],
        "merge_key": ["meet_date", "horse_name", "date", "segment"],
        "loader": load_treadmill,
    },
    {
        "name": "acceptances_bandages",
        "xlsx": "../data/cleaned/acceptances_cleaned/bandages.xlsx",
        "csv":  None,
        "fallback_columns": [],
        "merge_key": ["meet_date", "horse_name", "race_no"],
        "loader": load_bandages,
    },
    {
        "name": "acceptances_equipment",
        "xlsx": "../data/cleaned/acceptances_cleaned/equipment.xlsx",
        "csv":  None,
        "fallback_columns": [],
        "merge_key": ["meet_date", "horse_name", "race_no"],
        "loader": load_equipment,
    },
    # Tier 2 — declarations
    {
        "name": "declarations",
        "xlsx": "../data/cleaned/declarations/declarations.xlsx",
        "csv":  None,
        "fallback_columns": [],
        "merge_key": ["meet_date", "venue", "race_no", "horse_name"],
        "loader": load_declarations,
    },
    {
        "name": "equipment_changes",
        "xlsx": "../data/cleaned/declarations/equipment_changes.xlsx",
        "csv":  None,
        "fallback_columns": [],
        "merge_key": ["meet_date", "venue", "race_no", "horse_name"],
        "loader": load_equipment_changes,
    },
    # Tier 1 — ratings
    {
        "name": "ratings_changes",
        "xlsx": None,
        "csv":  "../data/cleaned/ratings/ratings_change.csv",
        "fallback_columns": [],
        "merge_key": ["meet_date", "horse_name"],
        "loader": load_ratings_changes,
    },
    # Tier 3 — regulatory (can load any time after races exist)
    {
        "name": "remarks",
        "xlsx": None,
        "csv":  "../data/cleaned/ratings/remarks.csv",
        "fallback_columns": [],
        "merge_key": ["meet_date", "race_no", "horse_name"],
        "loader": load_remarks,
    },
    {
        "name": "jockey_changes",
        "xlsx": None,
        "csv":  "../data/cleaned/raceday_report/jockey_changes.csv",
        "fallback_columns": [],
        "merge_key": ["meet_date", "venue", "race_no", "horse"],
        "loader": load_jockey_changes,
    },
    {
        "name": "horse_actions",
        "xlsx": None,
        "csv":  "../data/cleaned/raceday_report/summary_horse_actions.csv",
        "fallback_columns": [],
        "merge_key": ["date", "venue", "race_no", "horse"],
        "loader": load_horse_actions,
    },
    {
        "name": "penalties",
        "xlsx": None,
        "csv":  "../data/cleaned/raceday_report/summary_penalties.csv",
        "fallback_columns": [],
        "merge_key": ["date", "race_no", "person"],
        "loader": load_penalties,
    },
]


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def run_etl(db_path: str, schema_path: str):
    print(f"\n{'='*60}")
    print(f"  RWITC ETL — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  DB     : {db_path}")
    print(f"  Schema : {schema_path}")
    print(f"{'='*60}\n")

    con = duckdb.connect(db_path)
    log = ImportLog()

    # Apply schema — execute statement by statement after stripping comments.
    # DuckDB's con.execute() chokes on inline -- comments when given a full
    # multi-statement string, so we split on ';' and clean each chunk first.
    schema_sql = Path(schema_path).read_text(encoding='utf-8')

    def _strip_comments(sql: str) -> str:
        """Remove -- line comments and /* block comments */ from a SQL string."""
        sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
        sql = re.sub(r'--[^\n]*', '', sql)
        return sql

    statements = [
        s.strip()
        for s in _strip_comments(schema_sql).split(';')
        if s.strip()
    ]

    applied = 0
    for stmt in statements:
        try:
            con.execute(stmt)
            applied += 1
        except Exception as e:
            print(f"[SCHEMA ERROR] {e}\nStatement: {stmt[:120]}...")
            raise   # schema errors are fatal

    print(f"Schema applied ({applied} statements).\n")

    # Initialise entity registry (loads existing IDs from DB)
    reg = EntityRegistry(con)

    skipped = []

    for ds in DATASETS:
        name = ds['name']

        if ds['xlsx'] is None and ds['csv'] is None:
            skipped.append(name)
            continue

        try:
            df = read_source(
                dataset_name=name,
                xlsx_path=ds.get('xlsx'),
                csv_path=ds.get('csv'),
                fallback_columns=ds.get('fallback_columns'),
                log=log,
                merge_key=ds.get('merge_key'),
            )
            ds['loader'](df, con, log, reg)
            log.summary(name)

        except Exception as e:
            print(f"\n[FATAL] {name}: {e}")
            traceback.print_exc()
            # Never abort; continue to next dataset
            log.error(name, None, None, None, None, f"FATAL: {e}")

    if skipped:
        print(f"\nSkipped (no paths set): {', '.join(skipped)}")

    # Save import log
    log_path = f"log/import_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    log.save(log_path)

    con.close()
    print("\nETL complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RWITC ETL Pipeline')
    parser.add_argument('--db',     default='rwitc.db',         help='DuckDB database file')
    parser.add_argument('--schema', default='schema.sql', help='Schema SQL file')
    args = parser.parse_args()
    run_etl(args.db, args.schema)