#!/usr/bin/env python3
"""
RWITC Acceptances HTML Parser
================================
Parses horse racing acceptance pages from rwitc.com (2010–2026+) into a structured CSV.

Handles both HTML formats:
  - Legacy  (2010-era): flat stream of <table class='table table-bordered'> blocks,
    BUT rendered by browsers (and BS4) as deeply-nested tables because closing
    </table> tags are missing between races.  Fix: anchor on conteraceHeading
    headers and use recursive=False row iteration.
  - Modern (2026-era):  alternating pairs of bot10 header tables + tbbody runner
    tables at the TOP level of the soup.  Fix: iterate top-level tables in pairs.

Requirements: pip install beautifulsoup4

Usage:
  1. Edit the CONFIGURATION section below
  2. Run: python rwitc_acceptances_parser.py
"""

import os
import re
import csv
import sys
import glob
import logging
from bs4 import BeautifulSoup
import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_PATH     = "./raw_html/acceptances_2010-2017"   # single file OR folder of .html/.htm
OUTPUT_DIR     = "./raw"
OUTPUT_FILE    = "acceptances_10-17+.csv"
WRITE_MODE     = "a"          # "w" = overwrite, "a" = append
INPUT_ENCODING = "utf-8"      # try "latin-1" if utf-8 fails
LOG_LEVEL      = logging.INFO

# ============================================================================
# COLUMN DEFINITIONS
# ============================================================================

ACCEPTANCE_COLS = [
    "meet_date", "venue", "race_no", "race_name",
    "distance", "time",
    "horse_no", "horse_name", "horse_seq",
    "color_sex", "age", "weight", "rating",
    "sire", "sire_nat", "dam", "dam_nat",
    "trainer", "weight_update",
]

MEDICAL_COLS = [
    "meet_date",
    "venue",
    "race_no",
    "horse_name",
    "event_date",
    "event_type",
    "raw_text",
]

# ============================================================================
# SETUP
# ============================================================================

logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


# ============================================================================
# SHARED HELPERS
# ============================================================================

def clean(text):
    """Normalise whitespace and strip."""
    if not text:
        return ""
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_nationality(name):
    """
    'Naseem El Fajr(IRE)' -> ('Naseem El Fajr', 'IRE')   legacy paren format
    'Euqranian[USA]'      -> ('Euqranian',      'USA')   modern bracket format
    'Cagliari'            -> ('Cagliari',        '')
    """
    # Modern: [NAT]
    m = re.search(r"\[(\w+)\]\s*$", name)
    if m:
        return name[: m.start()].strip(), m.group(1)
    # Legacy: (NAT)  — ignore (#) placeholders
    m = re.search(r"\((\w+)\)\s*$", name)
    if m and m.group(1) not in ("", "#"):
        return name[: m.start()].strip(), m.group(1)
    return name.strip(), ""


def parse_date_from_text(text):
    """'Thursday 15th July 2010' -> '2010-07-15'"""
    m = re.search(
        r"(\d{1,2})\s*(?:st|nd|rd|th)?\s+"
        r"(January|February|March|April|May|June|July|"
        r"August|September|October|November|December)\s+"
        r"(\d{4})",
        text, re.IGNORECASE,
    )
    if m:
        day   = int(m.group(1))
        month = MONTHS[m.group(2).lower()]
        year  = int(m.group(3))
        return f"{year:04d}-{month:02d}-{day:02d}"
    return ""


def parse_distance_meters(text):
    """'(About) 1400 Metres.' -> '1400'"""
    m = re.search(r"(\d+)\s*(?:Metres|Mtrs)", text, re.IGNORECASE)
    return m.group(1) if m else ""


def parse_time(text):
    """Extract 'Time: 2.30 P.M.' -> '2.30 P.M.'"""
    m = re.search(r"Time\s*:\s*([\d.]+\s*[APap]\.?\s*[Mm]\.?)", text)
    return clean(m.group(1)) if m else ""


def parse_venue(text, filepath=""):
    upper = text.upper()
    if "MUMBAI" in upper:
        return "Mumbai"
    if "PUNE" in upper:
        return "Pune"
    fname = os.path.basename(filepath).upper()
    if "PUNE" in fname:
        return "Pune"
    return "Mumbai"


def parse_breeding(breed_text):
    """'Brave Hunter-Cagliari(IRE)' -> (sire, sire_nat, dam, dam_nat)"""
    sire, sire_nat, dam, dam_nat = "", "", "", ""
    if "-" in breed_text:
        sire_raw, dam_raw = breed_text.split("-", 1)
        sire, sire_nat   = extract_nationality(sire_raw.strip())
        dam,  dam_nat    = extract_nationality(dam_raw.strip())
    return sire, sire_nat, dam, dam_nat


# ============================================================================
# FORMAT DETECTION
# ============================================================================

def detect_format(soup):
    """
    'modern' if the page has class=perform_data runner rows (2026-era).
    'legacy' otherwise (2010-era).
    """
    if soup.find("tr", class_="perform_data"):
        return "modern"
    return "legacy"


# ============================================================================
# HEADER PARSING
# ============================================================================

def parse_header(soup, filepath=""):
    """Return (meet_date, venue) from the page heading block."""
    heading = (
        soup.find("div", class_="pageHeading")
        or soup.find("div", class_="pageHeader")
    )
    text = clean(heading.get_text()) if heading else ""
    meet_date = parse_date_from_text(text)
    venue     = parse_venue(text, filepath)
    if not meet_date:
        log.warning("Could not extract meet date — check HTML header")
    return meet_date, venue

# ============================================================================
# MEDICAL HISTORY PARSING
# ============================================================================

def parse_medical_event(text):
    """
    Extract MULTIPLE events from a single malformed string.
    Returns a list of events.
    """

    text = clean(text)

    # Split on date boundaries
    parts = re.split(r'(?=\d{2}/\d{2}/\d{2})', text)

    events = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        m = re.match(r"(\d{2}/\d{2}/\d{2})\s+(.*)", part)
        if not m:
            continue

        date_str = m.group(1)
        rest = m.group(2)

        # convert date
        try:
            d = datetime.datetime.strptime(date_str, "%d/%m/%y")
            event_date = d.strftime("%Y-%m-%d")
        except:
            event_date = ""

        tokens = rest.split()

        # detect horse name (uppercase sequence)
        horse_parts = []
        i = 0
        while i < len(tokens) and (tokens[i].isupper() or "'" in tokens[i]):
            horse_parts.append(tokens[i])
            i += 1

        horse_name = " ".join(horse_parts)
        event_type = " ".join(tokens[i:])

        events.append({
            "event_date": event_date,
            "horse_name": horse_name,
            "event_type": event_type,
            "raw_text": part,
        })

    return events

# ============================================================================
# MODERN FORMAT PARSER  (2026-era)
# ============================================================================
#
# Structure (top-level, no nesting):
#   <table class='table bot10'>          <- race header wrapper
#     <table style='background:#11a14e'> <- inner green header
#   <table class='table table-bordered tbbody'>  <- runners
#   <table class='table bot10'>          <- next race ...
#   ...
#
# The green inner table has class=['table','table-bordered','padd'] and
# style containing '#11a14e'.  The runner table has class 'tbbody'.
# They alternate at the TOP level of the document.

def _modern_race_info(green_table):
    """Extract race_no, race_name, distance, time from the green inner table."""
    ths = green_table.find_all("th")
    race_no   = ""
    race_name = ""
    detail    = ""

    for th in ths:
        t = clean(th.get_text())
        m = re.match(r"^(\d+)\.$", t)
        if m:
            race_no = m.group(1)
        elif len(t) > 5:
            detail = t

    if not race_no:
        m = re.match(r"^(\d+)\.", detail)
        if m:
            race_no = m.group(1)
            detail  = detail[m.end():].strip()

    name_m = re.match(r"^(.*?)\s*\(", detail)
    if name_m:
        race_name = name_m.group(1).strip()
    else:
        race_name = detail.split("\n")[0].strip()

    return {
        "race_no":  race_no,
        "race_name": race_name,
        "distance": parse_distance_meters(detail),
        "time":     parse_time(detail),
    }


def _modern_runners(tbbody_table, meet_date, venue, race_info):
    """Parse perform_data rows from a tbbody runner table."""
    weight_update = ""

    # Weight note: a colspan>=5 td at the bottom
    for td in tbbody_table.find_all("td"):
        try:
            span = int(td.get("colspan", "1"))
        except (ValueError, TypeError):
            span = 1
        if span >= 5:
            t = clean(td.get_text())
            if "weight" in t.lower():
                weight_update = t
                break

    rows = []
    medical_rows = []
    
    for row in tbbody_table.find_all("tr", class_="perform_data"):
        cells = row.find_all("td")
        if len(cells) < 7:
            continue

        # Horse number + name + seq
        first     = cells[0]
        cell_text = clean(first.get_text())
        num_m     = re.match(r"(\d+)", cell_text)
        horse_no  = num_m.group(1) if num_m else ""

        link       = first.find("a")
        horse_name = clean(link.get_text()) if link else ""
        horse_seq  = ""
        if link:
            m = re.search(r"horseseq=(\d+)", link.get("href", ""))
            if m:
                horse_seq = m.group(1)

        color_sex = clean(cells[1].get_text())
        age       = clean(cells[2].get_text())
        weight    = clean(cells[3].get_text())
        rating    = clean(cells[4].get_text())

        sire, sire_nat, dam, dam_nat = parse_breeding(clean(cells[5].get_text()))
        trainer = clean(cells[6].get_text())

        rows.append({
            "meet_date":     meet_date,
            "venue":         venue,
            "race_no":       race_info["race_no"],
            "race_name":     race_info["race_name"],
            "distance":      race_info["distance"],
            "time":          race_info["time"],
            "horse_no":      horse_no,
            "horse_name":    horse_name,
            "horse_seq":     horse_seq,
            "color_sex":     color_sex,
            "age":           age,
            "weight":        weight,
            "rating":        rating,
            "sire":          sire,
            "sire_nat":      sire_nat,
            "dam":           dam,
            "dam_nat":       dam_nat,
            "trainer":       trainer,
            "weight_update":        weight_update,
        })
    return rows


def parse_modern(soup, meet_date, venue):
    """
    Walk ALL tables in document order, keeping only bot10 and tbbody tables.
    They appear in alternating pairs: bot10 (header) then tbbody (runners).
    This approach is resilient to any level of div/container nesting.
    """
    all_rows = []

    # Collect bot10 and tbbody tables in document order
    relevant = [
        t for t in soup.find_all("table")
        if "bot10" in (t.get("class") or []) or "tbbody" in (t.get("class") or [])
    ]

    i = 0
    while i < len(relevant):
        tbl = relevant[i]
        cls = tbl.get("class", [])

        if "bot10" in cls:
            green = tbl.find("table", style=re.compile(r"11a14e"))
            if green:
                race_info = _modern_race_info(green)
                if race_info["race_no"]:
                    # The immediately following relevant table should be tbbody
                    j = i + 1
                    if j < len(relevant) and "tbbody" in (relevant[j].get("class") or []):
                        rows = _modern_runners(relevant[j], meet_date, venue, race_info)
                        all_rows.extend(rows)
                        log.debug(
                            f"  Modern race {race_info['race_no']} "
                            f"({race_info['race_name']}): {len(rows)} runners"
                        )
                        i = j  # skip the tbbody we just consumed
        i += 1

    return all_rows


# ============================================================================
# LEGACY FORMAT PARSER  (2010-era)
# ============================================================================
#
# Root problem: the legacy HTML omits closing </table> tags between races,
# so BS4 nests all races inside the first one.  find_all('table') returns
# them all including nested duplicates.
#
# Fix: anchor on <table class='conteraceHeading'> elements — there is exactly
# one per race.  Walk UP to its nearest table-bordered ancestor to get the
# outer race table, then use find_all('tr', recursive=False) so we only see
# that race's own rows, never rows from nested child races.

def _legacy_race_info(header_table):
    """Parse race no / name / distance / time from a conteraceHeading table."""
    ths = header_table.find_all("th")
    if len(ths) < 2:
        return None

    race_no_m = re.match(r"(\d+)\.", clean(ths[0].get_text()))
    if not race_no_m:
        return None
    race_no = race_no_m.group(1)

    detail    = clean(ths[1].get_text())
    name_m    = re.match(r"^(.*?)\s*\(", detail)
    race_name = name_m.group(1).strip() if name_m else detail

    return {
        "race_no":   race_no,
        "race_name": race_name,
        "distance":  parse_distance_meters(detail),
        "time":      parse_time(detail),
    }


def _legacy_runners(outer_table, meet_date, venue, race_info):
    """
    Extract runners from an outer table-bordered table.
    Uses recursive=False so we never accidentally walk into a nested race table.
    """
    rows          = []
    medical_rows  = []
    weight_update = ""

    direct_trs = outer_table.find_all("tr", recursive=False)

    for tr in direct_trs:
        # Only look at direct children td/th of this tr
        tds = tr.find_all("td", recursive=False)
        ths = tr.find_all("th", recursive=False)

        # Weight-update note row: single wide td
        if len(tds) == 1:
            t = clean(tds[0].get_text())
            if "weight" in t.lower():
                weight_update = t
                for r in rows:
                    r["weight_update"] = weight_update

            # medical event detection
            events = parse_medical_event(t)

            for event in events:
                medical_rows.append({
                    "meet_date": meet_date,
                    "venue": venue,
                    "race_no": race_info["race_no"],
                    "horse_name": event["horse_name"],
                    "event_date": event["event_date"],
                    "event_type": event["event_type"],
                    "raw_text": event["raw_text"],
                })

            continue # not a runner row

        # Column header row
        if ths and not tds:
            continue

        # Runner row: needs exactly 5 tds (Horse, Weight, Rating, Breeding, Trainer)
        if len(tds) < 5:
            continue

        # Horse number + name + seq
        first     = tds[0]
        cell_text = clean(first.get_text())
        num_m     = re.match(r"(\d+)", cell_text)
        horse_no  = num_m.group(1) if num_m else ""

        link       = first.find("a")
        horse_name = clean(link.get_text()) if link else ""
        # Strip leading "1. " prefix that legacy pages include in link text
        horse_name = re.sub(r"^\d+\.\s*", "", horse_name)
        horse_seq  = ""
        if link:
            m = re.search(r"horseseq=(\d+)", link.get("href", ""))
            if m:
                horse_seq = m.group(1)

        weight = clean(tds[1].get_text())
        rating = clean(tds[2].get_text())

        sire, sire_nat, dam, dam_nat = parse_breeding(clean(tds[3].get_text()))
        trainer = clean(tds[4].get_text())

        rows.append({
            "meet_date":     meet_date,
            "venue":         venue,
            "race_no":       race_info["race_no"],
            "race_name":     race_info["race_name"],
            "distance":      race_info["distance"],
            "time":          race_info["time"],
            "horse_no":      horse_no,
            "horse_name":    horse_name,
            "horse_seq":     horse_seq,
            "color_sex":     "",   # not in legacy format
            "age":           "",   # not in legacy format
            "weight":        weight,
            "rating":        rating,
            "sire":          sire,
            "sire_nat":      sire_nat,
            "dam":           dam,
            "dam_nat":       dam_nat,
            "trainer":       trainer,
            "weight_update": weight_update,
        })

    return rows, medical_rows


def parse_legacy(soup, meet_date, venue):
    """
    Strategy: find every conteraceHeading table (one per race).
    Walk UP to its enclosing table-bordered ancestor.
    Extract ONLY that table's own direct rows.
    """
    all_rows = []
    all_medical = []

    header_tables = soup.find_all("table", class_="conteraceHeading")
    log.debug(f"  Legacy: found {len(header_tables)} race headers")

    for h in header_tables:
        race_info = _legacy_race_info(h)
        if not race_info:
            continue

        # Walk up the DOM to find the nearest table-bordered ancestor
        outer = h.find_parent("table", class_="table-bordered")
        if not outer:
            log.warning(f"  No outer table found for race {race_info['race_no']}")
            continue

        rows, medical_rows = _legacy_runners(outer, meet_date, venue, race_info)
        all_rows.extend(rows)
        all_medical.extend(medical_rows)
        log.debug(
            f"  Legacy race {race_info['race_no']} "
            f"({race_info['race_name']}): {len(rows)} runners"
        )

    return all_rows, all_medical


# ============================================================================
# FILE ORCHESTRATOR
# ============================================================================

def parse_acceptance_file(filepath):
    log.info(f"Parsing: {filepath}")
    try:
        with open(filepath, "r", encoding=INPUT_ENCODING, errors="replace") as f:
            html = f.read()
    except Exception as e:
        log.error(f"Cannot read {filepath}: {e}")
        return []

    soup      = BeautifulSoup(html, "html.parser")
    fmt       = detect_format(soup)
    meet_date, venue = parse_header(soup, filepath)

    log.info(f"  Format={fmt}  Date={meet_date}  Venue={venue}")

    if fmt == "modern":
        rows = parse_modern(soup, meet_date, venue)
    else:
        rows = parse_legacy(soup, meet_date, venue)

    log.info(f"  Extracted {len(rows)} runner rows")
    return rows


# ============================================================================
# CSV OUTPUT
# ============================================================================

def write_csv(rows):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    need_header = (WRITE_MODE == "w") or not os.path.exists(path) or os.path.getsize(path) == 0
    mode = "w" if WRITE_MODE == "w" else "a"

    with open(path, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ACCEPTANCE_COLS, extrasaction="ignore")
        if need_header:
            w.writeheader()
        w.writerows(rows)

    log.info(f"Wrote {len(rows)} rows -> {path}")

def write_medical_csv(rows):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "medical_events_10-17+.csv")

    need_header = not os.path.exists(path) or os.path.getsize(path) == 0

    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MEDICAL_COLS, extrasaction="ignore")
        if need_header:
            w.writeheader()
        w.writerows(rows)

    log.info(f"Wrote {len(rows)} medical rows -> {path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    if os.path.isfile(INPUT_PATH):
        files = [INPUT_PATH]
    elif os.path.isdir(INPUT_PATH):
        files = sorted(
            glob.glob(os.path.join(INPUT_PATH, "*.html"))
            + glob.glob(os.path.join(INPUT_PATH, "*.htm"))
        )
    else:
        log.error(f"INPUT_PATH not found: {INPUT_PATH}")
        sys.exit(1)

    if not files:
        log.error(f"No .html/.htm files found in: {INPUT_PATH}")
        sys.exit(1)

    log.info(f"Found {len(files)} file(s) to process")

    all_rows = []
    med_rows = []
    for fp in files:
        rows, medical = parse_acceptance_file(fp)
        all_rows.extend(rows)
        med_rows.extend(medical)

    write_csv(all_rows)
    write_medical_csv(med_rows)
    log.info(f"Done — {len(all_rows)} total acceptance rows")


if __name__ == "__main__":
    main()