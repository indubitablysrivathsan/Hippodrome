#!/usr/bin/env python3
"""
RWITC Acceptances DOC/HTM Parser
===================================
Parses Word-saved-as-HTML (.htm/.doc) acceptance documents from RWITC
into structured CSVs.  These files are produced by Microsoft Word and
contain <p class="MsoPlainText"> paragraphs with fixed-width Courier text.

This is a DIFFERENT parser from the rwitc_acceptances_parser.py which
handles the rwitc.com web-page HTML format.

Data extracted (7 CSVs):
  acceptances.csv      — one row per horse per race (core data)
  medical.csv          — surgery / illness / injury history
  equipment.csv        — shoes, bits, hoods per horse
  bandages.csv         — bandage records per horse
  highest_ratings.csv  — peak rating ever achieved per horse
  trainer_changes.csv  — recent trainer changes
  pools.csv            — jackpot/treble/tanala pool race assignments

Requirements: pip install beautifulsoup4

Usage:
  1. Edit CONFIGURATION below
  2. python rwitc_doc_parser.py
"""

import os, re, csv, sys, glob, logging
from bs4 import BeautifulSoup

# ============================================================================
# CONFIGURATION
# ============================================================================

#INPUT_PATH   = "./raw_doc/acceptances_htm_2018-/2026-03-29.htm" # single .htm/.html file OR folder
INPUT_PATH   = "./raw_doc/acceptances_htm_2018-/2018-01-04.htm" # single .htm/.html file OR folder 
#INPUT_PATH   = "./raw_doc/acceptances_htm_2018-" # single .htm/.html file OR folder 
OUTPUT_DIR   = "./raw/acceptances_test/2018-"
WRITE_MODE   = "w"           # "w" = overwrite, "a" = append
INPUT_ENCODING = "utf-8"
LOG_LEVEL    = logging.INFO

# ============================================================================
# OUTPUT FILE NAMES
# ============================================================================

FILES = {
    "acceptances":    "acceptances.csv",
    "medical":        "medical.csv",
    "equipment":      "equipment.csv",
    "bandages":       "bandages.csv",
    "highest_ratings":"highest_ratings.csv",
    "trainer_changes":"trainer_changes.csv",
    "pools":          "pools.csv",
    "swimming":       "swimming.csv",
    "treadmill":      "treadmill.csv",
}

# ============================================================================
# COLUMN DEFINITIONS
# ============================================================================

ACCEPTANCE_COLS = [
    "meet_date", "venue", "session", "race_no", "race_name", "conditions",
    "distance", "time", "foreign_jockeys",
    "horse_name", "weight", "rating",
    "weight_update_handicap", "weight_update_acceptance",
    "late_entry_replaces",
]

MEDICAL_COLS = [
    "meet_date", "race_no", "horse_name", "condition", "date",
]

EQUIPMENT_COLS = [
    "meet_date", "race_no", "horse_name",
    "shoe_type", "bit", "hood_other",
]

BANDAGE_COLS = [
    "meet_date", "race_no", "horse_name", "bandage_type",
]

SWIMMING_COLS = [
    "meet_date", "race_no", "horse_name", "date", "rounds"
]

TREADMILL_COLS = [
    "meet_date", "race_no", "horse_name", "date",
    "segment", "speed", "distance"
]

HIGHEST_RATING_COLS = [
    "meet_date", "race_no", "horse_name", "highest_rating", "achieved_date",
]

TRAINER_CHANGE_COLS = [
    "meet_date", "race_no", "horse_name",
    "new_trainer", "old_trainer",
    "last_run_date", "took_charge_date",
]

POOL_COLS = [
    "meet_date", "pool_type", "races",
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
# HELPERS
# ============================================================================

def clean(text):
    if not text:
        return ""
    text = text.replace("\xa0", " ").replace("\u2019", "'")
    return re.sub(r"\s+", " ", text).strip()


def parse_date_text(text):
    """Handles BOTH:
       - 4TH JANUARY, 2018
       - 30TH, MARCH, 2026
    """
    m = re.search(
        r"(\d{1,2})\s*(?:ST|ND|RD|TH)?\s*,?\s*"
        r"(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|"
        r"AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)"
        r"\s*,?\s*(\d{4})",
        text,
        re.I
    )
    if m:
        day   = int(m.group(1))
        month = MONTHS[m.group(2).lower()]
        year  = int(m.group(3))
        return f"{year:04d}-{month:02d}-{day:02d}"
    return ""


def parse_ddmmyyyy(text):
    """'18/04/2025' -> '2025-04-18'"""
    m = re.match(r"(\d{2})/(\d{2})/(\d{4})", text.strip())
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    return text.strip()


def is_section_divider(text):
    """Lines like ===, ***, --- that separate document sections."""
    stripped = re.sub(r"[\s=\-\*]", "", text)
    return len(stripped) == 0 and len(text) > 5


# ============================================================================
# DOCUMENT TEXT EXTRACTION
# ============================================================================

def get_paragraphs(soup):
    texts = []

    # Existing paragraphs
    for p in soup.find_all("p"):
        t = clean(p.get_text())
        if t:
            texts.append(t)

    # ADD THIS: table cells (fixes 2018 bandages)
    for td in soup.find_all("td"):
        t = clean(td.get_text())
        if t:
            texts.append(t)

    return texts

def get_raw_paragraphs(soup):
    """
    Return paragraphs with ONLY \xa0→space and \n→space substitution.
    Multiple spaces are preserved — required for fixed-width swimming grid.
    """
    texts = []
    for p in soup.find_all("p"):
        t = p.get_text().replace("\xa0", " ").replace("\n", " ")
        if t.strip():
            texts.append(t)
    for td in soup.find_all("td"):
        t = td.get_text().replace("\xa0", " ").replace("\n", " ")
        if t.strip():
            texts.append(t)
    return texts


# ============================================================================
# SECTION SPLITTER
# ============================================================================

# Section markers (in order of appearance in document)
_SECTION_MARKERS = {
    "shoe_bits":       "shoe and bits are stated below",
    "bandages":        "bandages on record",
    "swimming":        "swimming record",
    "treadmill":       "treadmill data",
    "highest_ratings": "highest rating",
    "trainer_changes": "trainer changes",
    "swimming":        "swimming",
    "treadmill":       "treadmill",
}


def normalize_text(s):
    return re.sub(r"[^a-z]", "", s.lower())


def split_sections(paras):
    sections = {"header": []}
    current  = "header"

    for p in paras:
        norm = normalize_text(p)

        matched = False
        for name, marker in _SECTION_MARKERS.items():
            if marker.replace(" ", "") in norm:
                current = name
                sections.setdefault(current, [])
                matched = True
                break

        if not matched:
            sections.setdefault(current, []).append(p)

    return sections


# ============================================================================
# HEADER / MEETING INFO PARSER
# ============================================================================

def parse_meeting_info(header_paras):
    """Extract meet_date, venue, session from the header paragraphs."""
    info = {"meet_date": "", "venue": "Mumbai", "session": ""}
    for p in header_paras[:5]:
        if not info["meet_date"]:
            info["meet_date"] = parse_date_text(p)
        if "MUMBAI" in p.upper():
            info["venue"] = "Mumbai"
        elif "PUNE" in p.upper():
            info["venue"] = "Pune"
        if "EVENING" in p.upper():
            info["session"] = "Evening"
    return info


# ============================================================================
# RACE / ACCEPTANCE SECTION PARSER
# ============================================================================

# Patterns for runner lines (after clean() has collapsed all whitespace to single spaces):
# "( 38) ALLEZ L'ETOILE 59 ( 34) SEMURG 57 ( 29) HER CHARGE 54.5"
# "( - ) BEAST MODE 56 ( - ) REMARKABLE 56 ( - ) RED ROSE 54.5"
# Weight always appears immediately before the next "(" or end-of-string.
_RUNNER_TOKEN = re.compile(
    r"\(\s*(-|\d+)\s*\)\s+"           # ( rating )
    r"([A-Z][A-Z0-9 '&/\.\-]+?)\s+"  # NAME (non-greedy)
    r"(\d+\.?\d*)"                    # weight
    r"(?=\s*(?:\(|$))"               # lookahead: next ( or end-of-line
)

# Race header: "1. The Mid-Day Trophy  (Class IV; H'cap...)"
_RACE_HEADER = re.compile(r"^(\d+)\.\s+(.+)")

# Weight update lines: "# Weights raised by 3 kg." or "* Weights raised..."
_WEIGHT_HANDICAP   = re.compile(r"#\s*Weights?\s+(raised|lowered)\s+by\s+([\d.]+)\s*kg", re.I)
_WEIGHT_ACCEPTANCE = re.compile(r"\*\s*Weights?\s+(raised|lowered)\s+by\s+([\d.]+)\s*kg", re.I)

# Late entry: "MUQADDAR (late) GHOST OF VICTORY"
_LATE_ENTRY = re.compile(r"^([A-Z][A-Z0-9 '&/\-\.]+?)\s+\(late\)\s+(.+)$", re.I)

# Medical: "MARIUS - TIE FORWARD SURGERY 18/04/2025"
_MEDICAL = re.compile(r"^([A-Z][A-Z0-9 '&/\-\.]+?)\s+-\s+(.+?)\s+(\d{2}/\d{2}/\d{4})(.*)$")

# Distance line: "(About) 1400 Metres.  Time: 5:30 PM"
_DISTANCE = re.compile(r"\(About\)\s+(\d+)\s+Metres", re.I)
_TIME      = _TIME = re.compile(
    r"Time:\s*([\d:.]+\s*[AP]\.?M\.?)",
    re.I
)


def _parse_runner_line(line):
    """
    Parse up to 3 runners from a fixed-width Courier line.
    Returns list of (rating, name, weight) tuples.
    """
    return [(m.group(1).strip(), clean(m.group(2)), m.group(3))
            for m in _RUNNER_TOKEN.finditer(line)]


def parse_races(header_paras, meet_date, venue, session):
    """
    Parse the main acceptances section (header_paras up to shoe/bits marker).
    Returns (acceptances, medicals, pools).
    """
    acceptances = []
    medicals    = []
    pools       = []

    current_race = None
    wt_handicap  = ""
    wt_acceptance = ""
    foreign_jockeys = False
    current_horse = ""
    buffer = ""
    condition_buffer = ""

    def flush_race():
        pass  # race info is attached per-runner inline

    for p in header_paras:
        # ---- Pool lines (ROBUST VERSION) ----

        POOL_MAP = {
            "JKP": "JACKPOT",
            "JACKPOT": "JACKPOT",
            "SUPER JKP": "SUPER JACKPOT",
            "SUPER JACKPOT": "SUPER JACKPOT",
            "TREBLE": "TREBLE",
            "FIRST TREBLE": "TREBLE",
            "TANALA": "TANALA",
        }

        pool_m = re.search(
            r"([A-Z\s]+?)\s+POOL\s+RACES?\s*:\s*(.+)",
            p.upper()
        )

        if pool_m:
            raw_type = pool_m.group(1).strip()
            races    = pool_m.group(2).strip()

            # Normalize spacing (important for "SUPER   JKP" type cases)
            raw_type = re.sub(r"\s+", " ", raw_type)

            pool_type = POOL_MAP.get(raw_type, raw_type)

            pools.append({
                "meet_date": meet_date,
                "pool_type": pool_type,
                "races": races,
            })
            continue

        # ---- Race header ----
        race_m = _RACE_HEADER.match(p)
        if race_m:
            current_race   = {"race_no": race_m.group(1), "race_name": "", "conditions": "",
                               "distance": "", "time": ""}
            rest = clean(race_m.group(2))

            # ---- Extract time ----
            time_m = _TIME.search(rest)
            if time_m:
                current_race["time"] = time_m.group(1).strip()

            # ---- Extract distance ----
            dist_m = re.search(r"(\d{3,4})\s+Metres", rest, re.I)
            if dist_m:
                current_race["distance"] = dist_m.group(1)

            # ---- Remove trailing distance/time from string ----
            rest = re.sub(r"\(?About\)?\s*\d{3,4}\s+Metres.*", "", rest, flags=re.I)
            rest = re.sub(r"Time:\s*[\d:]+\s*[APM]+", "", rest, flags=re.I)

            # ---- Extract race name (before first bracket) ----
            name_m = re.match(r"^(.*?)\s*\(", rest)
            if name_m:
                current_race["race_name"] = name_m.group(1).strip()
            else:
                current_race["race_name"] = rest.strip()

            # ---- Extract first bracket as conditions ----
            cond_m = re.search(r"\((.*?)\)", rest)
            if cond_m:
                current_race["conditions"] = cond_m.group(1).strip()

            wt_handicap    = ""
            wt_acceptance  = ""
            foreign_jockeys = False
            continue

        if current_race is None:
            continue

        # ---- Distance / Time line ----
        dist_m = _DISTANCE.search(p)
        if dist_m:
            current_race["distance"] = dist_m.group(1)
            time_m = _TIME.search(p)
            if time_m:
                current_race["time"] = time_m.group(1).strip()
            continue

        # ---- Foreign jockeys ----
        if re.search(r"foreign jockeys eligible", p, re.I):
            foreign_jockeys = True
            continue

        # ---- Weight update lines — backfill all runners already added for this race ----
        wh_m = _WEIGHT_HANDICAP.search(p)
        if wh_m:
            direction = wh_m.group(1).lower()
            kg        = wh_m.group(2)
            wt_handicap = f"Weights {direction} by {kg} kg at Handicap stage."
            for row in acceptances:
                if row["race_no"] == current_race["race_no"]:
                    row["weight_update_handicap"] = wt_handicap
            continue

        wa_m = _WEIGHT_ACCEPTANCE.search(p)
        if wa_m:
            direction = wa_m.group(1).lower()
            kg        = wa_m.group(2)
            wt_acceptance = f"Weights {direction} by {kg} kg at Acceptance stage."
            for row in acceptances:
                if row["race_no"] == current_race["race_no"]:
                    row["weight_update_acceptance"] = wt_acceptance
            continue

        # ---- Late entry line ----
        late_m = _LATE_ENTRY.match(p)
        if late_m:
            horse   = clean(late_m.group(1))
            replaces = clean(late_m.group(2))
            # Tag existing acceptance row if already added
            for row in reversed(acceptances):
                if row["horse_name"] == horse and row["race_no"] == current_race["race_no"]:
                    row["late_entry_replaces"] = replaces
                    break
            continue

        # ---- Medical line (UNIFIED FINAL VERSION) ----

        DATE_PATTERN = re.compile(r"\d{2}/\d{2}/\d{4}")

        if DATE_PATTERN.search(p) or buffer:

            text = clean(p)

            # =========================
            # MODERN FORMAT (SAFE PATH)
            # =========================
            if " - " in text:

                horse, rest = text.split(" - ", 1)
                horse = clean(horse)

                parts = re.split(r",|&", rest)

                last_condition = ""

                for part in parts:
                    part = part.strip()

                    m = re.search(r"(.*?)\s*(\d{2}/\d{2}/\d{4})", part)
                    if not m:
                        continue

                    condition = m.group(1).strip()
                    date = m.group(2)

                    if not condition:
                        condition = last_condition
                    else:
                        last_condition = condition

                    if condition.startswith("("):
                        condition = "EIPH " + condition

                    medicals.append({
                        "meet_date": meet_date,
                        "race_no": current_race["race_no"],
                        "horse_name": horse,
                        "condition": clean(condition),
                        "date": parse_ddmmyyyy(date),
                    })

                continue

            # =========================
            # LEGACY FORMAT (FIXED)
            # =========================

            # ---- merge continuation ----
            if buffer:
                text = buffer + " " + text
                buffer = ""

            # ---- wait for full record ----
            if not DATE_PATTERN.search(text):
                buffer = text
                continue

            # ---- normalize spacing ----
            text = re.sub(r"\s+", " ", text)

            # ---- split into segments (comma-based) ----
            segments = re.split(r",|&", text)
            last_condition = ""

            for seg in segments:
                seg = seg.strip()
                if not seg:
                    continue

                # ---- detect horse (only if at start)
                horse_m = re.match(r"^([A-Z][A-Z '&\-\.]+)\s+(.*)", seg)
                if horse_m:
                    possible_horse = horse_m.group(1).strip()

                    # prevent EIPH as horse
                    if possible_horse not in ["EIPH", "BLEEDER"]:
                        current_horse = possible_horse
                        horse_tokens = current_horse.split()

                        if horse_tokens and horse_tokens[-1] in ["EIPH"]:
                            horse_tokens.pop()
                        
                        current_horse = " ".join(horse_tokens)
                        seg = horse_m.group(2).strip()

                # ---- extract condition + date ----
                m = re.search(r"(.*?)\s*(\d{2}/\d{2}/\d{4})", seg)

                if m and current_horse:
                    condition = m.group(1).strip()
                    # ---- FIX: reuse previous condition for "&" cases
                    if not condition:
                        condition = last_condition
                    else:
                        last_condition = condition
                    date = m.group(2)

                    # ---- attach continuation safely
                    if condition_buffer:
                        condition = condition_buffer + " " + condition
                        condition_buffer = ""

                    # ---- clean condition
                    condition = re.sub(r"[,&]+$", "", condition).strip()
                    condition = re.sub(r"^[,&]+", "", condition).strip()
                    condition = re.sub(r"\s+", " ", condition)

                    # ---- remove duplicate words
                    words = condition.split()
                    if len(words) >= 2 and words[0] == words[1]:
                        condition = " ".join(words[1:])

                    # ---- normalize EIPH
                    if condition.startswith("("):
                        condition = "EIPH " + condition

                    medicals.append({
                        "meet_date": meet_date,
                        "race_no": current_race["race_no"],
                        "horse_name": clean(current_horse),
                        "condition": clean(condition),
                        "date": parse_ddmmyyyy(date),
                    })

                else:
                    # ---- store dangling condition fragment
                    if current_horse and not DATE_PATTERN.search(seg):
                        condition_buffer = seg

            continue

        # ---- Runner line ----
        runners = _parse_runner_line(p)
        if runners and current_race:
            for rating, name, weight in runners:
                acceptances.append({
                    "meet_date":              meet_date,
                    "venue":                  venue,
                    "session":                session,
                    "race_no":                current_race["race_no"],
                    "race_name":              current_race["race_name"],
                    "conditions":             current_race["conditions"],
                    "distance":               current_race["distance"],
                    "time":                   current_race["time"],
                    "foreign_jockeys":        "Yes" if foreign_jockeys else "",
                    "horse_name":             name,
                    "weight":                 weight,
                    "rating":                 rating if rating != "-" else "",
                    "weight_update_handicap": wt_handicap,
                    "weight_update_acceptance": wt_acceptance,
                    "late_entry_replaces":    "",
                })

    return acceptances, medicals, pools


# ============================================================================
# SHOE/BITS/EQUIPMENT PARSER
# ============================================================================

# Equipment line format after clean():
# "ALLEZ L'ETOILE A - ADX SEMURG A - - HER CHARGE A - HB"
# "MANWAR A BFBS SWL SAVAGE PEACOCK A - -"
# Each horse block: NAME  shoe(A/S/P)  bit(-|code)  hood(-|code)
# We match: NAME then exactly: letter-group space letter-or-dash space letter-or-dash
_EQUIP_TOKEN = re.compile(
    r"([A-Z][A-Z0-9 '&/\.\-]+?)\s+"  # horse name (non-greedy)
    r"(A|S|P)\s+"                     # shoe type
    r"([-A-Z&]+)\s+"                  # bit
    r"([-A-Z&]+)"                     # hood/other
    r"(?=\s+[A-Z]|\s*$)"             # lookahead: next horse or end
)

def parse_equipment(shoe_paras, meet_date, race_map):
    """
    Parse the shoe/bits section.
    race_map: dict of race_name (lowercase) -> race_no
    """
    equipment  = []
    current_rno = ""

    for p in shoe_paras:
        # Race name header line e.g. "The Mid-Day Trophy"
        rno = _race_name_to_no(p, race_map)
        if rno:
            current_rno = rno
            continue

        for m in _EQUIP_TOKEN.finditer(p):
            equipment.append({
                "meet_date":  meet_date,
                "race_no":    current_rno,
                "horse_name": clean(m.group(1)),
                "shoe_type":  m.group(2),
                "bit":        m.group(3) if m.group(3) != "-" else "",
                "hood_other": m.group(4) if m.group(4) != "-" else "",
            })

    return equipment


def _race_name_to_no(text, race_map):
    """Try to match 'The Mid-Day Trophy' to a race number via race_map."""
    t = text.strip().lower()
    for name, rno in race_map.items():
        if name in t or t in name:
            return rno
    return ""


# ============================================================================
# BANDAGE PARSER
# ============================================================================

# "MANWAR - BF  SEMURG - BH  SAVAGE PEACOCK - BF"
_BANDAGE_TOKEN = re.compile(r"([A-Z][A-Z0-9 '&/\-\.]+?)\s+-\s+([A-Z]+)")

def parse_bandages(bandage_paras, meet_date):
    """Parse the bandages section."""
    bandages    = []
    current_rno = ""

    for p in bandage_paras:
        race_m = re.match(r"Race No:\s*(\d+)", p, re.I)
        if race_m:
            current_rno = race_m.group(1)
            continue
        if is_section_divider(p) or re.match(r"---+", p):
            continue
        for m in _BANDAGE_TOKEN.finditer(p):
            bandages.append({
                "meet_date":   meet_date,
                "race_no":     current_rno,
                "horse_name":  clean(m.group(1)),
                "bandage_type": m.group(2),
            })

    return bandages

# ============================================================================
# SWIMMING PARSER
# ============================================================================

def split_sections_raw(raw_paras):
    """
    Same section-splitting logic as split_sections() but operating on
    raw (non-collapsed) paragraphs for the swimming grid.
    Returns only the swimming section lines.
    """
    swimming = []
    in_swim  = False

    for p in raw_paras:
        norm = re.sub(r"[^a-z]", "", p.lower())

        # Start marker
        if "swimming" in norm or "swimmingdata" in norm:
            in_swim = True
            continue

        # Stop markers (next major section)
        if in_swim:
            if any(marker in norm for marker in [
                "treadmill", "highestrating", "trainerchange", "shoeandbits",
                "bandage",
            ]):
                break
            swimming.append(p)

    return swimming

# ============================================================================
# SWIMMING PARSER  (REPLACED)
# ============================================================================

def parse_swimming(swim_paras, meet_date):
    """
    Fixed-width grid parser for RWITC swimming data.

    swim_paras must be RAW paragraphs (only \xa0→space, \n→space substituted,
    multi-spaces preserved).  Pass the result of split_sections_raw().

    Algorithm:
      1. Parse Month header  → first_month, second_month
      2. Parse Day header    → col_positions [(day_str, char_pos), ...]
                               col_width (average spacing between columns)
      3. For each horse row:
         a. Find first digit → that is where the grid data begins
         b. Everything before the first digit = horse name
         c. Snap the first digit to the nearest header column → compute offset
         d. For every column: slice window around (hdr_pos + offset),
            extract digit if present
         e. Assign month: day >= first_day → first_month, else second_month
    """
    data        = []
    year        = meet_date.split("-")[0]

    col_positions  = []   # [(day_str, hdr_char_pos), ...]
    col_width      = 3    # computed from Day header spacing
    first_day      = None
    first_month    = None
    second_month   = None
    current_rno    = ""

    for raw in swim_paras:
        line = raw  # already has \xa0→space, \n→space; spaces preserved

        # ── Month header ──────────────────────────────────────────────────────
        if re.search(r"month\s*-+>", line, re.I):
            months = re.findall(r"\b(\d{1,2})\b", line)
            if len(months) >= 1:
                first_month = months[0].zfill(2)
            if len(months) >= 2:
                second_month = months[1].zfill(2)
            continue

        # ── Day header ────────────────────────────────────────────────────────
        if re.search(r"day\s*-+>", line, re.I):
            arrow_m   = re.search(r"day\s*-+>", line, re.I)
            arrow_end = arrow_m.end()

            col_positions = []
            for m in re.finditer(r"\b(\d{1,2})\b", line[arrow_end:]):
                col_positions.append((m.group(1), arrow_end + m.start()))

            if col_positions:
                first_day = int(col_positions[0][0])
                if len(col_positions) >= 2:
                    gaps = [
                        col_positions[i + 1][1] - col_positions[i][1]
                        for i in range(min(5, len(col_positions) - 1))
                    ]
                    col_width = round(sum(gaps) / len(gaps))
                else:
                    col_width = 3
            continue

        # ── Race number ───────────────────────────────────────────────────────
        race_m = re.search(r"race\s*no\s*:?\s*(\d+)", line, re.I)
        if race_m:
            current_rno = race_m.group(1)
            continue

        # ── Skip until headers parsed ─────────────────────────────────────────
        if not col_positions or not current_rno:
            continue

        # ── Skip note / blank lines ───────────────────────────────────────────
        if not re.search(r"\d", line):
            continue

        # ── Find first digit in line ──────────────────────────────────────────
        first_digit_m = re.search(r"\d", line)
        if not first_digit_m:
            continue
        first_digit_pos = first_digit_m.start()

        # ── Horse name = everything before first digit ────────────────────────
        horse = line[:first_digit_pos].strip()
        if not horse:
            continue

        # ── Snap first digit to nearest header column → compute row offset ────
        best_idx  = 0
        best_dist = abs(first_digit_pos - col_positions[0][1])
        for i, (_, hpos) in enumerate(col_positions):
            d = abs(first_digit_pos - hpos)
            if d < best_dist:
                best_dist = d
                best_idx  = i

        offset = first_digit_pos - col_positions[best_idx][1]

        # ── Read each column window ───────────────────────────────────────────
        for day_str, hdr_pos in col_positions:
            row_pos   = hdr_pos + offset
            win_start = max(0, row_pos - col_width // 2)
            win_end   = row_pos + col_width
            if win_start >= len(line):
                continue

            cell = line[win_start:win_end]
            m    = re.search(r"\d+", cell)
            if not m:
                continue

            rounds = m.group()

            # Month assignment
            d     = int(day_str)
            month = first_month if (d >= first_day) else (second_month or first_month)

            data.append({
                "meet_date":  meet_date,
                "race_no":    current_rno,
                "horse_name": horse,
                "date":       f"{year}-{month}-{str(d).zfill(2)}",
                "rounds":     rounds,
            })

    return data

# ============================================================================
# TREADMILL PARSER
# ============================================================================
def parse_treadmill(tm_paras, meet_date):

    data = []
    current_rno = ""
    year = meet_date.split("-")[0]

    buffer_name = ""

    for p in tm_paras:
        text = clean(p)

        # ---- Race ----
        race_m = re.search(r"Race\s*No\s*:?\s*(\d+)", text, re.I)
        if race_m:
            current_rno = race_m.group(1)
            buffer_name = ""
            continue

        if not current_rno:
            continue

        if not text.strip():
            continue

        # ---- Handle broken names ----
        if re.match(r"^[A-Z][A-Z\s']+$", text) and not re.search(r"\d", text):
            buffer_name += " " + text
            continue

        full = (buffer_name + " " + text).strip()
        buffer_name = ""

        # ---- Extract horse ----
        tokens = full.split()

        horse_tokens = []
        for t in tokens:
            if re.match(r"^[A-Z0-9 '&\-\.]+$", t):
                horse_tokens.append(t)
            else:
                break

        if not horse_tokens:
            continue

        horse = " ".join(horse_tokens)

        rest = full[len(horse):].strip()

        # ---- Extract date ----
        date_m = re.search(r"(\d{2}/\d{2})", rest)
        if not date_m:
            continue

        date = f"{date_m.group(1)}/{year}"

        # ---- Extract segments ----
        segments = re.split(r"\|", rest)

        seg_id = 1
        for seg in segments[1:]:  # skip left side (date/inc)
            nums = re.findall(r"\d+\.?\d*", seg)

            if len(nums) >= 2:
                speed = nums[0]
                dist = nums[1]

                if float(speed) == 0 and float(dist) == 0:
                    continue

                data.append({
                    "meet_date": meet_date,
                    "race_no": current_rno,
                    "horse_name": clean(horse),
                    "date": parse_ddmmyyyy(date),
                    "segment": seg_id,
                    "speed": speed,
                    "distance": dist,
                })

                seg_id += 1

    return data

# ============================================================================
# HIGHEST RATINGS PARSER
# ============================================================================

# "ALLEZ L'ETOILE 38 on 25/04/2025  SEMURG 39 on 24/10/2025  HER CHARGE 30 on 21/01/2026"
_HR_TOKEN = re.compile(
    r"([A-Z][A-Z0-9 '&/\-\.]+?)\s+(\d+)\s+on\s+(\d{2}/\d{2}/\d{4})"
)

def parse_highest_ratings(hr_paras, meet_date):
    ratings     = []
    current_rno = ""

    for p in hr_paras:
        race_m = re.match(r"Race No:\s*(\d+)", p, re.I)
        if race_m:
            current_rno = race_m.group(1)
            continue
        if is_section_divider(p) or re.match(r"---+", p):
            continue
        for m in _HR_TOKEN.finditer(p):
            ratings.append({
                "meet_date":     meet_date,
                "race_no":       current_rno,
                "horse_name":    clean(m.group(1)),
                "highest_rating": m.group(2),
                "achieved_date": parse_ddmmyyyy(m.group(3)),
            })

    return ratings


# ============================================================================
# TRAINER CHANGES PARSER
# ============================================================================

# "MARIUS  New Trainer: Aman Altaf Hussain  Old Trainer : P. Shroff"
# "Last run date : 22/02/2026  Dt took charge : 11/03/2026"

def parse_trainer_changes(tc_paras, meet_date):

    changes = []
    current_rno = ""
    current_entry = None

    pending_last_run = ""
    pending_took_charge = ""

    for p in tc_paras:

        text = re.sub(r"\s+", " ", p.strip())

        # ---- Race number ----
        race_m = re.search(r"Race No:\s*(\d+)", text, re.I)
        if race_m:
            current_rno = race_m.group(1)
            continue

        # ---- Extract dates FIRST (important) ----
        # ---- Dates (robust for 2018 + 2026) ----

        d1 = re.search(
            r"(?:Last\s*run\s*date|Lrundate)\s*:?\s*(\d{2}/\d{2}/\d{4})",
            text,
            re.I
        )

        
        if d1:
            pending_last_run = parse_ddmmyyyy(d1.group(1))

        d2 = re.search(
            r"(?:Dt\s*took\s*charge|Dt\s*tookchg)\s*:?\s*(\d{2}/\d{2}/\d{4})",
            text,
            re.I
        )
        
        if d2:
            pending_took_charge = parse_ddmmyyyy(d2.group(1))

        # ---- Horse line ----
        horse_m = re.match(
            r"^([A-Z][A-Z0-9 '&/\-\.]+?)\s+(?:New Trainer|New trn)\s*:\s*(.+?)\s+(?:Old Trainer|Old trn)\s*:\s*(.+)",
            text,
            re.I
        )

        if horse_m:
            # Save previous
            if current_entry:
                changes.append(current_entry)

            # Assign any pending dates
            current_entry = {
                "meet_date": meet_date,
                "race_no": current_rno,
                "horse_name": clean(horse_m.group(1)),
                "new_trainer": clean(horse_m.group(2)),
                "old_trainer": clean(horse_m.group(3)),
                "last_run_date": pending_last_run,
                "took_charge_date": pending_took_charge,
            }

            # Reset buffer after use
            pending_last_run = ""
            pending_took_charge = ""

            continue

        # ---- If dates come AFTER horse ----
        if current_entry:
            if d1:
                current_entry["last_run_date"] = pending_last_run
            if d2:
                current_entry["took_charge_date"] = pending_took_charge

    # Append last
    if current_entry:
        changes.append(current_entry)

    return changes


# ============================================================================
# FILE ORCHESTRATOR
# ============================================================================

def parse_doc_file(filepath):
    log.info(f"Parsing: {filepath}")
    try:
        with open(filepath, "r", encoding=INPUT_ENCODING, errors="replace") as f:
            html = f.read()
    except Exception as e:
        log.error(f"Cannot read {filepath}: {e}")
        return None

    soup  = BeautifulSoup(html, "html.parser")
    paras = get_paragraphs(soup)
    
    # ── Raw paragraphs for swimming (preserves fixed-width spacing) ───────────
    raw_paras    = get_raw_paragraphs(soup)
    swim_raw     = split_sections_raw(raw_paras)

    if not paras:
        log.warning("No MsoPlainText paragraphs found — wrong format?")
        return None

    secs = split_sections(paras)

    # rebuild swimming section with raw text
    raw_paras = [p.get_text().replace("\xa0", " ") for p in soup.find_all("p")]
    clean_paras = [clean(p) for p in raw_paras]

    swim_raw = []
    swim_clean = secs.get("swimming", [])

    i = 0
    for raw, clean_p in zip(raw_paras, clean_paras):
        if i < len(swim_clean) and clean_p == swim_clean[i]:
            swim_raw.append(raw)
            i += 1

    print("SWIM RAW SAMPLE:")
    for i, p in enumerate(swim_raw[:10]):
        print(i, repr(p))

    # print("SECTIONS:", secs.keys())
    # print("SWIMMING LINES:", len(secs.get("swimming", [])))
    # for i, x in enumerate(secs.get("swimming", [])[:10]):
    #     print(i, repr(x))

    # --- Meeting info ---
    info = parse_meeting_info(secs.get("header", []))
    meet_date = info["meet_date"]
    venue     = info["venue"]
    session   = info["session"]
    log.info(f"  Date={meet_date}  Venue={venue}  Session={session}")

    # --- Races / acceptances (everything before shoe/bits) ---
    acceptances, medicals, pools = parse_races(
        secs.get("header", []), meet_date, venue, session
    )
    log.info(f"  Acceptances: {len(acceptances)}  Medical: {len(medicals)}  Pools: {len(pools)}")

    # Build race_name->race_no lookup for equipment section
    race_map = {}
    for row in acceptances:
        key = row["race_name"].lower().strip()
        race_map[key] = row["race_no"]

    # --- Equipment (shoe/bits) ---
    equipment = parse_equipment(secs.get("shoe_bits", []), meet_date, race_map)
    log.info(f"  Equipment: {len(equipment)}")

    # --- Bandages ---
    bandages = parse_bandages(secs.get("bandages", []), meet_date)
    log.info(f"  Bandages: {len(bandages)}")

    # --- Swimming ---
    swimming = parse_swimming(swim_raw, meet_date)
    log.info(f"  Swimming: {len(swimming)}")

    # --- Treadmill ---
    treadmill = parse_treadmill(secs.get("treadmill", []), meet_date)
    log.info(f"  Treadmill: {len(treadmill)}")

    # --- Highest ratings ---
    highest_ratings = parse_highest_ratings(secs.get("highest_ratings", []), meet_date)
    log.info(f"  Highest ratings: {len(highest_ratings)}")

    # --- Trainer changes ---
    log.info(f"Trainer section lines: {len(secs.get('trainer_changes', []))}")
    trainer_changes = parse_trainer_changes(secs.get("trainer_changes", []), meet_date)
    log.info(f"  Trainer changes: {len(trainer_changes)}")

    return {
        "acceptances":     acceptances,
        "medical":         medicals,
        "equipment":       equipment,
        "bandages":        bandages,
        "highest_ratings": highest_ratings,
        "trainer_changes": trainer_changes,
        "pools":           pools,
        "swimming":        swimming,
        "treadmill":       treadmill,
    }


# ============================================================================
# CSV OUTPUT
# ============================================================================

SCHEMA = {
    "acceptances":     ACCEPTANCE_COLS,
    "medical":         MEDICAL_COLS,
    "equipment":       EQUIPMENT_COLS,
    "bandages":        BANDAGE_COLS,
    "highest_ratings": HIGHEST_RATING_COLS,
    "trainer_changes": TRAINER_CHANGE_COLS,
    "pools":           POOL_COLS,
    "swimming":        SWIMMING_COLS,
    "treadmill":       TREADMILL_COLS,
}


def write_csvs(all_data):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for key, cols in SCHEMA.items():
        rows = all_data.get(key, [])
        path = os.path.join(OUTPUT_DIR, FILES[key])
        need_header = (WRITE_MODE == "w") or not os.path.exists(path) or os.path.getsize(path) == 0
        mode = "w" if WRITE_MODE == "w" else "a"
        with open(path, mode, newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            if need_header:
                w.writeheader()
            w.writerows(rows)
        log.info(f"  {len(rows):>4d} rows -> {path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    if os.path.isfile(INPUT_PATH):
        files = [INPUT_PATH]
    elif os.path.isdir(INPUT_PATH):
        files = sorted(
            glob.glob(os.path.join(INPUT_PATH, "*.htm"))
            + glob.glob(os.path.join(INPUT_PATH, "*.html"))
        )
    else:
        log.error(f"INPUT_PATH not found: {INPUT_PATH}")
        sys.exit(1)

    if not files:
        log.error(f"No .htm/.html files found in: {INPUT_PATH}")
        sys.exit(1)

    log.info(f"Found {len(files)} file(s)")

    combined = {k: [] for k in SCHEMA}
    for fp in files:
        result = parse_doc_file(fp)
        if result:
            for k in SCHEMA:
                combined[k].extend(result.get(k, []))

    write_csvs(combined)
    log.info(f"\nDone — {len(combined['acceptances'])} acceptance rows total")


if __name__ == "__main__":
    main()