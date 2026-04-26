#!/usr/bin/env python3

import os, re, csv, sys, glob, logging
from bs4 import BeautifulSoup

# =============================================================================
# CONFIG
# =============================================================================

#INPUT_PATH   = "./raw_doc/race_day_report/2014-08-03.htm"
#INPUT_PATH   = "./raw_doc/race_day_report/2026-03-30.htm"
INPUT_PATH   = "./raw_doc/race_day_report"
OUTPUT_DIR   = "./raw/race_day_report_test"
WRITE_MODE   = "w"
INPUT_ENCODING = "utf-8"
LOG_LEVEL    = logging.INFO

FILES = {
    "incidents": "race_incidents.csv",
    "conditions": "horse_condition.csv",
    "actions": "jockey_trainer_actions.csv",
    "jockey_changes": "jockey_changes.csv",
    "summary_penalties": "summary_penalties.csv",
    "summary_horse_actions": "summary_horse_actions.csv",
    "summary_pacifiers": "summary_pacifiers.csv",
}

INCIDENT_COLS = [
    "date", "venue", "race_no", "horse",
    "incident_type", "severity", "position_phase"
]

CONDITION_COLS = [
    "date", "venue", "race_no", "horse",
    "condition_type", "action", "severity"
]

JOCKEY_CHANGE_COLS = [
    "meet_date", "venue", "race_no",
    "horse", "original_jockey",
    "replacement_jockey", "reason"
]

SUMMARY_PENALTY_COLS = ["date","venue","race_no","person","role","horse","action_type","penalty"]
SUMMARY_HORSE_COLS = ["date","venue","race_no","horse","action","condition"]
SUMMARY_PACIFIER_COLS = ["date","venue","race_no","horse"]

logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# =============================================================================
# HELPERS
# =============================================================================

def clean(t):
    if not t: return ""
    return re.sub(r"\s+", " ", t.replace("\xa0", " ")).strip()

def parse_date(text):
    m = re.search(
        r"(\d{1,2})\s*(?:ST|ND|RD|TH)?\s*,?\s*"
        r"(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|"
        r"AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)"
        r"\s*,?\s*(\d{4})",
        text,
        re.I
    )
    if not m:
        return ""
    
    day = int(m.group(1))
    month = ["january","february","march","april","may","june",
             "july","august","september","october","november","december"
            ].index(m.group(2).lower()) + 1
    year = int(m.group(3))
    
    return f"{year:04d}-{month:02d}-{day:02d}"

def detect_phase(line):
    l = line.lower()
    if "start" in l: return "START"
    if "final" in l or "last" in l or "400" in l or "200" in l: return "FINAL"
    return "MID"

def extract_race_no(text):
    """Extract race number from strings like 'R.1(181)', 'R 4 (29)', '8(33)', '1(181)'"""
    m = re.search(r"R\.?\s*(\d+)\s*\(", text, re.I)
    if m:
        return m.group(1)
    m = re.match(r"(\d+)\s*\(", text.strip())
    if m:
        return m.group(1)
    return ""

def extract_penalty_amount(text):
    """Extract fine amount or suspension duration."""
    m = re.search(r"Rs\.?\s*[\d,]+/-?", text, re.I)
    if m:
        return m.group(0)
    m = re.search(r"(\d+)\s+race\s+days?", text, re.I)
    if m:
        return m.group(0)
    return ""

def split_horse_entries(text):
    """Split a cell that contains multiple horse entries on separate lines."""
    return [l.strip() for l in re.split(r"\n|(?<=[a-z])\s{2,}", text) if l.strip()]

# ---------------- HORSE EXTRACTION (FIXED) ----------------

_NAME_BLACKLIST  = {
        "AFTER","THE","BOTH","AND","WERE","FOUND","CLUB",
        "HAS","HAVE","HAD","BEEN","TO","BY","OF"
    }

def extract_horses(line):
    candidates = re.findall(r"\b[A-Z][A-Z '\-\u2019]{2,}\b", line)

    horses = []
    for c in candidates:
        c = c.strip()
        if c not in _NAME_BLACKLIST  and len(c.split()) >= 2:
            horses.append(c)

    return horses

def extract_first_horse(line):
    horses = extract_horses(line)
    return horses[0] if horses else ""

# ---------------- INFERENCE ----------------

def infer_condition_type(line):
    l = line.lower()
    if "sore" in l: return "SORE"
    if "blood vessel" in l: return "BLEEDING"
    if "fractious" in l: return "FRACTIOUS"
    if "injury" in l: return "INJURY"
    if "collapsed" in l: return "COLLAPSE"
    return "INJURY"

def infer_action(line):
    l = line.lower()
    if "passed fit" in l: return "FIT_REQUIRED"
    if "not permitted" in l or "ban" in l: return "BANNED"
    if "remedial" in l: return "REMEDIAL"
    if "gate practice" in l: return "GATE_PRACTICE"
    return "OBSERVED"

def infer_severity(line):
    l = line.lower()
    if "collapsed" in l or "fracture" in l: return "HIGH"
    if "sore" in l or "injury" in l: return "MEDIUM"
    return "LOW"

# =============================================================================
# INCIDENT PARSER — subject-aware
# =============================================================================

# Strict horse name pattern — only matches ALL-CAPS words (with apostrophe/hyphen allowed)
# Matches 1 or more uppercase words, each word being [A-Z][A-Z''-]*
_H = r"([A-Z][A-Z'''\-]*(?:\s+[A-Z][A-Z'''\-]*)*)"


def _is_valid_horse_capture(candidate):
    """Final validation after regex capture and cleanup."""
    candidate = candidate.strip()
    if not candidate:
        return False
    # Must be all uppercase words
    words = candidate.split()
    if not words:
        return False
    for w in words:
        # Each word: starts with A-Z, rest A-Z or '-' or apostrophe
        if not re.match(r"^[A-Z][A-Z''\-]*$", w):
            return False
    # Reject pure stopword captures
    if all(w in _NAME_BLACKLIST for w in words):
        return False
    # Reject obvious junk phrases (3+ words that are all stopwords or fragments)
    JUNK_STARTS = {"SHIFTED", "REPORTED", "AS", "CAUSED", "DUE", "AFTER",
                   "BEFORE", "THEREBY", "PASSING", "APPROACHING", "NEAR",
                   "SOON", "DURING", "BETWEEN", "DESPITE", "PROCESS"}
    if words[0] in JUNK_STARTS:
        return False
    return True


def _extract_subject_horse(match_group):
    candidate = match_group.strip()
    # Normalize smart apostrophes
    candidate = candidate.replace("\u2019", "'").replace("\u2018", "'")
    # Remove trailing jockey bracket e.g. "(Darren Williams, dr.2)"
    candidate = re.sub(r"\s*\([^)]*\)", "", candidate).strip()
    # Strip trailing lowercase noise (e.g. "who", "and", "the")
    candidate = re.sub(r"\s+[a-z].*$", "", candidate).strip()
    # Strip trailing punctuation
    candidate = candidate.rstrip(",.;:-– ")
    if _is_valid_horse_capture(candidate):
        return candidate
    return None

# Patterns that mark the SUBJECT (the horse that suffered the incident)
# Each tuple: (incident_type, list of trigger phrases that precede the subject)
# Replace INCIDENT_SUBJECT_PATTERNS — fix regex char classes and add missing patterns
INCIDENT_SUBJECT_PATTERNS = [
    # ── AWKWARD_JUMP ──────────────────────────────────────────────────────────
    ("AWKWARD_JUMP", [
        # "HORSE jumped awkwardly" — horse is subject
        r"([\w][A-Z'''\-\s]*?)\s*(?:\([^)]*\)\s*)?jumped\s+(?:out\s+)?awkwardly",
        # "HORSE took an awkward jump"
        r"([\w][A-Z'''\-\s]*?)\s*(?:\([^)]*\)\s*)?took\s+an\s+awkward\s+jump",
    ]),

    # ── SLOW_START ────────────────────────────────────────────────────────────
    ("SLOW_START", [
        r"([\w][A-Z'''\-\s]*?)\s*(?:\([^)]*\)\s*)?jumped\s+out\s+slow",
        r"([\w][A-Z'''\-\s]*?)\s*(?:\([^)]*\)\s*)?lost\s+about\s+\d+\s+lengths?\s+at\s+the\s+start",
        r"([\w][A-Z'''\-\s]*?)\s*(?:\([^)]*\)\s*)?(?:was\s+)?reluctant\s+to\s+(?:be\s+stalled|reach\s+the\s+starting)",
    ]),

    # ── CHECKED ───────────────────────────────────────────────────────────────
    ("CHECKED", [
        # "causing HORSE to be checked/check/change course"
        r"causing\s+([\w][A-Z'''\-\s]*?)\s*(?:\([^)]*\)\s*)?to\s+(?:be\s+)?(?:check|change\s+course)",
        # "HORSE was checked/steadied/had to check"
        r"([\w][A-Z'''\-\s]*?)\s*(?:\([^)]*\)\s*)?(?:was|were|had\s+to)\s+(?:checked?|steadied)",
        # "HORSE who was checked/steadied" (relative clause)
        r"([\w][A-Z'''\-\s]*?)\s*(?:\([^)]*\)\s*)?who\s+was\s+(?:checked?|steadied)",
        # "HORSE had to be checked"
        r"([\w][A-Z'''\-\s]*?)\s*(?:\([^)]*\)\s*)?had\s+to\s+be\s+checked",
        # "due to the incident HORSE was inconvenienced and had to be checked"
        r"([\w][A-Z'''\-\s]*?)\s+was\s+inconvenienced\s+and\s+had\s+to\s+be\s+checked",
    ]),

    # ── BUMPED ────────────────────────────────────────────────────────────────
    ("BUMPED", [
        # "bumping HORSE" / "bumped HORSE"
        r"bumping\s+([\w][A-Z'''\-\s]*?)(?=\s*(?:\(|,|\.|who|causing|thereby|\Z))",
        r"bumped\s+([\w][A-Z'''\-\s]*?)(?=\s*(?:\(|,|\.|who|causing|thereby|\Z))",
        # "went on to HORSE" — contact chain
        r"went\s+on\s+to\s+([\w][A-Z'''\-\s]*?)(?=\s*(?:\(|,|\.|who|\Z))",
        # "made contact with HORSE"
        r"made\s+contact\s+with\s+([\w][A-Z'''\-\s]*?)(?=\s*(?:\(|,|\.|who|\Z))",
        # "jostled with HORSE"
        r"jostled\s+with\s+([\w][A-Z'''\-\s]*?)(?=\s*(?:\(|,|\.|who|\Z))",
        # "carried HORSE outwards/inwards"
        r"carried\s+([\w][A-Z'''\-\s]*?)\s*(?:\([^)]*\)\s*)?(?:out|in)wards?",
        # "jumped outwards onto HORSE"
        r"jumped\s+(?:out|in)wards?\s+(?:and\s+)?(?:went\s+)?onto\s+([\w][A-Z'''\-\s]*?)(?=\s*(?:\(|,|\.|who|\Z))",
        # "shifted out and went across HORSE" — victim
        r"went\s+across\s+([\w][A-Z'''\-\s]*?)(?=\s*(?:\(|,|\.|who|causing|\Z))",
    ]),

    # ── INTERFERENCE ──────────────────────────────────────────────────────────
    ("INTERFERENCE", [
        r"([\w][A-Z'''\-\s]*?)\s*(?:\([^)]*\)\s*)?(?:was|were)\s+inconvenienced",
        r"inconveniencing\s+([\w][A-Z'''\-\s]*?)(?=\s*(?:\(|,|\.|$))",
        r"tightening\s+([\w][A-Z'''\-\s]*?)(?=\s*(?:\(|,|\.|$))",
        r"tightened\s+([\w][A-Z'''\-\s]*?)(?=\s*(?:\(|,|\.|$))",
        r"([\w][A-Z'''\-\s]*?)\s*(?:\([^)]*\)\s*)?was\s+tightened",
        r"unbalancing\s+([\w][A-Z'''\-\s]*?)(?=\s*(?:\(|,|\.|$))",
        r"boring\s+([\w][A-Z'''\-\s]*?)(?=\s*(?:\(|,|\.|$))",
        r"([\w][A-Z'''\-\s]*?)\s*(?:\([^)]*\)\s*)?was\s+unbalanced",
        r"forcing\s+([\w][A-Z'''\-\s]*?)\s*(?:\([^)]*\)\s*)?(?:out|in)wards?",
    ]),

    # ── DRIFTING ──────────────────────────────────────────────────────────────
    ("DRIFTING", [
        r"([\w][A-Z'''\-\s]*?)\s*(?:\([^)]*\)\s*)?(?:was\s+observed\s+to\s+have\s+)?drifted",
        r"([\w][A-Z'''\-\s]*?)\s*(?:\([^)]*\)\s*)?drifting",
    ]),

    # ── HUNG_IN ───────────────────────────────────────────────────────────────
    ("HUNG_IN", [
        r"([\w][A-Z'''\-\s]*?)\s*(?:\([^)]*\)\s*)?(?:hung[\-\s]in|lugged[\-\s]in)",
        r"([\w][A-Z'''\-\s]*?)\s*(?:\([^)]*\)\s*)?was\s+inclined\s+inwards?",
    ]),

    # ── HUNG_OUT ──────────────────────────────────────────────────────────────
    ("HUNG_OUT", [
        r"([\w][A-Z'''\-\s]*?)\s*(?:\([^)]*\)\s*)?(?:hung[\-\s]out|lugged[\-\s]out)",
        r"([\w][A-Z'''\-\s]*?)\s*(?:\([^)]*\)\s*)?was\s+inclined\s+outwards?",
    ]),

    # ── NO_CLEAR_RUN ──────────────────────────────────────────────────────────
    ("NO_CLEAR_RUN", [
        r"([\w][A-Z'''\-\s]*?)\s*(?:\([^)]*\)\s*)?had\s+no\s+clear\s+run",
    ]),
]

def parse_incidents(lines, meta):
    rows = []
    seen = set()
    last_horse = ""  # track most recently named horse for "the latter" resolution

    for line in lines:
        if not meta["race_no"]:
            continue

        # Resolve "the latter" to last horse seen in this paragraph
        working_line = line
        if "the latter" in line.lower() and last_horse:
            working_line = re.sub(r"\bthe\s+latter\b", last_horse, line, flags=re.I)

        # Update last_horse from any all-caps name visible in this line
        caps_names = re.findall(
            r"\b([A-Z][A-Z'''\-]*(?:\s+[A-Z][A-Z'''\-]*)*)\b", line
        )
        for c in caps_names:
            if _is_valid_horse_capture(c):
                last_horse = c

        for itype, patterns in INCIDENT_SUBJECT_PATTERNS:
            for pattern in patterns:
                for m in re.finditer(pattern, working_line):
                    horse = _extract_subject_horse(m.group(1))
                    if not horse:
                        continue
                    key = (meta["race_no"], horse, itype)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append({
                        "date":           meta["date"],
                        "venue":          meta["venue"],
                        "race_no":        meta["race_no"],
                        "horse":          horse,
                        "incident_type":  itype,
                        "severity":       "MEDIUM",
                        "position_phase": detect_phase(working_line),
                    })

    return rows


# =============================================================================
# HORSE CONDITION PARSER
# =============================================================================

# Sentinel phrases that mark end of meaningful race content
_STOP_PHRASES = [
    "urine sample", "prohibited substance", "anabolic steroid",
    "sex hormone", "samples results", "additional report",
]

def extract_all_horses(text):
    text = text.replace("’", "'")
    return re.findall(r"\b[A-Z][A-Z '\-]{2,}\b", text)

def _is_stop_line(line):
    l = line.lower()
    return any(phrase in l for phrase in _STOP_PHRASES)


_H = r"([A-Z][A-Z'''\-]*(?:\s+[A-Z][A-Z'''\-]*)*)"

CONDITION_PATTERNS = [
    ("SORE", "FIT_REQUIRED", "MEDIUM", [
        rf"{_H}\s*(?:\([^)]*\)\s*)?(?:was\s+)?(?:examined\s+and\s+)?found\s+to\s+be\s+sore",
        rf"{_H}\s*(?:\([^)]*\)\s*)?returned\s+sore",
        rf"{_H}\s*\(Sore\)",                          # summary table
    ]),
    ("BLEEDING", "FIT_REQUIRED", "MEDIUM", [
        rf"{_H}\s*(?:\([^)]*\)\s*)?(?:was\s+observed\s+to\s+have\s+)?broken\s+a\s+blood\s+vessel",
        rf"{_H}\s*\(Blood\s+Vessel\)",                # summary table
    ]),
    ("FRACTIOUS", "GATE_PRACTICE", "LOW", [
        rf"{_H}\s*(?:\([^)]*\)\s*)?was\s+fractious",
        # "HORSE were restless in the stall" — multi-horse handled by findall
        rf"{_H}\s*(?:\([^)]*\)\s*)?(?:was|were)\s+restless\s+in\s+the\s+stall",
        # "HORSE was reluctant to be stalled / reach the starting"
        rf"{_H}\s*(?:\([^)]*\)\s*)?was\s+reluctant\s+to\s+(?:be\s+stalled|reach\s+the\s+starting)",
    ]),
    ("REMEDIAL", "REMEDIAL", "LOW", [
        # "HORSE drifted" / "was observed to have drifted"
        rf"{_H}\s*(?:\([^)]*\)\s*)?(?:was\s+observed\s+to\s+have\s+)?drifted",
        # "HORSE hung-in/out / lugged-in/out"
        rf"{_H}\s*(?:\([^)]*\)\s*)?(?:hung[\-\s](?:in|out)|lugged[\-\s](?:in|out))",
        # "HORSE was inclined outwards/inwards"
        rf"{_H}\s*(?:\([^)]*\)\s*)?was\s+inclined\s+(?:out|in)wards?",
        # "HORSE trailed the field"
        rf"{_H}\s*(?:\([^)]*\)\s*)?(?:was\s+observed\s+to\s+have\s+)?trailed\s+the\s+field",
        # "trainer asked to take remedial measures" — horse name precedes "reported that his mount"
        # Catches: "Jockey X (HORSE) reported that his mount hung/drifted..."
        # Also catches summary table: "HORSE – remedial measures"
        rf"{_H}\s*[-–]\s*remedial\s+measures",
    ]),
    # FIT_REQUIRED fallback — suppressed if specific condition already found
    ("INJURY", "FIT_REQUIRED", "MEDIUM", [
        rf"{_H}\s*(?:\([^)]*\)\s*)?(?:has\s+to|must|have\s+to)\s+be\s+passed\s+fit",
        rf"{_H}\s*(?:\([^)]*\)\s*)?to\s+be\s+passed\s+fit\s+(?:by\s+the\s+)?(?:Club|VO|V\.O)",
    ]),
]

_SPECIFIC_CONDITIONS = {"SORE", "BLEEDING", "FRACTIOUS", "REMEDIAL"}


def parse_conditions(lines, meta):
    rows = []
    seen = set()
    horse_conditions: dict = {}

    def _add(ctype, action, severity, horse):
        key = (meta["race_no"], horse, ctype)
        if key in seen:
            return
        seen.add(key)

        hkey = (meta["race_no"], horse)

        if ctype == "INJURY":
            existing = horse_conditions.get(hkey, set())
            if existing & _SPECIFIC_CONDITIONS:
                return

        horse_conditions.setdefault(hkey, set()).add(ctype)
        rows.append({
            "date":           meta["date"],
            "venue":          meta["venue"],
            "race_no":        meta["race_no"],
            "horse":          horse,
            "condition_type": ctype,
            "action":         action,
            "severity":       severity,
        })

    for line in lines:
        # Stop processing at urine/sample/additional report sections
        if _is_stop_line(line):
            break

        for ctype, action, severity, patterns in CONDITION_PATTERNS:
            for pattern in patterns:
                for m in re.finditer(pattern, line, re.I):
                    horse = _extract_subject_horse(m.group(1))
                    if not horse:
                        continue
                    _add(ctype, action, severity, horse)

    return rows

# =============================================================================
# JOCKEY CHANGE TABLE PARSER (FIXED CORE)
# =============================================================================

def find_change_of_jockey_table(soup):
    for table in soup.find_all("table"):
        txt = table.get_text(" ").upper()

        if (
            "RACE" in txt and
            "HORSE" in txt and
            "JOCKEY" in txt and
            "REPLACED" in txt
        ):
            return table

    return None

def parse_jockey_changes(soup, meet_date, venue):
    """
    Handles two layouts:
      - Standalone table preceded by a <p> containing 'CHANGE OF JOCKEY'
      - Table whose first row header contains 'CHANGE OF JOCKEY'
    """
    rows = []

    # Find the trigger paragraph first; the table immediately follows it.
    trigger_table = None

    trigger_table = find_change_of_jockey_table(soup)

    if not trigger_table:
        log.warning("No CHANGE OF JOCKEY table found.")
        return rows

    trs = trigger_table.find_all("tr")
    if not trs:
        return rows

    # Detect header row and skip it
    header_text = trs[0].get_text().upper()
    data_rows = trs[1:] if "RACE" in header_text or "HORSE" in header_text else trs

    for tr in data_rows:
        tds = [clean(td.get_text()) for td in tr.find_all("td")]
        if len(tds) < 4:
            continue

        race_raw, horse, original, replacement = tds[0], tds[1], tds[2], tds[3]
        reason = tds[4] if len(tds) >= 5 else ""

        race_no = extract_race_no(race_raw)
        if not race_no:
            # plain "8(33)" or "1(181)" format
            m = re.match(r"(\d+)", race_raw.strip())
            race_no = m.group(1) if m else ""

        if not horse:
            continue

        rows.append({
            "meet_date": meet_date,
            "venue": venue,
            "race_no": race_no,
            "horse": horse.upper(),
            "original_jockey": original,
            "replacement_jockey": replacement,
            "reason": reason,
        })

    log.info(f"jockey_changes={len(rows)}")
    return rows

# =============================================================================
# SUMMARY PENALTIES
# =============================================================================

def _parse_penalty_line(line, meet_date, venue, current_race):
    """
    Returns a list of penalty row dicts from a single text line.
    Handles:
      - Single: "Joc. Mukesh Kumar (MA CHERIE) – fined Rs.10,000/-"
      - Multi:  "Jockeys A.Sandesh & T.S.Jodha - fined Rs 1,000/- each"
      - Trainer: "Trainer Nina M.Lalvani - fined Rs 1,000/-"
    """
    line = line.replace("–", "-").replace("—", "-").strip()
    if not line:
        return []

    # Determine action type first
    if re.search(r"\bsuspend", line, re.I):
        action_type = "SUSPENSION"
    elif re.search(r"\bfine[d]?\b", line, re.I) or re.search(r"Rs\.?\s*[\d,]+", line, re.I):
        action_type = "FINE"
    else:
        return []

    penalty = extract_penalty_amount(line)

    # --- Split multiple people on one line ---
    # Strip leading role prefix to isolate the names+horse portion
    # e.g. "Jockeys A.Sandesh & T.S.Jodha - fined..."
    #      "Joc. Mukesh Kumar (MA CHERIE) – fined..."
    #      "Trainer Nina M.Lalvani - fined..."

    # Everything before the first " - fined" or " - suspended" is the people block
    people_block_match = re.split(r"\s+-\s+(?:fined|suspended)", line, maxsplit=1, flags=re.I)
    people_block = people_block_match[0] if people_block_match else line

    # Strip role prefixes from the start
    people_block = re.sub(
        r"^(?:Jockeys?\.?|Joc\.?|App\.?|Apprentices?|Trainers?)\s+",
        "", people_block, flags=re.I
    ).strip()

    # Split on & or comma+space (but not commas inside horse names which are rare)
    raw_names = re.split(r"\s*&\s*|\s*,\s*(?=[A-Z])", people_block)

    rows = []
    for raw in raw_names:
        raw = raw.strip()
        if not raw:
            continue

        # Horse name in parentheses
        horse = ""
        hm = re.search(r"\(([A-Z][A-Z '\-]+)\)", raw)
        if hm:
            horse = hm.group(1).strip()
            raw = raw[:hm.start()].strip()

        # Determine role from original line context or name prefix
        if re.search(r"\btrainer\b", line, re.I):
            role = "TRAINER"
        else:
            role = "JOCKEY"

        person = raw.strip().rstrip("-– ")

        if not person:
            continue

        rows.append({
            "date": meet_date,
            "venue": venue,
            "race_no": current_race,
            "person": person,
            "role": role,
            "horse": horse,
            "action_type": action_type,
            "penalty": penalty,
        })

    return rows


def parse_summary_penalties(soup, meet_date, venue):
    rows = []

    for table in soup.find_all("table"):
        txt_upper = table.get_text(" ").upper()
        if "SUSPENSION" not in txt_upper and "FINE" not in txt_upper:
            continue

        current_race = ""

        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if not tds:
                continue

            cell_texts_raw = [td for td in tds]  # keep as tags for multi-<p> support
            row_upper = " ".join(clean(td.get_text()) for td in tds).upper()

            if "ACTION ON HORSE" in row_upper:
                break

            if "SUSPENSION" in row_upper and len(tds) <= 2:
                continue

            # Identify race cell and detail cell (as BeautifulSoup tags)
            if len(tds) >= 3:
                race_td, detail_td = tds[1], tds[2]
            elif len(tds) == 2:
                race_td, detail_td = tds[0], tds[1]
            else:
                continue

            rno = extract_race_no(clean(race_td.get_text()))
            if rno:
                current_race = rno

            if not current_race:
                continue

            # *** Iterate each <p> inside the detail cell separately ***
            p_tags = detail_td.find_all("p")
            if p_tags:
                lines = [clean(p.get_text()) for p in p_tags if clean(p.get_text())]
            else:
                lines = [clean(detail_td.get_text())]

            for line in lines:
                rows.extend(_parse_penalty_line(line, meet_date, venue, current_race))

    log.info(f"summary_penalties={len(rows)}")
    return rows

# =============================================================================
# SUMMARY HORSE ACTIONS
# =============================================================================

def _parse_horse_action_line(line, meet_date, venue, current_race):
    """
    Returns list of horse action rows from one text line.
    Handles:
      - Single:  "APACHE (Hung-out) - Remedial measures."
      - Multi comma/amp same action:
                 "CHASE THE ACE, DISCOURSE & PURE BLISS – to be shown..."
                 "AGE OF REASON & SHANDAAR – to be given more gate practice"
      - Two horses different actions on same line (rare, space-separated):
                 "HUDSON HAWK – to be shown...   MOUNTAIN WARRIOR (Sore) – to be passed fit"
    """
    line = line.replace("–", "-").replace("—", "-").strip()
    if not line:
        return []

    rows = []

    # Handle the rare case of two distinct horse entries on one line,
    # separated by a long gap or a full stop followed by an uppercase horse name.
    # Split on: ". HORSENAME" or "   HORSENAME" where HORSENAME starts a new entry
    segments = re.split(r"(?<=\.)\s{2,}(?=[A-Z]{2})|(?<=trials\.)\s+(?=[A-Z]{2})", line)

    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue

        # Find where the action starts: first " - " or " – " after horse block
        action_split = re.split(r"\s+-\s+", seg, maxsplit=1)
        if len(action_split) == 2:
            horse_block, action = action_split[0].strip(), action_split[1].strip()
        else:
            # No dash — skip or treat whole thing as horse with unknown action
            horse_block = seg
            action = ""

        # Condition in parentheses within horse block
        condition = ""
        cm = re.search(r"\(([^)]+)\)", horse_block)
        if cm:
            condition = cm.group(1).strip()
            horse_block = horse_block[:cm.start()].strip()

        # Split multiple horse names: comma or &
        horse_names = re.split(r"\s*,\s*|\s*&\s*", horse_block)

        for h in horse_names:
            h = h.strip().rstrip("-– ")
            if not h or not re.match(r"[A-Z]", h):
                continue

            rows.append({
                "date": meet_date,
                "venue": venue,
                "race_no": current_race,
                "horse": h,
                "action": action,
                "condition": condition,
            })

    return rows


def parse_summary_horse_actions(soup, meet_date, venue):
    rows = []

    for table in soup.find_all("table"):
        if "ACTION ON HORSE" not in table.get_text(" ").upper():
            continue

        in_section = False
        current_race = ""

        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if not tds:
                continue

            row_upper = " ".join(clean(td.get_text()) for td in tds).upper()

            if "ACTION ON HORSE" in row_upper:
                in_section = True

            if in_section and "PACIFIERS" in row_upper:
                break

            if not in_section:
                continue

            if len(tds) >= 3:
                race_td, detail_td = tds[1], tds[2]
            elif len(tds) == 2:
                race_td, detail_td = tds[0], tds[1]
            else:
                continue

            rno = extract_race_no(clean(race_td.get_text()))
            if rno:
                current_race = rno

            if not current_race:
                continue

            # *** Iterate each <p> tag separately — each is one horse entry (usually) ***
            p_tags = detail_td.find_all("p")
            lines = [clean(p.get_text()) for p in p_tags if clean(p.get_text())] if p_tags \
                else [clean(detail_td.get_text())]

            for line in lines:
                rows.extend(_parse_horse_action_line(line, meet_date, venue, current_race))

    log.info(f"summary_horse_actions={len(rows)}")
    return rows


# =============================================================================
# SUMMARY PACIFIERS
# =============================================================================

def parse_summary_pacifiers(soup, meet_date, venue):
    """
    Parses the PACIFIERS USED section.
    Horse names are comma/ampersand separated within a cell.
    """
    rows = []

    for table in soup.find_all("table"):
        if "PACIFIERS" not in table.get_text(" ").upper():
            continue

        in_section = False
        current_race = ""

        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if not tds:
                continue

            cell_texts = [clean(td.get_text()) for td in tds]
            row_upper = " ".join(cell_texts).upper()

            if "PACIFIERS" in row_upper:
                in_section = True

            if not in_section:
                continue

            if len(cell_texts) >= 3:
                race_cell, detail_cell = cell_texts[1], cell_texts[2]
            elif len(cell_texts) == 2:
                race_cell, detail_cell = cell_texts[0], cell_texts[1]
            else:
                continue

            rno = extract_race_no(race_cell)
            if rno:
                current_race = rno

            if not current_race or not detail_cell:
                continue

            # Split on comma or ampersand
            for h in re.split(r"[,&]", detail_cell):
                h = h.strip()
                if not h:
                    continue
                rows.append({
                    "date": meet_date,
                    "venue": venue,
                    "race_no": current_race,
                    "horse": h.upper(),
                })

    log.info(f"summary_pacifiers={len(rows)}")
    return rows

# =============================================================================
# CONDITION PARSER (FIXED CORE)
# =============================================================================

def handle_respectively(line, meta):
    if "respectively" not in line.lower():
        return []

    horses = extract_horses(line)

    if len(horses) < 2:
        return []

    return [{
        "date": meta["date"],
        "venue": meta["venue"],
        "race_no": meta["race_no"],
        "horse": h,
        "condition_type": infer_condition_type(line),
        "action": infer_action(line),
        "severity": infer_severity(line)
    } for h in horses]

def parse_conditions(lines, meta):
    rows = []

    for line in lines:

        line_rows = []

        # ---- MULTI HORSE CASE ----
        line_rows = handle_respectively(line, meta)

        # ---- SINGLE HORSE ----
        if not line_rows:
            l = line.lower()

            if any(k in l for k in ["sore","blood vessel","fractious","injury","collapsed"]):
                horse = extract_first_horse(line)

                if horse:
                    line_rows.append({
                        "date": meta["date"],
                        "venue": meta["venue"],
                        "race_no": meta["race_no"],
                        "horse": horse,
                        "condition_type": infer_condition_type(line),
                        "action": infer_action(line),
                        "severity": infer_severity(line)
                    })

        rows += line_rows

    return rows

# =============================================================================
# MAIN PARSER
# =============================================================================

def parse_file(filepath):
    log.info(f"Parsing: {filepath}")

    with open(filepath, "r", encoding=INPUT_ENCODING, errors="replace") as f:
        text = f.read()

    soup = BeautifulSoup(text, "html.parser")
    all_lines = [clean(p.get_text()) for p in soup.find_all("p") if clean(p.get_text())]

    date = ""
    venue = "Mumbai"
    for l in all_lines[:10]:
        if not date:
            date = parse_date(l)
        if "PUNE" in l.upper():
            venue = "Pune"

    # Truncate at stop phrases — everything after is urine results / admin
    lines = []
    for l in all_lines:
        if _is_stop_line(l):
            break
        lines.append(l)

    incidents = []
    conditions = []
    current_race = ""

    for line in lines:
        rm = re.search(r"RACE\s+NO\.?\s*(\d+)", line, re.I)
        if rm:
            current_race = rm.group(1)
            continue

        meta = {"date": date, "venue": venue, "race_no": current_race}
        incidents += parse_incidents([line], meta)
        conditions += parse_conditions([line], meta)

    jockey_changes    = parse_jockey_changes(soup, date, venue)
    summary_penalties = parse_summary_penalties(soup, date, venue)
    summary_horse_actions = parse_summary_horse_actions(soup, date, venue)
    summary_pacifiers = parse_summary_pacifiers(soup, date, venue)

    log.info(f"incidents={len(incidents)} conditions={len(conditions)}")

    return {
        "incidents":             incidents,
        "conditions":            conditions,
        "jockey_changes":        jockey_changes,
        "summary_penalties":     summary_penalties,
        "summary_horse_actions": summary_horse_actions,
        "summary_pacifiers":     summary_pacifiers,
    }

# =============================================================================
# CSV WRITER
# =============================================================================

SCHEMA = {
    "incidents": INCIDENT_COLS,
    "conditions": CONDITION_COLS,
    "jockey_changes": JOCKEY_CHANGE_COLS,
    "summary_penalties": SUMMARY_PENALTY_COLS,
    "summary_horse_actions": SUMMARY_HORSE_COLS,
    "summary_pacifiers": SUMMARY_PACIFIER_COLS,
}

def write_csv(data):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for key, cols in SCHEMA.items():
        path = os.path.join(OUTPUT_DIR, FILES[key])
        with open(path, WRITE_MODE, newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(data[key])
        log.info(f"{len(data[key])} -> {path}")

# =============================================================================
# ENTRY
# =============================================================================

def main():
    if os.path.isfile(INPUT_PATH):
        files = [INPUT_PATH]
    else:
        files = glob.glob(os.path.join(INPUT_PATH, "*.htm"))

    combined = {k: [] for k in SCHEMA}

    for f in files:
        res = parse_file(f)
        for k in SCHEMA:
            combined[k].extend(res[k])

    write_csv(combined)

if __name__ == "__main__":
    main()