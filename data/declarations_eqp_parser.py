import os, re, csv, sys, glob, logging
from bs4 import BeautifulSoup

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_PATH     = "./raw_doc/declarations_htm_201907-"   # single .htm/.html file OR folder
OUTPUT_DIR     = "./raw/declarations/201907-"
WRITE_MODE     = "w"           # "w" = overwrite, "a" = append
INPUT_ENCODING = "utf-8"
LOG_LEVEL      = logging.INFO

OUTPUT_FILE    = "equipment_changes.csv"

COLS = ["meet_date", "venue", "race_no", "race_name", "horse_name", "equip_change"]

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
    """Handles both '4TH JANUARY, 2018' and '30TH, MARCH, 2026'"""
    m = re.search(
        r"(\d{1,2})\s*(?:ST|ND|RD|TH)?\s*,?\s*"
        r"(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|"
        r"AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)"
        r"\s*,?\s*(\d{4})",
        text, re.I
    )
    if m:
        day   = int(m.group(1))
        month = MONTHS[m.group(2).lower()]
        year  = int(m.group(3))
        return f"{year:04d}-{month:02d}-{day:02d}"
    return ""


# ============================================================================
# CORE PARSER
# ============================================================================

def parse_file(filepath):
    """
    Returns a list of dicts with keys matching COLS.
    """
    try:
        with open(filepath, encoding=INPUT_ENCODING, errors="replace") as f:
            html = f.read()
    except Exception as e:
        log.error(f"Cannot read {filepath}: {e}")
        return []

    soup = BeautifulSoup(html, "html.parser")

    # ------------------------------------------------------------------
    # 1. Collect all paragraph texts in document order
    # ------------------------------------------------------------------
    paras = []
    for p in soup.find_all("p"):
        t = clean(p.get_text())
        if t:
            paras.append(t)

    # ------------------------------------------------------------------
    # 2. Extract meet date & venue from header paragraphs
    # ------------------------------------------------------------------
    meet_date = ""
    venue     = "Mumbai"

    for p in paras[:20]:          # header is near the top
        if not meet_date:
            meet_date = parse_date_text(p)
        pu = p.upper()
        if "PUNE" in pu:
            venue = "Pune"
        elif "MUMBAI" in pu:
            venue = "Mumbai"
        if meet_date:             # once we have a date, check a few more for venue
            if any(kw in p.upper() for kw in ("PUNE", "MUMBAI")):
                break

    if not meet_date:
        log.warning(f"No meet date found in {filepath}")

    # ------------------------------------------------------------------
    # 3. Walk paragraphs, tracking current race; collect Chg.Equip lines
    # ------------------------------------------------------------------
    # Race header patterns
    # Type A (older): "1. THE BRAVE HUNTER PLATE   (Class V; ...)"  on its own line
    # Type A distance: "(About) 1600 Metres.  Time: 04:30 P.M. Race no: 181"
    # Type B (inline): "1. THE BEYOND EXPECTATION PLATE - DIVISION II  (Class V; ...)"
    #                   followed immediately by "(About) 1000 Metres. Time: 2.00 P.M. Race no: 1"

    _RACE_NAME_LINE = re.compile(r"^(\d+)\.\s+(.+)", re.S)
    _RACE_NO_LINE   = re.compile(r"Race\s+no[:\s]*(\d+)", re.I)
    _DIST_TIME_LINE = re.compile(r"\(About\)\s+\d+\s+Metres", re.I)

    current_race_no   = ""
    current_race_name = ""
    rows = []

    for p in paras:
        # ---- Try to detect a race header line ----
        race_m = _RACE_NAME_LINE.match(p)
        if race_m:
            candidate_no   = race_m.group(1)
            candidate_rest = clean(race_m.group(2))

            # Skip runner lines: they start with a small number and contain "|"
            # Runner lines look like: "1. CLYMENE   59   (102) |T. S. Jodha ..."
            if "|" not in candidate_rest:
                # This looks like a real race header
                current_race_no = candidate_no

                # Try to extract race name (everything before first "(")
                name_m = re.match(r"^(.*?)\s*\(", candidate_rest)
                current_race_name = name_m.group(1).strip() if name_m else candidate_rest.strip()
                # Clean trailing dash/whitespace
                current_race_name = re.sub(r"\s*[-–]\s*$", "", current_race_name).strip()

        # ---- If distance/time line contains "Race no:", update race_no ----
        if _DIST_TIME_LINE.search(p):
            rno_m = _RACE_NO_LINE.search(p)
            if rno_m:
                current_race_no = rno_m.group(1)

        # ---- Chg.Equip. line ----
        if "Chg.Equip" in p or ";" in p:

            line = p.replace("Chg.Equip.", "").replace("Chg.Equip:", "")

            entries = re.split(r"(?=:\s*[A-Z])|;", line)

            for entry in entries:

                entry = clean(entry)
                entry = re.sub(r"^:\s*", "", entry)

                m = re.search(r"\b(BLK|TS|VISOR|HOOD|EP|PACI|CNB|RM|SSCP|PB|LES|RES)\b", entry)
                if not m:
                    continue

                idx = m.start()

                horse_name = entry[:idx].strip().rstrip(",")
                equip_change = entry[idx:].strip().rstrip(",")

                rows.append({
                    "meet_date": meet_date,
                    "venue": venue,
                    "race_no": current_race_no,
                    "race_name": current_race_name,
                    "horse_name": horse_name,
                    "equip_change": equip_change,
                })
                log.debug(f"  Chg.Equip: race={current_race_no} horse={horse_name!r} change={equip_change!r}")

    return rows


# ============================================================================
# FILE DISCOVERY
# ============================================================================

def collect_files(path):
    if os.path.isfile(path):
        return [path]
    files = []
    for ext in ("*.htm", "*.html"):
        files.extend(glob.glob(os.path.join(path, "**", ext), recursive=True))
    return sorted(files)


# ============================================================================
# MAIN
# ============================================================================

def main():
    input_path = INPUT_PATH
    if len(sys.argv) > 1:
        input_path = sys.argv[1]

    files = collect_files(input_path)
    if not files:
        log.error(f"No .htm/.html files found at: {input_path}")
        sys.exit(1)

    log.info(f"Found {len(files)} file(s) to process")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    all_rows = []
    for fp in files:
        log.info(f"Parsing: {fp}")
        rows = parse_file(fp)
        log.info(f"  → {len(rows)} Chg.Equip row(s)")
        all_rows.extend(rows)

    with open(out_path, WRITE_MODE, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLS)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({
                "meet_date":   row["meet_date"],
                "venue":       row["venue"],
                "race_no":     row["race_no"],
                "race_name":   row["race_name"],
                "horse_name":  row["horse_name"],
                "equip_change":row["equip_change"],
            })

    log.info(f"Written {len(all_rows)} rows → {out_path}")


if __name__ == "__main__":
    main()