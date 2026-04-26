import os, re, csv, glob, logging
from bs4 import BeautifulSoup

# ============================================================================
# CONFIGURATION
# ============================================================================

#INPUT_PATH   = "./raw_html/rating_change/2018-01-04.html"
#INPUT_PATH   = "./raw_html/rating_change/2025-08-24.html"
INPUT_PATH   = "./raw_html/rating_change"
OUTPUT_DIR   = "./raw/ratings"
WRITE_MODE   = "w"
INPUT_ENCODING = "utf-8"

# ============================================================================
# OUTPUT FILES
# ============================================================================

FILES = {
    "ratings_change":    "ratings_change.csv",
    "additional":        "additional_ratings.csv",
    "starters_remark":   "remarks.csv",
}

# ============================================================================
# COLUMNS
# ============================================================================

RATING_COLS = ["meet_date", "race_range", "horse_name", "new_rating", "old_rating"]
ADDITIONAL_COLS = ["meet_date", "race_range", "horse_name", "rating"]
REMARK_COLS = ["meet_date", "race_no", "horse_name", "remark", "remark_source"]

# ============================================================================
# SETUP
# ============================================================================

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MONTHS = {
    "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
    "july":7,"august":8,"september":9,"october":10,"november":11,"december":12
}

# ============================================================================
# HELPERS
# ============================================================================

def clean(t):
    if not t: return ""
    t = t.replace("\xa0"," ")
    return re.sub(r"\s+"," ",t).strip()

def parse_date(text):
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

def get_paragraphs(soup):
    texts = []
    for p in soup.find_all("p"):
        # IMPORTANT: no strip=True
        t = p.get_text(" ")
        t = t.replace("\xa0", " ")
        t = re.sub(r"\s+", " ", t).strip()
        if t:
            texts.append(t)
    return texts

def get_paragraphs_with_gaps(soup):
    texts = []

    for p in soup.find_all("p"):
        t = p.get_text(" ")
        t = t.replace("\xa0", " ")

        # normalize but DO NOT strip completely
        t_clean = re.sub(r"\s+", " ", t).strip()

        # preserve empty paragraphs as ""
        if t_clean:
            texts.append(t_clean)
        else:
            texts.append("")   # 🔴 this is the key

    return texts

# ============================================================================
# PARSERS
# ============================================================================

def extract_race_range(text):
    m = re.search(r"From Race No\s+(\d+)\s+to\s+(\d+)", text)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return ""

# ------------------ RATINGS ------------------

def parse_ratings(lines, meet_date):
    rows = []
    race_range = ""
    in_section = False

    for line in lines:

        if "REVISED RATINGS" in line:
            in_section = True
            continue

        if in_section and "From Race No" in line:
            race_range = extract_race_range(line)
            continue

        if in_section and "Additional Ratings" in line:
            break

        if not in_section:
            continue

        # match: NAME 45(50)
        pattern = re.findall(
            r"([A-Z' ]+?)\s+(\d+)(?:\s*\(\s*(\d+)\s*\))?",
            line
        )

        for name, new, old in pattern:
            rows.append({
                "meet_date": meet_date,
                "race_range": race_range,
                "horse_name": clean(name),
                "new_rating": int(new),
                "old_rating": int(old) if old else None
            })

    return rows


# ------------------ ADDITIONAL ------------------

def parse_additional(lines, meet_date, race_range):
    rows = []
    in_section = False

    for line in lines:

        if "Additional Ratings" in line:
            in_section = True
            continue

        if "STARTER" in line:
            break

        if not in_section:
            continue

        pattern = re.findall(r"([A-Z' ]+?)\s+(\d+)", line)

        for name, rating in pattern:
            rows.append({
                "meet_date": meet_date,
                "race_range": race_range,
                "horse_name": clean(name),
                "rating": int(rating)
            })

    return rows


# ------------------ REMARKS ------------------

def parse_remarks(lines, meet_date):
    horse_pat = re.compile(r"RACE\s*NO[:\s]*?(\d+)\s*[:\s]*([A-Z][A-Z '\-]+)")
    remark_pat = re.compile(r"}\s*(.+)")

    rows = []

    current_source = None
    pending_horses = []
    remark_parts = []

    def flush():
        if not pending_horses or not remark_parts or not current_source:
            return

        full_remark = " ".join(remark_parts)
        full_remark = re.sub(r"\s+", " ", full_remark).strip()
        full_remark = re.sub(r"([a-z])([A-Z])", r"\1 \2", full_remark)

        for rn, hn in pending_horses:
            rows.append({
                "meet_date": meet_date,
                "race_no": rn,
                "horse_name": hn,
                "remark": full_remark,
                "remark_source": current_source
            })

    for line in lines:
        line_u = line.upper()

        # -------- SECTION HEADER --------
        if "REMARK" in line_u:
            flush()
            pending_horses = []
            remark_parts = []

            if "STIPENDIARY" in line_u or "STEWARD" in line_u:
                current_source = "STEWARDS"
            elif "STARTER" in line_u:
                current_source = "STARTER"
            else:
                current_source = clean(line.split("REMARK")[0])

            continue

        if not current_source:
            continue

        # -------- GROUP BOUNDARY (empty line) --------
        if line.strip() == "":
            flush()
            pending_horses = []
            remark_parts = []
            continue

        h = horse_pat.search(line)
        r = remark_pat.search(line)

        # -------- HORSE --------
        if h:
            race_no = int(h.group(1))
            horse = clean(h.group(2))
            pending_horses.append((race_no, horse))

        # -------- REMARK --------
        if r:
            txt = clean(r.group(1))
            if txt:
                remark_parts.append(txt)

    # -------- FINAL FLUSH --------
    flush()

    return rows


# ============================================================================
# MAIN PROCESS
# ============================================================================

def process_file(fp):
    with open(fp, "r", encoding=INPUT_ENCODING, errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")

    lines = get_paragraphs(soup)
    remark_lines = get_paragraphs_with_gaps(soup)
    
    full = " ".join(lines)

    meet_date = parse_date(full)
    race_range = extract_race_range(full)

    ratings = parse_ratings(lines, meet_date)
    additional = parse_additional(lines, meet_date, race_range)
    remarks = parse_remarks(remark_lines, meet_date)

    return ratings, additional, remarks


# ============================================================================
# CSV
# ============================================================================

def write_csv(path, rows, cols):
    write_header = not os.path.exists(path) or WRITE_MODE == "w"

    with open(path, WRITE_MODE, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if write_header:
            w.writeheader()
        w.writerows(rows)


# ============================================================================
# MAIN
# ============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_ratings, all_additional, all_remarks = [], [], []

    if os.path.isfile(INPUT_PATH):
        r, a, rm = process_file(INPUT_PATH)
        all_ratings += r
        all_additional += a
        all_remarks += rm

    else:
        files = glob.glob(os.path.join(INPUT_PATH, "*.htm")) + glob.glob(os.path.join(INPUT_PATH, "*.html"))
        for i, f in enumerate(files, 1):
            log.info(f"[{i}/{len(files)}] Processing: {os.path.basename(f)}")
            r, a, rm = process_file(f)
            all_ratings += r
            all_additional += a
            all_remarks += rm

    write_csv(os.path.join(OUTPUT_DIR, FILES["ratings_change"]), all_ratings, RATING_COLS)
    write_csv(os.path.join(OUTPUT_DIR, FILES["additional"]), all_additional, ADDITIONAL_COLS)
    write_csv(os.path.join(OUTPUT_DIR, FILES["starters_remark"]), all_remarks, REMARK_COLS)

    log.info(f"ratings: {len(all_ratings)}")
    log.info(f"additional: {len(all_additional)}")
    log.info(f"remarks: {len(all_remarks)}")


if __name__ == "__main__":
    main()