import os
import re
import html
import logging
from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

#INPUT_PATH     = "./raw_html/declarations_201907-/2019-07-26.html"
#INPUT_PATH     = "./raw_html/declarations_201907-/2026-03-30.html"
INPUT_PATH     = "./raw_html/declarations_201907-"
OUTPUT_DIR     = "./raw/declarations/201907-"
WRITE_MODE     = "w"
INPUT_ENCODING = "utf-8"
LOG_LEVEL      = logging.INFO

OUTPUT_FILE    = "declarations.csv"

# ============================================================================

logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s | %(message)s")


# ============================================================================
# CLEANING
# ============================================================================
def clean_html(raw_html):
    text = html.unescape(raw_html)
    text = text.replace("Â", "").replace("\xa0", " ")
    return text


def clean_text(x):
    if not x:
        return ""
    return re.sub(r"\s+", " ", x).strip()

# ============================================================================
# FILE HANDLING
# ============================================================================
def get_input_files(path):
    p = Path(path)

    if p.is_file():
        return [p]

    if p.is_dir():
        return sorted(p.glob("*.htm")) + sorted(p.glob("*.html"))

    raise ValueError("Invalid INPUT_PATH")


# ============================================================================
# HEADER
# ============================================================================
def extract_meet_info(soup):
    header = soup.find("div", class_="pageHeading")

    if not header:
        return {"venue": "", "meet_date": ""}

    text = clean_text(header.get_text(" "))

    venue = "Mumbai" if "MUMBAI" in text.upper() else "Pune"

    match = re.search(r"(\d{1,2})(st|nd|rd|th)?\s+([A-Za-z]+)\s+(\d{4})", text)

    if match:
        day = int(match.group(1))
        month = match.group(3)
        year = int(match.group(4))

        dt = datetime.strptime(f"{day} {month} {year}", "%d %B %Y")
        meet_date = dt.strftime("%Y-%m-%d")
    else:
        meet_date = ""

    return {"venue": venue, "meet_date": meet_date}


# ============================================================================
# FORMAT DETECTION
# ============================================================================
def detect_format(soup):
    if soup.find("tr", class_="perform_data"):
        return "A"
    return "B"


# ============================================================================
# FORMAT A PARSER
# ============================================================================
def parse_format_a(soup, meet_info):
    data = []

    headers = soup.find_all("th", class_="racehead")
    tables = soup.find_all("table", class_="tops")

    for i, header in enumerate(headers):
        race_no_match = re.search(r"Race No\.\:?(\d+)", header.text)
        race_no = int(race_no_match.group(1)) if race_no_match else i + 1

        table = tables[i] if i < len(tables) else None
        if not table:
            continue

        rows = table.find_all("tr", class_="perform_data")

        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 8:
                continue

            horse_name = clean_text(cols[0].get_text())
            horse_name = re.sub(r"^\d+\.\s*", "", horse_name)

            data.append({
                "venue": meet_info["venue"],
                "meet_date": meet_info["meet_date"],
                "race_no": race_no,
                "horse_name": horse_name,
                "weight": clean_text(cols[2].text),
                "rating": clean_text(cols[3].text),
                "trainer": clean_text(cols[4].text),
                "jockey": clean_text(cols[5].text),
                "horse_weight": clean_text(cols[6].text),
                "shoe": clean_text(cols[7].text),
                "draw": clean_text(cols[8].text) if len(cols) > 8 else ""
            })

    return data


# ============================================================================
# FORMAT B PARSER (YOUR PASTED FILE)
# ============================================================================
def parse_format_b(soup, meet_info):
    data = []

    race_tables = soup.find_all("table", class_="table-bordered")

    for table in race_tables:
        # --- Extract race number ---
        race_header = table.find("table", class_="raceHeading")
        if not race_header:
            continue

        header_text = race_header.get_text(" ", strip=True)

        # 🔥 Extract REAL race number
        match = re.search(r"Race\s*No\.?\s*:?[\s]*(\d+)", header_text, re.IGNORECASE)

        if not match:
            continue

        race_no = int(match.group(1))

        # --- Extract horse rows ---
        for row in table.find_all("tr"):
            # 🚫 Skip rows that belong to nested tables (next races)
            if row.find_parent("table", class_="table-bordered") != table:
                continue

            cols = row.find_all("td")
            if len(cols) < 7:
                continue

            name_tag = cols[0].find("a")
            if not name_tag:
                continue

            horse_name = clean_text(name_tag.text)

            data.append({
                "venue": meet_info["venue"],
                "meet_date": meet_info["meet_date"],
                "race_no": race_no,
                "horse_name": horse_name,
                "weight": clean_text(cols[1].text),
                "rating": clean_text(cols[2].text),
                "trainer": clean_text(cols[3].text),
                "jockey": clean_text(cols[4].text),
                "horse_weight": clean_text(cols[5].text),
                "shoe": clean_text(cols[6].text),
                "draw": clean_text(cols[7].text) if len(cols) > 7 else ""
            })

    return data

# ============================================================================
# FILE PARSER
# ============================================================================
def parse_file(file_path):
    logging.info(f"Processing: {file_path}")

    with open(file_path, "r", encoding=INPUT_ENCODING, errors="ignore") as f:
        html_content = f.read()

    soup = BeautifulSoup(clean_html(html_content), "lxml")

    meet_info = extract_meet_info(soup)
    fmt = detect_format(soup)

    if fmt == "A":
        return parse_format_a(soup, meet_info)
    else:
        return parse_format_b(soup, meet_info)


# ============================================================================
# MAIN
# ============================================================================
def main():
    files = get_input_files(INPUT_PATH)

    all_data = []

    for f in files:
        try:
            all_data.extend(parse_file(f))
        except Exception as e:
            logging.error(f"Failed: {f} | {e}")

    if not all_data:
        logging.warning("No data extracted")
        return

    df = pd.DataFrame(all_data)

    df = df.drop_duplicates(
    subset=["meet_date", "race_no", "horse_name"]
)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    write_header = not (WRITE_MODE == "a" and os.path.exists(out_path))

    df.to_csv(out_path, mode=WRITE_MODE, header=write_header, index=False)

    logging.info(f"Saved: {out_path}")
    logging.info(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()