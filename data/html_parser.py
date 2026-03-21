#!/usr/bin/env python3
"""
RWITC Race Results HTML Parser
================================
Parses horse racing result pages from rwitc.com (2010–2026+) into structured CSVs.

Requirements: pip install beautifulsoup4

Usage:
  1. Edit the CONFIGURATION section below
  2. Run: python rwitc_parser.py
"""

import os
import re
import csv
import sys
import glob
import logging
from bs4 import BeautifulSoup

# ============================================================================
# CONFIGURATION — Edit these lines as needed
# ============================================================================

INPUT_PATH = "./raw_html"         # <-- EDIT: single .html file path OR folder of .html files
OUTPUT_DIR = "./raw"              # <-- EDIT: directory where CSV files will be saved
MEETINGS_FILE = "meetings.csv"    # <-- EDIT: meetings CSV filename
RACES_FILE = "races.csv"          # <-- EDIT: races CSV filename
RUNNERS_FILE = "runners.csv"      # <-- EDIT: runners CSV filename
EXOTICS_FILE = "exotics.csv"      # <-- EDIT: exotics CSV filename
WRITE_MODE = "w"                  # <-- EDIT: "w" = overwrite (fresh), "a" = append to existing
INPUT_ENCODING = "utf-8"          # <-- EDIT: encoding of your HTML files (try "latin-1" if utf-8 fails)
LOG_LEVEL = logging.INFO          # <-- EDIT: DEBUG for verbose, INFO for normal, WARNING for quiet

# ============================================================================
# COLUMN DEFINITIONS — defines CSV headers and field order
# ============================================================================

MEETING_COLS = [
    'meet_date', 'venue', 'season', 'meeting_day_desc', 'session',
    'weather', 'track_condition', 'penetrometer', 'false_rails',
]

RACE_COLS = [
    'meet_date', 'venue', 'race_no', 'card_seq', 'race_name', 'class_conditions',
    'scheduled_time', 'distance_text', 'distance_meters', 'video_url',
    'ownership', 'breeder', 'margins', 'results_by_card',
    'tote_favourite', 'win_div', 'place_div', 'shp_div', 'for_div', 'qnl_div', 'tnl_div',
]

RUNNER_COLS = [
    'meet_date', 'venue', 'race_no', 'placing', 'horse_name', 'horse_seq',
    'sire', 'sire_nat', 'dam', 'dam_nat', 'weight', 'jockey', 'jockey_claim',
    'trainer', 'odds', 'finish_time', 'horse_body_wt',
]

EXOTIC_COLS = [
    'meet_date', 'venue', 'pool_type', 'legs', 'winners',
    'div_70pct', 'tickets_70pct', 'div_30pct', 'tickets_30pct',
    'dividend', 'tickets', 'carried_forward',
]

# ============================================================================
# INTERNAL CONSTANTS
# ============================================================================

logging.basicConfig(level=LOG_LEVEL, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

MONTHS = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
    'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
}

POOL_KEYWORDS = [
    'SUPER JACKPOT', 'FIRST JACKPOT', 'SECOND JACKPOT', 'JACKPOT',
    'FIRST TREBLE', 'SECOND TREBLE', 'THIRD TREBLE', 'TREBLE',
]


# ============================================================================
# HELPERS
# ============================================================================

def clean(text):
    """Normalize whitespace, replace non-breaking spaces, strip."""
    if not text:
        return ''
    text = text.replace('\xa0', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_nationality(name):
    """'Speaking of Which[IRE]' -> ('Speaking of Which', 'IRE')"""
    m = re.search(r'\[(\w+)\]\s*$', name)
    if m:
        return name[:m.start()].strip(), m.group(1)
    return name.strip(), ''


def parse_jockey_field(raw):
    """'P. Trevor - 3.5' -> ('P. Trevor', '3.5'); 'A. Sandesh' -> ('A. Sandesh', '')"""
    raw = clean(raw)
    m = re.match(r'^(.+?)\s+-\s+(\d+\.?\d*)\s*$', raw)
    if m:
        return m.group(1).strip(), m.group(2)
    return raw, ''


def parse_date_from_text(text):
    """Extract date from text like 'Saturday 7th March 2026' -> '2026-03-07'."""
    m = re.search(
        r'(\d{1,2})\s*(?:st|nd|rd|th)?\s+'
        r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+'
        r'(\d{4})', text, re.IGNORECASE
    )
    if m:
        day, month_name, year = int(m.group(1)), m.group(2).lower(), int(m.group(3))
        return f"{year:04d}-{MONTHS[month_name]:02d}-{day:02d}"
    return ''


def parse_distance_meters(text):
    """Extract numeric meters from '(About) 2400 Metres.' -> '2400'."""
    m = re.search(r'(\d+)\s*(?:Metres|Mtrs)', text, re.IGNORECASE)
    return m.group(1) if m else ''


def parse_tote_dividends(text):
    """Parse 'WIN : 33 PLACE : 21,12 SHP : 21 ...' into dict."""
    r = {'win_div': '', 'place_div': '', 'shp_div': '', 'for_div': '', 'qnl_div': '', 'tnl_div': ''}
    text = clean(text)
    if not text:
        return r
    for key, pat in [
        ('win_div',   r'WIN\s*:?\s*(.*?)(?=\s*PLACE\s*:|$)'),
        ('place_div', r'PLACE\s*:?\s*(.*?)(?=\s*SHP\s*:|$)'),
        ('shp_div',   r'SHP\s*:?\s*(.*?)(?=\s*FOR\s*:|$)'),
        ('for_div',   r'FOR\s*:?\s*(.*?)(?=\s*QNL\s*:|$)'),
        ('qnl_div',   r'QNL\s*:?\s*(.*?)(?=\s*TNL\s*:|$)'),
        ('tnl_div',   r'TNL\s*:?\s*(.+?)$'),
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            r[key] = m.group(1).strip()
    return r


# ============================================================================
# TABLE CLASSIFICATION
# ============================================================================

def classify_tables(soup):
    """Sort all contentTable elements into conditions / race / exotic buckets."""
    conditions_table = None
    race_tables = []
    exotic_tables = []

    for table in soup.find_all('table', class_='contentTable'):
        th_texts = [clean(th.get_text()).upper() for th in table.find_all('th')]
        joined = ' '.join(th_texts)

        # 1) Conditions table — has WEATHER or PENETROMETER but NOT Placing
        if ('WEATHER' in joined or 'PENETROMETER' in joined) and 'PLACING' not in joined:
            conditions_table = table
            continue

        # 2) Race table — has Placing column header
        if 'PLACING' in th_texts:
            race_tables.append(table)
            continue

        # 3) Exotic table — has a pool keyword
        if any(pk in joined for pk in POOL_KEYWORDS):
            exotic_tables.append(table)
            continue

    return conditions_table, race_tables, exotic_tables


# ============================================================================
# MEETING HEADER PARSING
# ============================================================================

def parse_meeting_header(soup):
    """Extract date, venue, season, description, session from page header."""
    info = {'meet_date': '', 'venue': '', 'season': '', 'meeting_day_desc': '', 'session': ''}

    heading = soup.find('div', class_='pageHeading') or soup.find('div', class_='pageHeader')
    if not heading:
        log.warning("No pageHeading/pageHeader div found")
        return info

    full = clean(heading.get_text())

    # Date
    info['meet_date'] = parse_date_from_text(full)

    # Venue
    upper = full.upper()
    if 'MUMBAI' in upper:
        info['venue'] = 'Mumbai'
    elif 'PUNE' in upper:
        info['venue'] = 'Pune'
    else:
        info['venue'] = 'Mumbai'  # default for older pages

    # Season (e.g. "2025/26" or "2015")
    m = re.search(r'(\d{4}/\d{2,4})', full)
    if m:
        info['season'] = m.group(1)
    else:
        m = re.search(r'MEETING\s+(\d{4})', full, re.IGNORECASE)
        if m:
            info['season'] = m.group(1)

    # Meeting day description
    m = re.search(
        r'((?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|'
        r'ELEVENTH|TWELFTH|THIRTEENTH|FOURTEENTH|FIFTEENTH|SIXTEENTH|SEVENTEENTH|'
        r'EIGHTEENTH|NINETEENTH|TWENTIETH|TWENTY[\s-]?\w+|SPECIAL\s+RACE)\s+DAY)',
        full, re.IGNORECASE
    )
    if m:
        info['meeting_day_desc'] = m.group(1).strip()

    # Session
    m = re.search(r'\(?\s*(EVENING\s+RACE\s+DAY)\s*\)?', full, re.IGNORECASE)
    if m:
        info['session'] = m.group(1).strip()

    return info


def parse_conditions_table(table):
    """Parse the weather/track/penetrometer/rails table."""
    c = {'weather': '', 'track_condition': '', 'penetrometer': '', 'false_rails': ''}
    if table is None:
        return c

    for row in table.find_all('tr'):
        th = row.find('th')
        td = row.find('td')
        if not th or not td:
            continue
        label = clean(th.get_text()).upper()
        value = clean(td.get_text())

        if 'WEATHER' in label and 'TRACK' not in label:
            # Older format packs track condition into the weather cell
            if re.search(r'Track\s+Condition', value, re.IGNORECASE):
                parts = re.split(r'Track\s+Condition\s*:?\s*', value, maxsplit=1, flags=re.IGNORECASE)
                c['weather'] = parts[0].strip().rstrip(',').strip()
                if len(parts) > 1:
                    c['track_condition'] = parts[1].strip().rstrip('.')
            else:
                c['weather'] = value

        elif 'TRACK' in label and 'CONDITION' in label:
            c['track_condition'] = value

        elif 'PENETROMETER' in label:
            c['penetrometer'] = value

        elif 'RAIL' in label:
            c['false_rails'] = value

    return c


# ============================================================================
# RACE TABLE PARSING
# ============================================================================

def parse_race_table(table, meet_date, venue):
    """Parse one race table -> (race_dict, [runner_dicts])."""
    race = {col: '' for col in RACE_COLS}
    race['meet_date'] = meet_date
    race['venue'] = venue
    runners = []

    rows = table.find_all('tr')

    # ---- Locate the "Placing" header row ----
    placing_idx = None
    for i, row in enumerate(rows):
        if any(clean(th.get_text()).upper() == 'PLACING' for th in row.find_all('th')):
            placing_idx = i
            break

    if placing_idx is None:
        log.warning("No Placing header found — skipping table")
        return None, []

    # ---- Row 0: race number + details + video ----
    row0_ths = rows[0].find_all('th')

    # Race number from first th: "No.: 137"
    if row0_ths:
        m = re.search(r'No\.?\s*:?\s*(\d+)', clean(row0_ths[0].get_text()))
        if m:
            race['race_no'] = m.group(1)

    # Details cell: the th with colspan > 1
    details_th = None
    for th in row0_ths:
        try:
            if int(th.get('colspan', '1')) > 1:
                details_th = th
                break
        except (ValueError, TypeError):
            pass

    if details_th:
        spans = details_th.find_all('span')
        full_detail = clean(details_th.get_text())

        # Race name = first span
        if spans:
            race['race_name'] = clean(spans[0].get_text())

        # Class/conditions = second span + any extra non-time/distance spans
        if len(spans) > 1:
            race['class_conditions'] = clean(spans[1].get_text())
            extras = []
            for s in spans[2:]:
                st = clean(s.get_text())
                if st and not re.match(r'^Time\s*:', st) and not re.match(r'^\(?\s*About', st, re.IGNORECASE):
                    extras.append(st)
            if extras:
                race['class_conditions'] += ' ' + ' '.join(extras)
                race['class_conditions'] = clean(race['class_conditions'])

        # Time
        m = re.search(r'Time\s*:\s*([\d.:]+\s*[APap]\.?\s*[Mm]\.?)', full_detail)
        if m:
            race['scheduled_time'] = clean(m.group(1))

        # Distance
        dm = re.search(r'(\(?\s*About\s*\)?\s*\d+\s*(?:Metres|Mtrs)\.?)', full_detail, re.IGNORECASE)
        if dm:
            race['distance_text'] = clean(dm.group(1))
        race['distance_meters'] = parse_distance_meters(full_detail)

    # Video URL
    for th in row0_ths:
        a = th.find('a')
        if a and 'video' in clean(a.get_text()).lower():
            race['video_url'] = a.get('href', '')
            break

    # ---- Row 1: card sequence number ----
    if placing_idx > 1:
        row1_ths = rows[1].find_all('th')
        if row1_ths:
            seq = clean(row1_ths[0].get_text())
            if seq.isdigit():
                race['card_seq'] = seq

    # ---- Rows after Placing: runners + post-race info ----
    for row in rows[placing_idx + 1:]:
        cells = row.find_all(['th', 'td'])
        if not cells:
            continue

        if cells[0].name == 'td':
            # Runner row
            runner = _parse_runner(cells, meet_date, venue, race['race_no'])
            if runner:
                runners.append(runner)
        elif cells[0].name == 'th':
            # Post-race info row
            label = clean(cells[0].get_text()).upper()
            val_cells = [c for c in cells if c.name == 'td']
            value = clean(' '.join(c.get_text() for c in val_cells))

            if 'OWNERSHIP' in label:
                race['ownership'] = value
            elif 'BREEDER' in label:
                race['breeder'] = value
            elif label == 'DISTANCE' or ('DISTANCE' in label and 'METRE' not in label):
                race['margins'] = value
            elif 'RESULT' in label and 'CARD' in label:
                race['results_by_card'] = value
            elif 'FAVOURITE' in label:
                race['tote_favourite'] = value
            elif 'DIVIDEND' in label:
                race.update(parse_tote_dividends(value))

    return race, runners


def _parse_runner(cells, meet_date, venue, race_no):
    """Parse a single runner row (list of 8 td elements)."""
    if len(cells) < 8:
        return None

    r = {col: '' for col in RUNNER_COLS}
    r['meet_date'] = meet_date
    r['venue'] = venue
    r['race_no'] = race_no

    # 0: Placing
    r['placing'] = clean(cells[0].get_text())

    # 1: Horse (name, seq, sire, dam)
    hcell = cells[1]
    link = hcell.find('a')
    if link:
        r['horse_name'] = clean(link.get_text()).rstrip(')')
        href = link.get('href', '')
        m = re.search(r'horseseq=(\d+)', href)
        if not m:
            m = re.search(r'horseseq[^0-9]*(\d+)', str(link))
        if m:
            r['horse_seq'] = m.group(1)

    breed_span = hcell.find('span')
    if breed_span:
        # Get sire from text before the dam-link, dam from the link
        dam_link = breed_span.find('a')

        # Full text method as fallback
        breed_text = clean(breed_span.get_text()).strip('()')

        if '-' in breed_text:
            sire_raw, dam_raw = breed_text.split('-', 1)
            sire_raw = sire_raw.strip()
            dam_raw = dam_raw.strip().rstrip(')')

            r['sire'], r['sire_nat'] = extract_nationality(sire_raw)

            # Prefer dam name from link text (avoids trailing-paren issues)
            if dam_link:
                dam_from_link = clean(dam_link.get_text()).rstrip(')')
                dam_clean, dam_nat_text = extract_nationality(dam_from_link)
                r['dam'] = dam_clean
                r['dam_nat'] = dam_nat_text
                # Fallback: damnat URL param
                if not r['dam_nat']:
                    m2 = re.search(r'damnat=(\w+)', dam_link.get('href', ''))
                    if m2:
                        r['dam_nat'] = m2.group(1)
            else:
                r['dam'], r['dam_nat'] = extract_nationality(dam_raw)

    # 2: Weight
    r['weight'] = clean(cells[2].get_text())

    # 3: Jockey
    r['jockey'], r['jockey_claim'] = parse_jockey_field(clean(cells[3].get_text()))

    # 4: Trainer
    r['trainer'] = clean(cells[4].get_text())

    # 5: Odds
    r['odds'] = clean(cells[5].get_text())

    # 6: Finish time
    r['finish_time'] = clean(cells[6].get_text())

    # 7: Horse body weight
    r['horse_body_wt'] = clean(cells[7].get_text())

    return r


# ============================================================================
# EXOTIC POOL PARSING
# ============================================================================

def parse_exotic_tables(tables, meet_date, venue):
    """Parse all exotic tables (jackpots, trebles) — handles multiple pools per table."""
    results = []
    for table in tables:
        rows = table.find_all('tr')
        current_pool_type = None
        current_rows = []

        for row in rows:
            # Check if this row is a pool-type header (th with colspan>=3 and known keyword)
            header_type = _detect_pool_header(row)
            if header_type:
                # Flush previous pool
                if current_pool_type and current_rows:
                    results.append(_parse_pool(current_pool_type, current_rows, meet_date, venue))
                current_pool_type = header_type
                current_rows = []
            elif current_pool_type is not None:
                current_rows.append(row)

        # Flush last pool in table
        if current_pool_type and current_rows:
            results.append(_parse_pool(current_pool_type, current_rows, meet_date, venue))

    return results


def _detect_pool_header(row):
    """If this row is a pool-type header, return the pool type string; else None."""
    for th in row.find_all('th'):
        try:
            colspan = int(th.get('colspan', '1'))
        except (ValueError, TypeError):
            colspan = 1
        if colspan >= 3:
            text = clean(th.get_text()).upper()
            if text in POOL_KEYWORDS:
                return text
    return None


def _parse_pool(pool_type, rows, meet_date, venue):
    """Parse detail rows for one exotic pool."""
    p = {col: '' for col in EXOTIC_COLS}
    p['meet_date'] = meet_date
    p['venue'] = venue
    p['pool_type'] = pool_type

    for row in rows:
        cells = row.find_all(['th', 'td'])
        if len(cells) < 2:
            continue
        label = clean(cells[0].get_text()).upper()

        if 'LEG' in label:
            p['legs'] = clean(cells[1].get_text())
        elif 'WINNER' in label:
            p['winners'] = clean(cells[1].get_text())
        elif 'CARRIED' in label or 'FORWARD' in label:
            p['carried_forward'] = clean(cells[1].get_text())
        elif '70%' in label:
            p['div_70pct'] = clean(cells[1].get_text())
            if len(cells) >= 4 and 'TICKET' in clean(cells[2].get_text()).upper():
                p['tickets_70pct'] = clean(cells[3].get_text())
        elif '30%' in label:
            p['div_30pct'] = clean(cells[1].get_text())
            if len(cells) >= 4 and 'TICKET' in clean(cells[2].get_text()).upper():
                p['tickets_30pct'] = clean(cells[3].get_text())
        elif 'DIVIDEND' in label or ('DIV' in label and '70' not in label and '30' not in label):
            p['dividend'] = clean(cells[1].get_text())
            if len(cells) >= 4 and 'TICKET' in clean(cells[2].get_text()).upper():
                p['tickets'] = clean(cells[3].get_text())

    return p


# ============================================================================
# FILE PARSING (orchestrator)
# ============================================================================

def parse_file(filepath):
    """Parse one HTML file -> (meeting_dict, [race_dicts], [runner_dicts], [exotic_dicts])."""
    log.info(f"Parsing: {filepath}")
    try:
        with open(filepath, 'r', encoding=INPUT_ENCODING, errors='replace') as f:
            html = f.read()
    except Exception as e:
        log.error(f"Cannot read {filepath}: {e}")
        return None, [], [], []

    soup = BeautifulSoup(html, 'html.parser')

    # Meeting header
    header = parse_meeting_header(soup)
    md, venue = header['meet_date'], header['venue']
    if not md:
        log.warning(f"  Could not extract date from {filepath}")

    # Classify tables
    cond_tbl, race_tbls, exo_tbls = classify_tables(soup)
    log.info(f"  Tables: {len(race_tbls)} race, {len(exo_tbls)} exotic, cond={'yes' if cond_tbl else 'no'}")

    # Conditions
    conditions = parse_conditions_table(cond_tbl)
    meeting = {**header, **conditions}

    # Races + runners
    all_races, all_runners = [], []
    for tbl in race_tbls:
        race, runners = parse_race_table(tbl, md, venue)
        if race:
            all_races.append(race)
            all_runners.extend(runners)

    # Exotics
    all_exotics = parse_exotic_tables(exo_tbls, md, venue)

    log.info(f"  Result: {len(all_races)} races, {len(all_runners)} runners, {len(all_exotics)} exotics")
    return meeting, all_races, all_runners, all_exotics


# ============================================================================
# CSV OUTPUT
# ============================================================================

def write_csvs(meetings, races, runners, exotics):
    """Write all data to CSV files."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    datasets = [
        (os.path.join(OUTPUT_DIR, MEETINGS_FILE), MEETING_COLS, meetings),
        (os.path.join(OUTPUT_DIR, RACES_FILE),    RACE_COLS,    races),
        (os.path.join(OUTPUT_DIR, RUNNERS_FILE),  RUNNER_COLS,  runners),
        (os.path.join(OUTPUT_DIR, EXOTICS_FILE),  EXOTIC_COLS,  exotics),
    ]

    for path, cols, data in datasets:
        need_header = (WRITE_MODE == 'w') or not os.path.exists(path) or os.path.getsize(path) == 0
        mode = 'w' if WRITE_MODE == 'w' else 'a'
        with open(path, mode, newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
            if need_header:
                w.writeheader()
            w.writerows(data)
        log.info(f"  Wrote {len(data):>4d} rows -> {path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Collect input files
    if os.path.isfile(INPUT_PATH):
        files = [INPUT_PATH]
    elif os.path.isdir(INPUT_PATH):
        files = sorted(glob.glob(os.path.join(INPUT_PATH, '*.html'))
                        + glob.glob(os.path.join(INPUT_PATH, '*.htm')))
    else:
        log.error(f"INPUT_PATH not found: {INPUT_PATH}")
        sys.exit(1)

    if not files:
        log.error(f"No .html/.htm files in: {INPUT_PATH}")
        sys.exit(1)

    log.info(f"Found {len(files)} file(s) to process")

    all_m, all_r, all_run, all_e = [], [], [], []
    for fp in files:
        meeting, races, runners, exotics = parse_file(fp)
        if meeting:
            all_m.append(meeting)
        all_r.extend(races)
        all_run.extend(runners)
        all_e.extend(exotics)

    write_csvs(all_m, all_r, all_run, all_e)
    log.info(f"\nDone — {len(all_m)} meetings, {len(all_r)} races, {len(all_run)} runners, {len(all_e)} exotics")


if __name__ == '__main__':
    main()
