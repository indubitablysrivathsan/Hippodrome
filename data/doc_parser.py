import re
import os
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime


###############################################################
# UTILS
###############################################################

def parse_odds(odds_str):
    """
    Convert fractional odds like '5/2' -> implied probability
    """
    if not odds_str or "/" not in odds_str:
        return None

    try:
        num, den = odds_str.split("/")
        num = float(num)
        den = float(den)

        return den / (num + den)

    except:
        return None


def normalize_probs(probs):
    """
    Normalize probabilities removing bookmaker overround
    """
    total = sum(p for p in probs if p is not None)

    if total == 0:
        return [None] * len(probs)

    return [(p / total) if p else None for p in probs]


def extract_date(text):
    """
    Example:
    FOURTH DAY, Sunday 8th August 2010
    """
    m = re.search(r'(\d{1,2}\w{2}\s+\w+\s+\d{4})', text)
    if not m:
        return None

    try:
        return datetime.strptime(m.group(1), "%dth %B %Y")
    except:
        try:
            return datetime.strptime(m.group(1), "%d %B %Y")
        except:
            return None


def clean_text(x):
    return re.sub(r'\s+', ' ', x).strip()


###############################################################
# PARSE RACE HEADER
###############################################################

def parse_race_header(header_text):

    """
    Example header:

    The Acres Club Trophy Class IV; H'cap, Indian Horses rated 20 to 46 (0 to 19 eligible).
    Time: 4.40 P.M.
    (About) 1600 Metres.
    """

    race_name = None
    race_class = None
    distance = None

    # distance
    d = re.search(r'(\d+)\s*Metres', header_text)
    if d:
        distance = int(d.group(1))

    # class
    c = re.search(r'(Class\s+[IVX]+)', header_text)
    if c:
        race_class = c.group(1)

    # race name
    name = header_text.split("Class")[0].strip()
    race_name = name

    return race_name, race_class, distance


###############################################################
# MAIN SCRAPER
###############################################################

def parse_race_results(html_file, venue="MUMBAI"):

    with open(html_file, "r", encoding="utf8", errors="ignore") as f:
        soup = BeautifulSoup(f, "lxml")

    tables = soup.find_all("table", class_="contentTable")

    race_rows = []
    horse_rows = []

    race_id_counter = 1

    # extract date
    header = soup.find("div", class_="subHeading")

    date = None
    for s in soup.find_all("div", class_="subHeading"):
        if "DAY" in s.text:
            date = extract_date(s.text)
            break

    if date:
        date = date.date()

    going = None

    for t in tables:

        if "Placing" not in t.text:
            continue

        text = clean_text(t.text)

        ##################################################
        # race number
        ##################################################

        m = re.search(r'No\.\:\s*(\d+)', text)

        if not m:
            continue

        race_number = int(m.group(1))

        ##################################################
        # race header block
        ##################################################

        header_block = ""

        for span in t.find_all("span"):
            header_block += span.get_text(" ", strip=True) + " "

        race_name, race_class, distance = parse_race_header(header_block)

        ##################################################
        # horse rows
        ##################################################

        rows = t.find_all("tr")

        horse_data = []

        for r in rows:

            cols = r.find_all("td")

            if len(cols) < 8:
                continue

            placing = clean_text(cols[0].text)

            if placing == "WD":
                continue

            try:
                finish_pos = int(placing)
            except:
                continue

            horse = clean_text(cols[1].text.split("\n")[0])

            odds = clean_text(cols[5].text)

            horse_data.append({
                "finish_pos": finish_pos,
                "horse": horse,
                "odds": odds
            })

        if not horse_data:
            continue

        ##################################################
        # probabilities
        ##################################################

        raw_probs = [parse_odds(h["odds"]) for h in horse_data]

        norm_probs = normalize_probs(raw_probs)

        overround = sum(p for p in raw_probs if p)

        ##################################################
        # race uid
        ##################################################

        race_seq_id = f"{date}_{venue}_R{race_number}"

        ##################################################
        # horse rows
        ##################################################

        for i, h in enumerate(horse_data):

            horse_rows.append({

                "race_uid": race_seq_id,
                "race_id": race_id_counter,
                "horse_id": h["horse"],

                "odds": h["odds"],

                "implied_prob_raw": raw_probs[i],
                "implied_prob": norm_probs[i],

                "prob_rank": None,

                "finish_pos": h["finish_pos"],

                "margin": None,

                "overround": overround
            })

        ##################################################
        # rank probabilities
        ##################################################

        tmp = pd.DataFrame(horse_rows)

        mask = tmp.race_id == race_id_counter

        tmp.loc[mask, "prob_rank"] = tmp[mask]["implied_prob"].rank(ascending=False)

        horse_rows = tmp.to_dict("records")

        ##################################################
        # race row
        ##################################################

        race_rows.append({

            "race_id": race_id_counter,
            "date": date,
            "venue": venue,
            "going": going,
            "race_number": race_number,
            "race_seq_id": race_seq_id,
            "distance_m": distance,
            "class": race_class,
            "field_size": len(horse_data)
        })

        race_id_counter += 1

    races_df = pd.DataFrame(race_rows)
    horses_df = pd.DataFrame(horse_rows)

    return races_df, horses_df


###############################################################
# RUN SCRAPER
###############################################################

def scrape_folder(folder):

    race_tables = []
    horse_tables = []

    for f in os.listdir(folder):

        if not f.endswith(".html"):
            continue

        path = os.path.join(folder, f)

        r, h = parse_race_results(path)

        race_tables.append(r)
        horse_tables.append(h)

    races = pd.concat(race_tables, ignore_index=True)
    horses = pd.concat(horse_tables, ignore_index=True)

    return races, horses


###############################################################
# EXECUTE
###############################################################

folder = "raw_doc"

races, horses = scrape_folder(folder)

races.to_csv("races.csv", index=False)
horses.to_csv("horses.csv", index=False)

print("Done")