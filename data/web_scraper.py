from pathlib import Path
import requests
import time

ARCHIVE_API = "https://rwitc.com/new/lib/fetchArchives.php"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Research scraper for horse racing data)"
}

RAW_DIR = Path("raw_html")
RAW_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------
# Get race dates from archive API
# -----------------------------------------------------------
def get_race_dates(start, end):

    params = {
        "start": start,
        "end": end
    }

    try:
        r = requests.get(ARCHIVE_API, params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()

    except Exception as e:
        print("Archive API error:", start, end, e)
        return []

    race_dates = []

    for item in data:

        if item.get("className") == "raceresults":

            race_dates.append(item["start"][:10])

    return sorted(set(race_dates))


# -----------------------------------------------------------
# Build result URL
# -----------------------------------------------------------
def build_result_url(date):

    return f"https://rwitc.com/new/erp_raceresult.php?date={date}"


# -----------------------------------------------------------
# Download individual result page
# -----------------------------------------------------------
def download_result_page(date):

    url = build_result_url(date)

    file_path = RAW_DIR / f"{date}.html"

    if file_path.exists():
        print("Skipping", date)
        return

    try:
        r = requests.get(url, headers=HEADERS, timeout=20)

        if r.status_code != 200:
            print("Missing page", date)
            return

        file_path.write_text(r.text, encoding="utf-8")

        print("Saved", date)

    except Exception as e:
        print("Download failed", date, e)

    time.sleep(1)  # polite delay


# -----------------------------------------------------------
# MAIN PIPELINE
# -----------------------------------------------------------
def run_scraper():

    for year in range(2008, 2027):

        print("\nChecking year", year)

        start = f"{year}-01-01"
        end = f"{year}-12-31"

        race_dates = get_race_dates(start, end)

        print("Race days found:", len(race_dates))

        for d in race_dates:

            download_result_page(d)


# -----------------------------------------------------------
# RUN
# -----------------------------------------------------------
if __name__ == "__main__":

    run_scraper()