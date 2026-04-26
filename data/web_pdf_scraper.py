from pathlib import Path
import requests
import time
import pdfkit   # NEW

ARCHIVE_API = "https://rwitc.com/new/lib/fetchArchives.php"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Research scraper for horse racing data)"
}

PDF_DIR = Path("pdf/acceptance_pages")   # CHANGED
PDF_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------
# Get race dates from archive API (UNCHANGED)
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
# Build ACCEPTANCE URL (CHANGED)
# -----------------------------------------------------------
def build_acceptance_url(date):

    return f"https://rwitc.com/run_races/Acceptance_{date}.htm"


# -----------------------------------------------------------
# Download + convert to PDF (CHANGED)
# -----------------------------------------------------------
def download_acceptance_pdf(date):

    url = build_acceptance_url(date)
    pdf_path = PDF_DIR / f"{date}.pdf"

    if pdf_path.exists():
        print("Skipping", date)
        return

    try:
        r = requests.get(url, headers=HEADERS, timeout=20)

        if r.status_code != 200:
            print("Missing page", date)
            return

        # TEMP HTML (needed for pdfkit)
        temp_html = PDF_DIR / f"{date}.html"
        temp_html.write_text(r.text, encoding="utf-8")

        # Convert → PDF
        pdfkit.from_file(str(temp_html), str(pdf_path))

        temp_html.unlink()  # cleanup

        print("Saved PDF", date)

    except Exception as e:
        print("Download failed", date, e)

    time.sleep(1)


# -----------------------------------------------------------
# MAIN PIPELINE (MINIMAL CHANGE)
# -----------------------------------------------------------
def run_scraper():

    for year in range(2023, 2027):

        print("\nChecking year", year)

        start = f"{year}-01-01"
        end = f"{year}-12-31"

        race_dates = get_race_dates(start, end)

        print("Race days found:", len(race_dates))

        for d in race_dates:
            download_acceptance_pdf(d)   # CHANGED


# -----------------------------------------------------------
# RUN
# -----------------------------------------------------------
if __name__ == "__main__":
    run_scraper()