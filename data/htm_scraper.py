from pathlib import Path
import requests
import time
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Research scraper for horse racing data)"
}

INPUT_DIR = Path("./raw_html/race_day_report") # single .htm/.html file OR folder 09-22 2025-10-19
HTM_DIR = Path("./raw_doc/race_day_report")

HTM_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------
# Extract download link from HTML file
# -----------------------------------------------------------
def extract_download_link(html_path):

    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    tag = soup.find("a", class_="download")

    if tag and tag.get("href"):
        return tag["href"]

    return None

# -----------------------------------------------------------
# Decode encoding
# -----------------------------------------------------------
def robust_decode(r):
    # Step 1: let requests decide
    r.encoding = r.apparent_encoding
    text = r.text

    # Step 2: fix broken UTF-8 cases ONLY if needed
    if "â" in text or "Â" in text:
        try:
            repaired = text.encode("latin-1").decode("utf-8")
            # accept repair only if it improves
            if repaired.count("â") < text.count("â"):
                text = repaired
        except:
            pass

    # Step 3: normalize spacing
    text = text.replace("\xa0", " ")

    return text

# -----------------------------------------------------------
# Download HTM instead of PDF
# -----------------------------------------------------------
def download_htm_from_link(url, name):

    htm_path = HTM_DIR / f"{name}.htm"

    if htm_path.exists():
        print("Skipping", name)
        return

    try:
        r = requests.get(url, headers=HEADERS, timeout=20)

        if r.status_code != 200:
            print("Failed to fetch", url)
            return

        # FORCE decode (no guessing)
        text = robust_decode(r)

        # normalize problematic characters
        text = text.replace("\xa0", " ")   # NBSP → space

        htm_path.write_text(text, encoding="utf-8")

        print("Saved clean HTM", name)

    except Exception as e:
        print("Error:", name, e)

    time.sleep(0.5)
    
# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
def run_from_html_dir():

    html_files = list(INPUT_DIR.glob("*.html"))

    print("Found HTML files:", len(html_files))

    for file in html_files:

        link = extract_download_link(file)

        if not link:
            print("No download link in", file.name)
            continue

        name = file.stem

        download_htm_from_link(link, name)


# -----------------------------------------------------------
# RUN
# -----------------------------------------------------------
if __name__ == "__main__":
    run_from_html_dir()