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

    # 1. Try exact text match
    tag = soup.find("a", string=lambda x: x and "Download Race Day Report" in x)

    if tag and tag.get("href"):
        return tag["href"]

    # 2. Fallback: any .htm/.html link
    for a in soup.find_all("a", href=True):
        href = a["href"].lower()
        if href.endswith(".htm") or href.endswith(".html"):
            return a["href"]

    return None

# -----------------------------------------------------------
# Decode encoding
# -----------------------------------------------------------

def decode_response(r):
    text = r.content.decode("cp1252", errors="replace")
    text = text.replace("\xa0", " ")  # fix NBSP
    return text

def normalize_text(text):
    # Fix common mojibake patterns first
    replacements = {
        "â€“": "–",
        "â€”": "—",
        "â€˜": "‘",
        "â€™": "’",
        "â€œ": "“",
        "â€\x9d": "”",
        "Â ": " ",
        "Â": "",
    }

    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # Fix cp1252 control characters
    text = text.replace("\x96", "–")
    text = text.replace("\x97", "—")
    text = text.replace("\x92", "’")

    # NBSP
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

        text = r.content.decode("cp1252", errors="replace")
        text = text.replace("\xa0", " ")
        for i, ch in enumerate(text):
                snippet = text[i:i+5]
                print("VISIBLE:", snippet)
                print("REPR   :", repr(snippet))
                print("CODES  :", [hex(ord(c)) for c in snippet])
                break

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