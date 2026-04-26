from pathlib import Path
import requests
import time
import pdfkit
from bs4 import BeautifulSoup

config = pdfkit.configuration(
    wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Research scraper for horse racing data)"
}

INPUT_DIR = Path("./raw_html/acceptances_2018-")
PDF_DIR = Path("./raw_doc/acceptances_2018-")

PDF_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------
# Extract download link from HTML file
# -----------------------------------------------------------
def extract_download_link(html_path):

    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    # find anchor with class "download"
    tag = soup.find("a", class_="download")

    if tag and tag.get("href"):
        return tag["href"]

    return None


# -----------------------------------------------------------
# Download + convert to PDF (same logic, but URL passed directly)
# -----------------------------------------------------------
def download_pdf_from_link(url, name):

    pdf_path = PDF_DIR / f"{name}.pdf"

    if pdf_path.exists():
        print("Skipping", name)
        return

    try:
        r = requests.get(url, headers=HEADERS, timeout=20)

        if r.status_code != 200:
            print("Failed to fetch", url)
            return

        temp_html = PDF_DIR / f"{name}.html"
        temp_html.write_text(r.text, encoding="utf-8")

        pdfkit.from_file(str(temp_html), str(pdf_path), configuration=config)

        temp_html.unlink()

        print("Saved PDF", name)

    except Exception as e:
        print("Error:", name, e)

    time.sleep(1)


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

        # use filename (without extension) as identifier
        name = file.stem

        download_pdf_from_link(link, name)


# -----------------------------------------------------------
# RUN
# -----------------------------------------------------------
if __name__ == "__main__":
    run_from_html_dir()