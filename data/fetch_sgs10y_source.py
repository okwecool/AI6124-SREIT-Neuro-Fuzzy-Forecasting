"""Save a fixed official MAS daily 10-year SGS benchmark-yield response."""

from __future__ import annotations

import csv
import hashlib
import html
import http.cookiejar
import io
import json
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
URL = "https://eservices.mas.gov.sg/statistics/fdanet/BenchmarkPricesAndYields.aspx/1000"
RAW = ROOT / "raw/mas_sgs10y_202111_202509.html"
NORMALIZED = ROOT / "processed/mas_sgs10y_202111_202509.csv"
MANIFEST = ROOT / "source_sgs10y_manifest.json"
PREFIX = "ctl00$ContentPlaceHolder1$"
FIELDS = {
    PREFIX + "StartYearDropDownList": "2021",
    PREFIX + "StartMonthDropDownList": "11",
    PREFIX + "EndYearDropDownList": "2025",
    PREFIX + "EndMonthDropDownList": "9",
    PREFIX + "FrequencyDropDownList": "D",
    PREFIX + "TenYearBondYieldCheckBox": "on",
    PREFIX + "DisplayButton": "Display",
}


def parse_rows(page: str) -> list[tuple[str, float]]:
    """Fail closed if the MAS HTML table no longer has the expected layout."""
    tables = re.findall(r'<table class="results-table"[^>]*>.*?</table>', page, re.I | re.S)
    if len(tables) != 5 or not all("10-Year" in table for table in tables):
        raise ValueError("MAS response did not contain five expected annual 10-year tables")
    observations: list[tuple[str, float]] = []
    for table in tables:
        year = month = ""
        for row in re.findall(r"<tr\b[^>]*>(.*?)</tr>", table, re.I | re.S):
            cells = [html.unescape(re.sub(r"<[^>]*>", "", cell)).strip()
                     for cell in re.findall(r"<td\b[^>]*>(.*?)</td>", row, re.I | re.S)]
            if not cells:
                continue
            if len(cells) != 4:
                raise ValueError(f"Unexpected MAS data row: {cells!r}")
            year = cells[0] or year
            month = cells[1] or month
            day, value = cells[2], cells[3]
            if not year or not month or not day or not value:
                raise ValueError(f"Incomplete MAS data row: {cells!r}")
            parsed = datetime.strptime(f"{year} {month} {day}", "%Y %b %d").date()
            observations.append((parsed.isoformat(), float(value)))
    dates = [row[0] for row in observations]
    if (not observations or dates != sorted(set(dates))
            or dates[0] != "2021-11-01" or dates[-1] != "2025-09-30"
            or not 900 <= len(dates) <= 1100
            or any(not 0 <= value <= 10 for _, value in observations)):
        raise ValueError("Unexpected MAS date range, yield, or row count")
    return observations


def run() -> dict:
    if any(path.exists() for path in (RAW, NORMALIZED, MANIFEST)):
        raise FileExistsError("Fixed MAS source snapshot already exists")
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    headers = {"User-Agent": "Mozilla/5.0 (AI6124 academic course project)"}
    with opener.open(urllib.request.Request(URL, headers=headers), timeout=30) as response:
        initial = response.read().decode("utf-8")
    form = {}
    for key in ("__VIEWSTATE", "__VIEWSTATEGENERATOR", "__EVENTVALIDATION"):
        match = re.search(r'name="' + key + r'"[^>]*value="([^"]*)"', initial)
        if match is None:
            raise ValueError(f"MAS form lacks {key}")
        form[key] = html.unescape(match.group(1))
    form.update(FIELDS)
    request = urllib.request.Request(URL, urllib.parse.urlencode(form).encode(), headers=headers)
    with opener.open(request, timeout=30) as response:
        raw = response.read()
    observations = parse_rows(raw.decode("utf-8"))
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(("date", "sgs10y_yield_pct"))
    writer.writerows(observations)
    normalized = buffer.getvalue().encode("utf-8")
    manifest = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_url": URL,
        "query": {key.removeprefix(PREFIX): value for key, value in FIELDS.items()},
        "raw_file": str(RAW.relative_to(ROOT)),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "normalized_file": str(NORMALIZED.relative_to(ROOT)),
        "normalized_sha256": hashlib.sha256(normalized).hexdigest(),
        "rows": len(observations),
        "first_date": observations[0][0],
        "last_date": observations[-1][0],
        "note": "MAS table provides dates but no verified same-day publication time; modeling uses strictly prior dates.",
    }
    RAW.write_bytes(raw)
    NORMALIZED.write_bytes(normalized)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
