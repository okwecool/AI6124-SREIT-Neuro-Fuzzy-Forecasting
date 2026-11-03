"""Download a fixed historical snapshot for the SRT course project.

This uses the public Yahoo Finance chart feed for the two Yahoo tickers named in
the project and MAS's own Domestic Interest Rates CSV download form for SORA.
The original responses are saved unchanged in data/raw/.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

from lxml import html


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
START = dt.datetime(2021, 10, 1, tzinfo=dt.timezone.utc)
END_EXCLUSIVE = dt.datetime(2026, 10, 1, tzinfo=dt.timezone.utc)
MAS_URL = "https://eservices.mas.gov.sg/statistics/dir/"
HEADERS = {"User-Agent": "Mozilla/5.0 (academic course project)"}


def fetch(url: str, *, data: bytes | None = None, referer: str | None = None) -> tuple[bytes, str]:
    headers = dict(HEADERS)
    if referer:
        headers["Referer"] = referer
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read(), response.headers.get("Content-Type", "")


def download_yahoo(symbol: str, filename: str) -> dict:
    params = urllib.parse.urlencode({
        "period1": int(START.timestamp()),
        "period2": int(END_EXCLUSIVE.timestamp()),
        "interval": "1d",
        "events": "div,splits",
        "includeAdjustedClose": "true",
    })
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol, safe='')}?{params}"
    body, content_type = fetch(url)
    if "json" not in content_type.lower():
        raise ValueError(f"Yahoo returned {content_type} for {symbol}")
    payload = json.loads(body)
    if payload.get("chart", {}).get("error"):
        raise ValueError(f"Yahoo error for {symbol}: {payload['chart']['error']}")
    result = payload["chart"]["result"][0]
    if result["meta"].get("symbol") != symbol:
        raise ValueError(f"Yahoo returned wrong symbol for {symbol}")
    path = RAW / filename
    path.write_bytes(body)
    return {
        "symbol": symbol,
        "url": url,
        "file": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(body).hexdigest(),
        "bytes": len(body),
        "rows": len(result.get("timestamp", [])),
        "currency": result["meta"].get("currency"),
        "exchange_timezone": result["meta"].get("exchangeTimezoneName"),
    }


def download_mas() -> dict:
    page, content_type = fetch(MAS_URL)
    if "html" not in content_type.lower():
        raise ValueError(f"Unexpected MAS form content type: {content_type}")
    root = html.fromstring(page)
    forms = root.xpath('//form[@id="form1"]')
    if len(forms) != 1:
        raise ValueError("Could not locate MAS Domestic Interest Rates form")
    form = forms[0]
    payload = {
        element.get("name"): element.get("value", "")
        for element in form.xpath('.//input[@type="hidden"]')
    }
    sora_labels = root.xpath('//label[normalize-space(.)="SORA"]/@for')
    if len(sora_labels) != 1:
        raise ValueError("Could not locate SORA checkbox")
    checkbox = root.xpath(f'//input[@id="{sora_labels[0]}"]')
    if len(checkbox) != 1:
        raise ValueError("Could not locate SORA checkbox input")
    payload.update({
        "ctl00$ContentPlaceHolder1$StartYearDropDownList": "2021",
        "ctl00$ContentPlaceHolder1$EndYearDropDownList": "2026",
        "ctl00$ContentPlaceHolder1$StartMonthDropDownList": "10",
        "ctl00$ContentPlaceHolder1$EndMonthDropDownList": "9",
        checkbox[0].get("name"): "on",
        "ctl00$ContentPlaceHolder1$Button2": "Download",
    })
    body, response_type = fetch(
        MAS_URL,
        data=urllib.parse.urlencode(payload).encode("utf-8"),
        referer=MAS_URL,
    )
    if "csv" not in response_type.lower() or b"SORA Publication Date" not in body[:1000]:
        raise ValueError(f"Unexpected MAS download: {response_type}, {body[:100]!r}")
    path = RAW / "mas_sora_202110_202609.csv"
    path.write_bytes(body)
    return {
        "url": MAS_URL,
        "file": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(body).hexdigest(),
        "bytes": len(body),
        "content_type": response_type,
        "selection": "SORA; October 2021 through September 2026",
    }


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    sources = {
        "retrieved_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "requested_period": "2021-10-01 to 2026-09-30 inclusive",
        "srt": download_yahoo("SRT.SI", "yahoo_srt_si_20211001_20260930.json"),
        "sti": download_yahoo("^STI", "yahoo_sti_20211001_20260930.json"),
        "sora": download_mas(),
    }
    (ROOT / "source_manifest.json").write_text(
        json.dumps(sources, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(sources, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
