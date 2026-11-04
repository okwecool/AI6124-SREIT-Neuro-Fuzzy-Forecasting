"""Fetch a fixed Yahoo chart snapshot for the older Singapore S-REIT ETF CLR."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw/yahoo_clr_si_20171001_20260930.json"
MANIFEST = ROOT / "source_clr_manifest.json"


def run() -> dict:
    if RAW.exists() or MANIFEST.exists():
        raise FileExistsError("CLR source snapshot already exists")
    start = dt.datetime(2017, 10, 1, tzinfo=dt.timezone.utc)
    end = dt.datetime(2026, 10, 1, tzinfo=dt.timezone.utc)
    query = urllib.parse.urlencode({"period1": int(start.timestamp()),
        "period2": int(end.timestamp()), "interval": "1d", "events": "div,splits",
        "includeAdjustedClose": "true"})
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/CLR.SI?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (academic course project)"})
    with urllib.request.urlopen(request, timeout=30) as response:
        if "json" not in response.headers.get("Content-Type", "").lower():
            raise ValueError("Yahoo returned non-JSON content")
        body = response.read()
    payload = json.loads(body)
    if payload.get("chart", {}).get("error"):
        raise ValueError(payload["chart"]["error"])
    result = payload["chart"]["result"][0]
    meta = result["meta"]
    if meta.get("symbol") != "CLR.SI" or meta.get("currency") != "SGD" or meta.get("exchangeTimezoneName") != "Asia/Singapore":
        raise ValueError("Unexpected CLR identity, currency or exchange timezone")
    if len(result.get("timestamp", [])) < 2000:
        raise ValueError("CLR daily source is unexpectedly short")
    manifest = {"retrieved_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "symbol": "CLR.SI", "requested_period": "2017-10-01 to 2026-09-30 inclusive",
        "url": url, "file": str(RAW.relative_to(ROOT)), "sha256": hashlib.sha256(body).hexdigest(),
        "bytes": len(body), "rows": len(result["timestamp"]), "currency": meta["currency"],
        "exchange_timezone": meta["exchangeTimezoneName"]}
    RAW.write_bytes(body)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
