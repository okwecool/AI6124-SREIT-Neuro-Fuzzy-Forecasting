"""Download only the two fixed development-period Yahoo ETF snapshots."""

import datetime as dt
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run():
    for symbol in ("VNQ", "SPY"):
        raw = ROOT / f"raw/yahoo_{symbol.lower()}_20210901_20250930.json"
        manifest = ROOT / f"source_us_{symbol.lower()}_manifest.json"
        query = urllib.parse.urlencode(
            {
                "period1": int(dt.datetime(2021, 9, 1, tzinfo=dt.timezone.utc).timestamp()),
                "period2": int(dt.datetime(2025, 10, 1, tzinfo=dt.timezone.utc).timestamp()),
                "interval": "1d",
                "events": "div,splits",
                "includeAdjustedClose": "false",
            }
        )
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?{query}"
        if raw.exists() or manifest.exists():
            if not (raw.exists() and manifest.exists()):
                raise FileExistsError("Incomplete prior snapshot; do not overwrite")
            old = json.loads(manifest.read_text(encoding="utf-8"))
            if old["url"] != url or old["sha256"] != hashlib.sha256(raw.read_bytes()).hexdigest():
                raise ValueError("Existing snapshot does not match request/hash")
            print(
                json.dumps({"symbol": symbol, "status": "existing verified", "rows": old["rows"]})
            )
            continue
        request = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (academic course project)"}
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            if "json" not in response.headers.get("Content-Type", "").lower():
                raise ValueError("Non-JSON response")
            body = response.read()
        payload = json.loads(body)
        if payload["chart"]["error"]:
            raise ValueError(payload["chart"]["error"])
        result = payload["chart"]["result"][0]
        meta = result["meta"]
        if (
            meta["symbol"] != symbol
            or meta["currency"] != "USD"
            or meta["exchangeTimezoneName"] != "America/New_York"
        ):
            raise ValueError("Unexpected source identity")
        if len(result["timestamp"]) < 1000 or result.get("events", {}).get("splits"):
            raise ValueError("Short source or an unhandled split")
        record = {
            "symbol": symbol,
            "url": url,
            "requested_period": "2021-09-01 to 2025-09-30",
            "retrieved_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "file": raw.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(body).hexdigest(),
            "rows": len(result["timestamp"]),
            "bytes": len(body),
            "currency": meta["currency"],
            "exchange_timezone": meta["exchangeTimezoneName"],
            "limitation": "Historical revised snapshot, not archived point-in-time vendor deliveries; bar availability must be explicitly modeled",
        }
        with raw.open("xb") as output:
            output.write(body)
        with manifest.open("x", encoding="utf-8") as output:
            output.write(json.dumps(record, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "symbol": symbol,
                    "status": "saved",
                    "rows": record["rows"],
                    "sha256": record["sha256"],
                }
            )
        )


if __name__ == "__main__":
    run()
