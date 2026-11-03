"""Prepare leakage-aware daily and 20-day-window SRT forecasting datasets.

Run after download_sources.py. No scalers are fitted here: fit them separately
inside every training fold. The model's primary preliminary target is the next
SRT *raw close* log return; a dividend-inclusive candidate is also exported.
"""

from __future__ import annotations

import calendar
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
PROCESSED = ROOT / "processed"
SGT = ZoneInfo("Asia/Singapore")
WINDOW = 20
FEATURES = [
    "srt_log_return_raw",
    "srt_momentum_5",
    "srt_volatility_20",
    "sti_log_return_asof",
    "sora_rate_pct_asof",
    "sora_change_20_publications_pp",
]


def yahoo_frame(path: Path, expected_symbol: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload["chart"]["result"][0]
    meta = result["meta"]
    if meta.get("symbol") != expected_symbol or meta.get("exchangeTimezoneName") != "Asia/Singapore":
        raise ValueError(f"Unexpected Yahoo symbol/timezone in {path}")
    stamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    adjusted = result["indicators"].get("adjclose", [{}])[0].get("adjclose")
    if adjusted is None:
        adjusted = [None] * len(stamps)
    frame = pd.DataFrame({
        "date": pd.to_datetime([dt.datetime.fromtimestamp(int(t), SGT).date() for t in stamps]),
        "open": quote["open"],
        "high": quote["high"],
        "low": quote["low"],
        "close": quote["close"],
        "volume": quote["volume"],
        "adjclose_yahoo_diagnostic": adjusted,
    })
    if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
        raise ValueError(f"Duplicate or unsorted Yahoo dates: {expected_symbol}")
    events = result.get("events", {}).get("dividends", {})
    dividends = pd.DataFrame([
        {
            "date": dt.datetime.fromtimestamp(int(event["date"]), SGT).date(),
            "cash_dividend_sgd": float(event["amount"]),
            "event_id": event_id,
        }
        for event_id, event in events.items()
    ], columns=["date", "cash_dividend_sgd", "event_id"])
    if not dividends.empty:
        dividends["date"] = pd.to_datetime(dividends["date"])
        dividends = dividends.sort_values("date").reset_index(drop=True)
    info = {
        "symbol": expected_symbol,
        "source_rows": len(frame),
        "null_close_rows": int(frame["close"].isna().sum()),
        "null_close_dates": [str(x.date()) for x in frame.loc[frame["close"].isna(), "date"]],
        "first_date": str(frame["date"].min().date()),
        "last_date": str(frame["date"].max().date()),
        "dividend_events": len(dividends),
    }
    return frame, dividends, info


def mas_sora_frame(path: Path) -> tuple[pd.DataFrame, dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    header_positions = [i for i, row in enumerate(rows) if row and row[0] == "SORA Value Date"]
    if not header_positions:
        raise ValueError("MAS CSV does not contain a SORA header")
    header = rows[header_positions[0]]
    if len(header) < 5 or header[3:5] != ["SORA Publication Date", "SORA"]:
        raise ValueError(f"Unexpected MAS SORA header: {header}")
    year = None
    month = None
    records = []
    for row in rows[header_positions[0] + 1:]:
        if len(row) < 5 or not row[2].strip().isdigit():
            continue
        if row[0].strip().isdigit():
            year = int(row[0])
        if row[1].strip():
            month = list(calendar.month_abbr).index(row[1].strip().title())
        if year is None or month is None:
            raise ValueError(f"Cannot reconstruct MAS value date: {row}")
        value_date = dt.date(year, month, int(row[2]))
        publication_date = dt.datetime.strptime(row[3].strip(), "%d %b %Y").date()
        records.append({
            "value_date": value_date,
            "publication_date": publication_date,
            "sora_rate_pct": float(row[4]),
        })
    frame = pd.DataFrame(records).sort_values("publication_date").reset_index(drop=True)
    if frame.empty or frame["value_date"].duplicated().any() or frame["publication_date"].duplicated().any():
        raise ValueError("Missing or duplicate MAS SORA rows")
    if not (frame["publication_date"] > frame["value_date"]).all():
        raise ValueError("MAS publication date must follow value date")
    frame["value_date"] = pd.to_datetime(frame["value_date"])
    frame["publication_date"] = pd.to_datetime(frame["publication_date"])
    frame["sora_change_20_publications_pp"] = frame["sora_rate_pct"].diff(20)
    info = {
        "source_rows": len(frame),
        "first_value_date": str(frame["value_date"].min().date()),
        "last_value_date": str(frame["value_date"].max().date()),
        "first_publication_date": str(frame["publication_date"].min().date()),
        "last_publication_date": str(frame["publication_date"].max().date()),
        "missing_rate_rows": int(frame["sora_rate_pct"].isna().sum()),
    }
    return frame, info


def apply_srt_price_repair(frame: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, list[dict]]:
    """Repair a missing Yahoo trading row using an independently checked quote.

    The Yahoo-normalized export remains untouched. The supplemental record is
    applied only to the modelling copy, with provenance carried into the audit.
    """
    repairs = pd.read_csv(path, dtype={"symbol": str, "currency": str})
    required = {"date", "symbol", "currency", "close", "low", "high", "volume", "source_url", "corroboration_url"}
    if set(repairs.columns) < required or len(repairs) != 1:
        raise ValueError("Expected one complete SRT supplemental quote")
    repair = repairs.iloc[0]
    if repair["date"] != "2025-10-24" or repair["symbol"] != "SRT.SI" or repair["currency"] != "SGD":
        raise ValueError("Supplemental quote has unexpected identity")
    close, low, high = (float(repair[name]) for name in ("close", "low", "high"))
    volume = int(repair["volume"])
    if not (0 < low <= close <= high and volume > 0):
        raise ValueError("Supplemental quote fails range or volume check")
    result = frame.copy()
    match = result["date"].eq(pd.Timestamp(repair["date"]))
    if match.sum() != 1 or result.loc[match, "close"].notna().any():
        raise ValueError("Supplemental quote does not match exactly one missing Yahoo close")
    result["close_source"] = "Yahoo Finance"
    for name, value in {"close": close, "low": low, "high": high, "volume": volume}.items():
        if result.loc[match, name].notna().any():
            raise ValueError(f"Yahoo {name} was already present on repair date")
        result.loc[match, name] = value
    result.loc[match, "close_source"] = "SGTrades; cross-checked with The Business Times"
    record = {
        "date": str(repair["date"]),
        "close_sgd": close,
        "low_sgd": low,
        "high_sgd": high,
        "volume": volume,
        "open_sgd": None,
        "source_url": str(repair["source_url"]),
        "corroboration_url": str(repair["corroboration_url"]),
        "source_file": str(path.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    return result, [record]


def assign_period(target: dt.date) -> str:
    if dt.date(2024, 4, 1) <= target <= dt.date(2024, 9, 30):
        return "validation_1"
    if dt.date(2024, 10, 1) <= target <= dt.date(2025, 3, 31):
        return "validation_2"
    if dt.date(2025, 4, 1) <= target <= dt.date(2025, 9, 30):
        return "validation_3"
    if dt.date(2025, 10, 1) <= target <= dt.date(2026, 9, 30):
        return "final_test"
    return "development_training"


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    srt, dividends, srt_info = yahoo_frame(RAW / "yahoo_srt_si_20211001_20260930.json", "SRT.SI")
    sti, _, sti_info = yahoo_frame(RAW / "yahoo_sti_20211001_20260930.json", "^STI")
    sora, sora_info = mas_sora_frame(RAW / "mas_sora_202110_202609.csv")
    srt.to_csv(PROCESSED / "srt_yahoo_normalized.csv", index=False)
    sti.to_csv(PROCESSED / "sti_yahoo_normalized.csv", index=False)
    sora.to_csv(PROCESSED / "sora_mas_normalized.csv", index=False)
    dividends.to_csv(PROCESSED / "srt_dividends_yahoo.csv", index=False)

    srt, srt_price_repairs = apply_srt_price_repair(
        srt, RAW / "srt_2025-10-24_sgtrades_excerpt.csv"
    )
    srt.to_csv(PROCESSED / "srt_repaired_normalized.csv", index=False)

    # Keep a date with a missing SRT close on the modelling calendar. Dropping
    # it would silently relabel a multi-day return as a next-trading-day target.
    srt = srt.copy().reset_index(drop=True)
    if (srt["close"] <= 0).any():
        raise ValueError("SRT close must be positive")
    srt = srt.rename(columns={name: f"srt_{name}" for name in srt.columns if name != "date"})
    srt["srt_log_return_raw"] = np.log(srt["srt_close"]).diff()
    srt["srt_momentum_5"] = np.log(srt["srt_close"] / srt["srt_close"].shift(5))
    srt["srt_volatility_20"] = srt["srt_log_return_raw"].rolling(20, min_periods=20).std(ddof=0)

    # STI levels are observable at the Singapore close on the same date. If
    # its calendar has a gap, carry only the last *known* level and record age.
    sti = sti.loc[sti["close"].notna(), ["date", "close"]].rename(
        columns={"date": "sti_source_date", "close": "sti_close_asof"}
    )
    aligned = pd.merge_asof(
        srt.sort_values("date"),
        sti.sort_values("sti_source_date"),
        left_on="date", right_on="sti_source_date", direction="backward",
    )
    aligned["sti_stale_calendar_days"] = [
        (origin - source).days if pd.notna(source) else np.nan
        for origin, source in zip(aligned["date"], aligned["sti_source_date"])
    ]
    aligned["sti_log_return_asof"] = np.log(aligned["sti_close_asof"]).diff()

    # MAS explicitly supplies the publication date. By t's post-close
    # forecast, only records with publication_date <= t are eligible.
    aligned = pd.merge_asof(
        aligned.sort_values("date"),
        sora.rename(columns={
            "value_date": "sora_value_date_asof",
            "publication_date": "sora_publication_date_asof",
            "sora_rate_pct": "sora_rate_pct_asof",
        }).sort_values("sora_publication_date_asof"),
        left_on="date", right_on="sora_publication_date_asof", direction="backward",
    )
    if not (aligned["sora_publication_date_asof"].dropna() <= aligned.loc[
        aligned["sora_publication_date_asof"].notna(), "date"
    ]).all():
        raise AssertionError("Future SORA publication leaked into an input row")
    if not (aligned["sti_source_date"].dropna() <= aligned.loc[
        aligned["sti_source_date"].notna(), "date"
    ]).all():
        raise AssertionError("Future STI close leaked into an input row")

    # Yahoo's adjusted close is retained only for inspection. Actual cash
    # distributions appear as a separate candidate total-return target.
    dividend_by_date = dividends.groupby("date")["cash_dividend_sgd"].sum() if not dividends.empty else pd.Series(dtype=float)
    aligned["srt_dividend_sgd_on_date"] = aligned["date"].map(dividend_by_date).fillna(0.0)
    aligned["target_next_date"] = aligned["date"].shift(-1)
    aligned["target_next_close_raw"] = aligned["srt_close"].shift(-1)
    aligned["target_next_dividend_sgd"] = aligned["srt_dividend_sgd_on_date"].shift(-1)
    aligned["target_next_log_return_raw"] = np.log(
        aligned["target_next_close_raw"] / aligned["srt_close"]
    )
    aligned["target_next_log_total_return_candidate"] = np.log(
        (aligned["target_next_close_raw"] + aligned["target_next_dividend_sgd"])
        / aligned["srt_close"]
    )
    aligned.to_csv(PROCESSED / "srt_daily_aligned.csv", index=False, float_format="%.10g")

    # A window is eligible only if all 20 input rows are finite and the next
    # SRT close exists. No full-sample scaling or future price enters X.
    feature_values = aligned[FEATURES].to_numpy(dtype=np.float64)
    windows = []
    rows = []
    for end_index in range(WINDOW - 1, len(aligned) - 1):
        start_index = end_index - WINDOW + 1
        x = feature_values[start_index:end_index + 1]
        target = aligned.iloc[end_index]
        if not np.isfinite(x).all() or not math.isfinite(target["target_next_log_return_raw"]):
            continue
        origin_date = target["date"].date()
        next_date = target["target_next_date"].date()
        if next_date <= origin_date:
            raise AssertionError("Non-forward target")
        windows.append(x.astype(np.float32))
        rows.append({
            "window_id": len(rows),
            "daily_start_row": start_index,
            "daily_end_row": end_index,
            "window_start_date": aligned.iloc[start_index]["date"].date(),
            "forecast_date": origin_date,
            "target_date": next_date,
            "target_next_log_return_raw": target["target_next_log_return_raw"],
            "target_next_log_total_return_candidate": target["target_next_log_total_return_candidate"],
            "period": assign_period(next_date),
        })
    if not windows:
        raise ValueError("No complete 20-day windows")
    window_index = pd.DataFrame(rows)
    window_index.to_csv(PROCESSED / "window_index.csv", index=False, float_format="%.10g")
    fold_definitions = [
        ("validation_1", "2024-03-31", "2024-04-01", "2024-09-30"),
        ("validation_2", "2024-09-30", "2024-10-01", "2025-03-31"),
        ("validation_3", "2025-03-31", "2025-04-01", "2025-09-30"),
        ("final_holdout", "2025-09-30", "2025-10-01", "2026-09-30"),
    ]
    fold_rows = []
    fold_counts = {}
    for fold_name, train_end, evaluation_start, evaluation_end in fold_definitions:
        train = window_index[window_index["target_date"].astype(str) <= train_end]
        evaluation = window_index[
            (window_index["target_date"].astype(str) >= evaluation_start)
            & (window_index["target_date"].astype(str) <= evaluation_end)
        ]
        if train.empty or evaluation.empty or train["target_date"].max() >= evaluation["target_date"].min():
            raise AssertionError(f"Invalid chronological fold: {fold_name}")
        role = "test" if fold_name == "final_holdout" else "validation"
        fold_rows.extend({"fold": fold_name, "window_id": int(i), "role": "train"} for i in train["window_id"])
        fold_rows.extend({"fold": fold_name, "window_id": int(i), "role": role} for i in evaluation["window_id"])
        fold_counts[fold_name] = {"train": len(train), role: len(evaluation)}
    pd.DataFrame(fold_rows).to_csv(PROCESSED / "fold_assignments.csv", index=False)
    x_array = np.stack(windows)
    np.savez_compressed(
        PROCESSED / "srt_windows_unscaled.npz",
        X=x_array,
        y_raw=window_index["target_next_log_return_raw"].to_numpy(dtype=np.float32),
        y_total_return_candidate=window_index["target_next_log_total_return_candidate"].to_numpy(dtype=np.float32),
        forecast_date=window_index["forecast_date"].astype(str).to_numpy(),
        target_date=window_index["target_date"].astype(str).to_numpy(),
        feature_names=np.asarray(FEATURES),
    )

    audit = {
        "source_manifest": "source_manifest.json",
        "target_definition": "Next SRT raw closing-price log return; dividend-inclusive candidate exported separately",
        "feature_columns_in_X_order": FEATURES,
        "window_length_srt_trading_days": WINDOW,
        "scaling": "none; fit only on training partition inside each fold",
        "srt": srt_info,
        "srt_price_repairs": srt_price_repairs,
        "srt_after_repairs": {
            "valid_close_rows": int(srt["srt_close"].notna().sum()),
            "missing_open_rows": int(srt["srt_open"].isna().sum()),
        },
        "sti": sti_info,
        "sora": sora_info,
        "aligned_rows_total": len(aligned),
        "aligned_rows_with_valid_srt_close": int(aligned["srt_close"].notna().sum()),
        "sti_same_day_rows": int((aligned["sti_stale_calendar_days"] == 0).sum()),
        "sti_stale_rows": int((aligned["sti_stale_calendar_days"] > 0).sum()),
        "aligned_rows_without_sora": int(aligned["sora_rate_pct_asof"].isna().sum()),
        "first_complete_feature_date": str(aligned.loc[aligned[FEATURES].notna().all(axis=1), "date"].min().date()),
        "complete_windows": len(window_index),
        "first_window_forecast_date": str(window_index["forecast_date"].min()),
        "last_window_forecast_date": str(window_index["forecast_date"].max()),
        "window_counts_by_period": window_index["period"].value_counts().to_dict(),
        "fold_counts": fold_counts,
        "raw_target_nonfinite_windows": int(window_index["target_next_log_return_raw"].isna().sum()),
        "dividend_dates_within_srt_calendar": int((aligned["srt_dividend_sgd_on_date"] > 0).sum()),
        "raw_target_abs_over_10pct": int((window_index["target_next_log_return_raw"].abs() > 0.10).sum()),
    }
    (PROCESSED / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=int) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False, default=int))


if __name__ == "__main__":
    main()
