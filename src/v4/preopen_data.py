"""Build auditable New-York-session as-of inputs on the SRT forecast calendar."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.project_eval import Dataset
from src.v3.hybrid_training import inputs_and_residual


ROOT = Path(__file__).resolve().parents[2]
SYMBOLS = ("VNQ", "SPY")


def us_frame(symbol: str) -> pd.DataFrame:
    if symbol not in SYMBOLS:
        raise ValueError("Unregistered US source")
    manifest = json.loads((ROOT / f"data/source_us_{symbol.lower()}_manifest.json").read_text(encoding="utf-8"))
    path = ROOT / "data" / manifest["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("US source hash mismatch")
    source = json.loads(path.read_text(encoding="utf-8"))["chart"]["result"][0]
    if (source["meta"]["symbol"] != symbol or source["meta"]["currency"] != "USD"
            or source["meta"]["exchangeTimezoneName"] != "America/New_York"
            or source.get("events", {}).get("splits")):
        raise ValueError("Unexpected instrument/timezone/split")
    dates = pd.to_datetime(source["timestamp"], unit="s", utc=True).tz_convert("America/New_York").tz_localize(None).normalize()
    close = np.asarray(source["indicators"]["quote"][0]["close"], dtype=float)
    if dates.duplicated().any() or not dates.is_monotonic_increasing or not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError("Invalid source dates/closes")
    events = {}
    for event in source.get("events", {}).get("dividends", {}).values():
        date = pd.Timestamp(event["date"], unit="s", tz="UTC").tz_convert("America/New_York").tz_localize(None).normalize()
        if date not in dates or event["amount"] <= 0:
            raise ValueError("Invalid/unmatched dividend event")
        events[date] = events.get(date, 0) + event["amount"]
    dividend = np.array([events.get(date, 0.0) for date in dates])
    log_index = np.zeros(len(close))
    log_index[1:] = np.cumsum(np.log((close[1:] + dividend[1:]) / close[:-1]))
    # Localize each date separately before converting: never assume a fixed UTC offset.
    nominal_close = (dates + pd.Timedelta(hours=16)).tz_localize("America/New_York").tz_convert("UTC")
    return pd.DataFrame({"session_date": dates, "close_usd": close, "dividend_usd": dividend,
                         "log_total_return_index": log_index, "nominal_close_utc": nominal_close,
                         "assumed_available_at": nominal_close + pd.Timedelta(hours=2)})


def asof_values(cutoffs: pd.Series, source: pd.DataFrame) -> pd.DataFrame:
    left = pd.DataFrame({"cutoff": pd.to_datetime(cutoffs, utc=True)})
    joined = pd.merge_asof(left.sort_values("cutoff"), source.sort_values("assumed_available_at"),
                          left_on="cutoff", right_on="assumed_available_at", direction="backward")
    if joined.session_date.isna().any() or not (joined.assumed_available_at <= joined.cutoff).all():
        raise ValueError("Unavailable US source at a forecast cutoff")
    joined["return_since_previous_cutoff"] = joined.log_total_return_index.diff().fillna(0)
    joined["stale_hours"] = (joined.cutoff - joined.nominal_close_utc).dt.total_seconds() / 3600
    if (joined.stale_hours > 24 * 8).any():
        raise ValueError("Unexpected source staleness")
    return joined


def build(data: Dataset) -> tuple[dict, np.ndarray, pd.DataFrame]:
    base, residual = inputs_and_residual(data, "gross")
    # Only dates whose next SRT session is still in the development period.
    daily = data.daily[["date"]].copy()
    daily["target_session"] = daily.date.shift(-1)
    daily = daily.loc[daily.target_session.le(pd.Timestamp("2025-09-30"))].copy()
    matrices, alignment = {}, []
    for clock in ("close", "preopen"):
        local = daily.date + pd.Timedelta(hours=18) if clock == "close" else daily.target_session + pd.Timedelta(hours=8)
        cutoffs = local.dt.tz_localize("Asia/Singapore").dt.tz_convert("UTC")
        values = []
        for symbol in SYMBOLS:
            frame = asof_values(cutoffs, us_frame(symbol))
            frame["symbol"], frame["clock"] = symbol, clock
            frame["srt_forecast_date"] = daily.date.to_numpy()
            frame["srt_target_date"] = daily.target_session.to_numpy()
            alignment.append(frame)
            values.append(frame.return_since_previous_cutoff.to_numpy())
        daily_us = np.column_stack(values)
        x = np.full((len(base), 20, 8), np.nan, dtype=np.float32)
        eligible = data.index.target_date.le(pd.Timestamp("2025-09-30"))
        for row in data.index.loc[eligible].itertuples():
            x[row.window_id, :, :6] = base[row.window_id]
            x[row.window_id, :, 6:] = daily_us[row.daily_start_row:row.daily_end_row + 1]
        if not np.isfinite(x[eligible]).all():
            raise ValueError("Missing aligned development feature; do not drop dates")
        matrices[clock] = x
    return matrices, residual, pd.concat(alignment, ignore_index=True)
