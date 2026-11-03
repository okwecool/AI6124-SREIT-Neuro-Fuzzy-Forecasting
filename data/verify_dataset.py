"""Check the saved snapshot, model windows and chronological split contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / "source_manifest.json").read_text(encoding="utf-8"))
audit = json.loads((ROOT / "processed" / "audit.json").read_text(encoding="utf-8"))
for key in ("srt", "sti", "sora"):
    source = manifest[key]
    raw = (ROOT / source["file"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == source["sha256"], key
for repair in audit["srt_price_repairs"]:
    raw = (ROOT / repair["source_file"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == repair["source_sha256"]

daily = pd.read_csv(ROOT / "processed" / "srt_daily_aligned.csv", parse_dates=[
    "date", "sti_source_date", "sora_value_date_asof", "sora_publication_date_asof", "target_next_date"
])
index = pd.read_csv(ROOT / "processed" / "window_index.csv", parse_dates=[
    "window_start_date", "forecast_date", "target_date"
])
folds = pd.read_csv(ROOT / "processed" / "fold_assignments.csv")
arrays = np.load(ROOT / "processed" / "srt_windows_unscaled.npz", allow_pickle=False)
features = audit["feature_columns_in_X_order"]

assert len(daily) == 1222
assert len(index) == audit["complete_windows"] == arrays["X"].shape[0]
assert audit["srt_after_repairs"] == {"valid_close_rows": 1222, "missing_open_rows": 1}
assert arrays["X"].shape[1:] == (20, len(features))
assert list(arrays["feature_names"]) == features
assert np.isfinite(arrays["X"]).all()
assert np.isfinite(arrays["y_raw"]).all()
assert np.isfinite(arrays["y_total_return_candidate"]).all()
assert (daily["sora_publication_date_asof"] <= daily["date"]).all()
assert (daily["sti_source_date"] <= daily["date"]).all()
assert (index["target_date"] > index["forecast_date"]).all()

for position in (0, len(index) // 2, len(index) - 1):
    row = index.iloc[position]
    original = daily.iloc[int(row.daily_start_row):int(row.daily_end_row) + 1][features].to_numpy(dtype=np.float32)
    np.testing.assert_allclose(arrays["X"][position], original, rtol=1e-6, atol=1e-8)
    assert daily.iloc[int(row.daily_end_row)]["date"] == row.forecast_date
    assert daily.iloc[int(row.daily_end_row) + 1]["date"] == row.target_date

on_sep_30 = daily.loc[daily["date"] == "2026-09-30"].iloc[0]
assert str(on_sep_30.sora_value_date_asof.date()) == "2026-09-29"
assert str(on_sep_30.sora_publication_date_asof.date()) == "2026-09-30"
repair_day = daily.loc[daily["date"] == "2025-10-24"].iloc[0]
assert np.isclose(repair_day.srt_close, 0.801)
assert np.isclose(repair_day.srt_low, 0.799)
assert np.isclose(repair_day.srt_high, 0.803)
assert int(repair_day.srt_volume) == 152293
assert pd.isna(repair_day.srt_open)
assert repair_day.srt_close_source.startswith("SGTrades")
assert pd.read_csv(ROOT / "processed" / "srt_yahoo_normalized.csv").loc[
    lambda x: x.date == "2025-10-24", "close"
].isna().all()
assert len(index) == 1182
assert len(index.loc[index.period == "final_test"]) == 251
assert ((index.forecast_date == pd.Timestamp("2025-10-23")) &
        (index.target_date == pd.Timestamp("2025-10-24"))).any()
assert ((index.forecast_date == pd.Timestamp("2025-10-24")) &
        (index.target_date == pd.Timestamp("2025-10-27"))).any()

for fold_name, group in folds.groupby("fold"):
    train_ids = group.loc[group.role == "train", "window_id"]
    other_ids = group.loc[group.role != "train", "window_id"]
    assert set(train_ids).isdisjoint(other_ids), fold_name
    train_dates = index.iloc[train_ids]["target_date"]
    evaluation_dates = index.iloc[other_ids]["target_date"]
    assert train_dates.max() < evaluation_dates.min(), fold_name

print(json.dumps({
    "status": "pass",
    "windows": len(index),
    "X_shape": list(arrays["X"].shape),
    "folds": sorted(folds.fold.unique().tolist()),
    "source_hashes": "matched",
}, indent=2))
