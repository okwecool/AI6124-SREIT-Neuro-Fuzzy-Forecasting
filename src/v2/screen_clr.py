"""Screen one same-day CLR return for incremental SRT residual prediction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.prepare_dataset import yahoo_frame
from src.project_eval import Dataset, PREDICTION_COLUMNS, VALIDATION_FOLDS
from src.v2.run_residual_fuzzy import ResidualDataset
from src.v2.screen_distributions import EVENTS, announced_adjustments
from src.v2.screen_ohlcv import fit_one


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs/v2_clr_screen_contract_2026-11-04.json"
CLR_RAW = ROOT / "data/raw/yahoo_clr_si_20171001_20260930.json"
OUT = ROOT / "output/v2/clr_screen"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aligned_clr_windows(data: Dataset) -> tuple[np.ndarray, pd.DataFrame]:
    clr, _, _ = yahoo_frame(CLR_RAW, "CLR.SI")
    if clr.date.duplicated().any():
        raise ValueError("Duplicate CLR quote dates")
    aligned = data.daily[["date"]].merge(
        clr[["date", "close"]].rename(columns={"close": "clr_close"}),
        on="date", how="left", validate="one_to_one")
    development = aligned.date.le(pd.Timestamp("2025-09-30"))
    if aligned.loc[development, "clr_close"].isna().any() or (aligned.loc[development, "clr_close"] <= 0).any():
        raise ValueError("CLR is not present on every SRT development trading date")
    aligned["clr_log_return_on_srt_calendar"] = np.log(aligned.clr_close / aligned.clr_close.shift(1))
    values = aligned.clr_log_return_on_srt_calendar.to_numpy(dtype=np.float32)
    starts = data.index.daily_start_row.to_numpy(dtype=int)
    ends = data.index.daily_end_row.to_numpy(dtype=int)
    feature = np.stack([values[start:end + 1] for start, end in zip(starts, ends)])[:, :, None]
    x = np.concatenate((data.x, feature), axis=-1)
    for fold in VALIDATION_FOLDS:
        used = np.concatenate((data.ids(fold, "train"), data.ids(fold, "validation")))
        if not np.isfinite(x[used]).all():
            raise ValueError(f"Nonfinite CLR feature in {fold}")
    return x, aligned.loc[development].copy()


def run() -> dict:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refuse to overwrite results: {OUT}")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if tuple(contract["folds"]) != VALIDATION_FOLDS or tuple(contract["alpha_grid"]) != (10.0, 100.0, 1000.0):
        raise ValueError("Unexpected CLR screen contract")
    original = Dataset(ROOT)
    residual = ResidualDataset(ROOT)
    x_clr, aligned = aligned_clr_windows(residual)
    adjustment = announced_adjustments(original, pd.read_csv(EVENTS)).set_index("window_id").adjustment
    rows, manifests = [], []
    for model_id, x in (("RidgeResidual6", residual.x), ("RidgeResidualCLR", x_clr)):
        for fold in VALIDATION_FOLDS:
            ids, residual_pred, manifest = fit_one(residual, x, fold, tuple(contract["alpha_grid"]))
            raw_pred = residual_pred + adjustment.loc[ids].to_numpy()
            rows.append(pd.DataFrame([(model_id, "all", 0, fold, 0, int(i), float(v))
                for i, v in zip(ids, raw_pred)], columns=PREDICTION_COLUMNS))
            manifest.update({"model_id": model_id, "features": x.shape[-1]})
            manifests.append(manifest)
    prediction = pd.concat(rows, ignore_index=True)
    original.validate_predictions(prediction)
    schedule = pd.read_csv(ROOT / "output/v2/announced_distribution_screen/predictions.csv")
    schedule = schedule.loc[schedule.model_id.eq("DivSchedule")]
    compare = pd.concat([schedule, prediction], ignore_index=True)
    pooled = original.pooled_validation_scores(compare).sort_values("return_rmse")
    by_fold = original.forecast_scores(compare).sort_values(["model_id", "fold"])
    scores = pooled.set_index("model_id")
    clr_score = scores.loc["RidgeResidualCLR"]
    base_score = scores.loc["RidgeResidual6"]
    schedule_score = scores.loc["DivSchedule"]
    joined = by_fold.loc[by_fold.model_id.eq("RidgeResidualCLR"), ["fold", "return_rmse"]].merge(
        by_fold.loc[by_fold.model_id.eq("DivSchedule"), ["fold", "return_rmse"]],
        on="fold", suffixes=("_clr", "_schedule"), validate="one_to_one")
    wins = int((joined.return_rmse_clr < joined.return_rmse_schedule).sum())
    decision = {"experiment_id": contract["experiment_id"],
        "schedule_rmse": float(schedule_score.return_rmse),
        "ridge_residual6_rmse": float(base_score.return_rmse),
        "ridge_residual_clr_rmse": float(clr_score.return_rmse),
        "clr_vs_6_rmse_gain_pct": float(100 * (1 - clr_score.return_rmse / base_score.return_rmse)),
        "clr_vs_schedule_rmse_gain_pct": float(100 * (1 - clr_score.return_rmse / schedule_score.return_rmse)),
        "fold_wins_vs_schedule": wins,
        "passes_screen": bool(clr_score.return_rmse <= 0.99 * base_score.return_rmse and
                              clr_score.return_rmse <= 0.99 * schedule_score.return_rmse and
                              wins >= 2 and clr_score.return_mae <= 1.01 * schedule_score.return_mae)}
    OUT.mkdir(parents=True, exist_ok=False)
    aligned.to_csv(OUT / "clr_development_alignment.csv", index=False)
    prediction.to_csv(OUT / "predictions.csv", index=False)
    pooled.to_csv(OUT / "comparison_pooled.csv", index=False)
    by_fold.to_csv(OUT / "comparison_by_fold.csv", index=False)
    (OUT / "manifests.json").write_text(json.dumps({"contract_sha256": sha256(CONTRACT),
        "source_sha256": sha256(Path(__file__)), "clr_source_sha256": sha256(CLR_RAW),
        "srt_data_sha256": sha256(ROOT / "data/processed/srt_windows_unscaled.npz"),
        "folds_sha256": sha256(ROOT / "data/processed/fold_assignments.csv"),
        "fits": manifests}, indent=2), encoding="utf-8")
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return decision


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
