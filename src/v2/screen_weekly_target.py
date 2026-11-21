"""Explore one predeclared five-trading-day SRT price target on development data."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.project_eval import Dataset, Standardizer, VALIDATION_FOLDS
from src.v2.baselines import fit_ridge, predict_ridge


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs/v2_weekly_target_screen_contract_2026-11-04.json"
OUT = ROOT / "output/v2/weekly_target_screen"
HORIZON = 5


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_target(data: Dataset, cutoff: pd.Timestamp) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate labels from observed future closes, never as model inputs."""
    ends = data.index.daily_end_row.to_numpy(dtype=int)
    close = data.daily.srt_close.to_numpy(dtype=float)
    dates = data.daily.date.to_numpy()
    target_rows = ends + HORIZON
    valid = target_rows < len(data.daily)
    valid &= dates[np.minimum(target_rows, len(dates) - 1)] <= np.datetime64(cutoff)
    safe_target_rows = np.minimum(target_rows, len(close) - 1)
    valid &= np.isfinite(close[ends]) & np.isfinite(close[safe_target_rows])
    valid &= (close[ends] > 0) & (close[safe_target_rows] > 0)
    y = np.full(len(data.index), np.nan, dtype=np.float64)
    y[valid] = np.log(close[target_rows[valid]] / close[ends[valid]])
    return y, valid, dates[safe_target_rows]


def purge(data: Dataset, ids: np.ndarray, first_eval_id: int, label_end_dates: np.ndarray) -> np.ndarray:
    first_forecast = np.datetime64(data.index.forecast_date.iloc[first_eval_id])
    kept = ids[label_end_dates[ids] < first_forecast]
    if not len(kept) or label_end_dates[kept].max() >= first_forecast:
        raise ValueError("Five-day labels leak into evaluation forecast period")
    return kept


def split_fold(data: Dataset, valid: np.ndarray, label_end_dates: np.ndarray, fold: str):
    if fold not in VALIDATION_FOLDS:
        raise ValueError("Only development folds are permitted")
    raw_eval = data.ids(fold, "validation")
    eval_ids = raw_eval[valid[raw_eval]]
    if not len(eval_ids):
        raise ValueError("No eligible development evaluation labels")
    raw_train = data.ids(fold, "train")
    train_ids = purge(data, raw_train[valid[raw_train]], int(eval_ids[0]), label_end_dates)
    stop_count = math.ceil(0.15 * len(train_ids))
    stop_ids = train_ids[-stop_count:]
    inner_fit_ids = purge(data, train_ids[:-stop_count], int(stop_ids[0]), label_end_dates)
    if not len(inner_fit_ids):
        raise ValueError("No inner fit rows after label purge")
    return inner_fit_ids, stop_ids, train_ids, eval_ids


def fit_and_predict(x: np.ndarray, y: np.ndarray, fit_ids: np.ndarray, eval_ids: np.ndarray,
                    alpha: float) -> tuple[np.ndarray, float]:
    scaler = Standardizer.fit(x[fit_ids], y[fit_ids])
    fitted = fit_ridge(scaler.x_transform(x[fit_ids]), scaler.y_transform(y[fit_ids]), alpha)
    prediction = scaler.y_inverse(predict_ridge(scaler.x_transform(x[eval_ids]), fitted))
    return prediction, float(np.linalg.norm(fitted[0]))


def run() -> dict:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refuse to overwrite results: {OUT}")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if tuple(contract["folds"]) != VALIDATION_FOLDS or tuple(contract["alpha_grid"]) != (10.0, 100.0, 1000.0):
        raise ValueError("Unexpected weekly screen contract")
    data = Dataset(ROOT)
    cutoff = pd.Timestamp(contract["development_cutoff"])
    y, valid, label_end_dates = make_target(data, cutoff)
    c = data.daily.srt_close.to_numpy(dtype=float)
    ends = data.index.daily_end_row.to_numpy(dtype=int)
    momentum = np.log(c[ends] / c[ends - HORIZON])
    if not np.isfinite(momentum).all():
        raise ValueError("Nonfinite past five-day momentum")
    records, manifests = [], []
    for fold in VALIDATION_FOLDS:
        inner_fit, stop, outer_train, eval_ids = split_fold(data, valid, label_end_dates, fold)
        trials = []
        for alpha in contract["alpha_grid"]:
            stop_pred, _ = fit_and_predict(data.x, y, inner_fit, stop, alpha)
            trials.append({"alpha": alpha, "inner_stop_rmse": float(np.sqrt(np.mean((stop_pred - y[stop]) ** 2)))})
        selected = min(trials, key=lambda row: (row["inner_stop_rmse"], -row["alpha"]))
        ridge_pred, weight_l2 = fit_and_predict(data.x, y, outer_train, eval_ids, selected["alpha"])
        models = {"Z0_5day": np.zeros(len(eval_ids)),
                  "MeanTrain_5day": np.full(len(eval_ids), float(y[outer_train].mean())),
                  "Momentum_5day": momentum[eval_ids], "Ridge_5day": ridge_pred}
        for model, pred in models.items():
            for i, value in zip(eval_ids, pred):
                records.append({"model": model, "fold": fold, "window_id": int(i),
                                "forecast_date": str(data.index.forecast_date.iloc[i].date()),
                                "label_end_date": str(pd.Timestamp(label_end_dates[i]).date()),
                                "y_true": float(y[i]), "y_pred": float(value)})
        manifests.append({"fold": fold, "outer_raw_train_rows": len(data.ids(fold, "train")),
                          "inner_fit_rows": len(inner_fit), "inner_stop_rows": len(stop),
                          "outer_fit_rows_after_purge": len(outer_train), "eval_raw_rows": len(data.ids(fold, "validation")),
                          "eval_eligible_rows": len(eval_ids), "first_eval_forecast": str(data.index.forecast_date.iloc[eval_ids[0]].date()),
                          "last_eval_label_end": str(pd.Timestamp(label_end_dates[eval_ids[-1]]).date()),
                          "alpha_trials": trials, "selected_alpha": selected["alpha"],
                          "ridge_weight_l2": weight_l2})
    predictions = pd.DataFrame(records)
    if predictions.duplicated(["model", "fold", "window_id"]).any():
        raise ValueError("Duplicate weekly prediction key")
    if (pd.to_datetime(predictions.label_end_date) > cutoff).any():
        raise ValueError("Viewed final-test dates entered weekly screen")
    counts = predictions.groupby("model").size()
    if counts.nunique() != 1 or not np.isfinite(predictions[["y_true", "y_pred"]].to_numpy()).all():
        raise ValueError("Model evaluation calendars differ or nonfinite values exist")
    def score(frame: pd.DataFrame) -> pd.Series:
        e = frame.y_pred.to_numpy() - frame.y_true.to_numpy()
        return pd.Series({"n": len(frame), "rmse": float(np.sqrt(np.mean(e * e))),
                          "mae": float(np.mean(np.abs(e))),
                          "pred_mean": float(frame.y_pred.mean()),
                          "pred_std": float(frame.y_pred.std(ddof=0))})
    pooled = predictions.groupby("model", sort=True).apply(score, include_groups=False).reset_index()
    by_fold = predictions.groupby(["model", "fold"], sort=True).apply(score, include_groups=False).reset_index()
    baselines = pooled.loc[pooled.model.ne("Ridge_5day")]
    best = baselines.sort_values("rmse").iloc[0]
    candidate = pooled.loc[pooled.model.eq("Ridge_5day")].iloc[0]
    joined = by_fold.loc[by_fold.model.eq("Ridge_5day"), ["fold", "rmse"]].merge(
        by_fold.loc[by_fold.model.eq(best.model), ["fold", "rmse"]],
        on="fold", suffixes=("_candidate", "_baseline"), validate="one_to_one")
    wins = int((joined.rmse_candidate < joined.rmse_baseline).sum())
    decision = {"experiment_id": contract["experiment_id"], "best_simple_baseline": str(best.model),
                "best_baseline_rmse": float(best.rmse), "ridge_rmse": float(candidate.rmse),
                "relative_rmse_improvement_pct": float(100 * (1 - candidate.rmse / best.rmse)),
                "fold_wins": wins, "eval_rows_per_model": int(counts.iloc[0]),
                "passes_screen": bool(candidate.rmse <= 0.99 * best.rmse and wins >= 2 and candidate.mae <= 1.01 * best.mae)}
    OUT.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(OUT / "predictions.csv", index=False)
    pooled.to_csv(OUT / "comparison_pooled.csv", index=False)
    by_fold.to_csv(OUT / "comparison_by_fold.csv", index=False)
    (OUT / "manifest.json").write_text(json.dumps({"contract_sha256": digest(CONTRACT),
        "source_sha256": digest(Path(__file__)),
        "data_sha256": digest(ROOT / "data/processed/srt_daily_aligned.csv"),
        "windows_sha256": digest(ROOT / "data/processed/srt_windows_unscaled.npz"),
        "folds_sha256": digest(ROOT / "data/processed/fold_assignments.csv"),
        "folds": manifests}, indent=2), encoding="utf-8")
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return decision


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
