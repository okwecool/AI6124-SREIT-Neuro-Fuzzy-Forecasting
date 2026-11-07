"""Screen 20/60-trading-day SRT returns on the existing development folds.

This is a new exploratory target study. It never scores the previously viewed
final holdout and does not replace the approved next-day Fuzzy-LSTM experiment.
"""

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
CONTRACT = ROOT / "docs/v11_analyst_hypotheses_contract_2026-11-06.json"
OUT = ROOT / "output/v11/horizon_screen"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def labels_and_momentum(data: Dataset, horizon: int, cutoff: pd.Timestamp):
    """Use observed closes through t+h for labels and closes through t for inputs."""
    if horizon < 1:
        raise ValueError("Horizon must be positive")
    ends = data.index.daily_end_row.to_numpy(dtype=int)
    close = data.daily.srt_close.to_numpy(dtype=float)
    dates = data.daily.date.to_numpy(dtype="datetime64[ns]")
    safe_end = np.minimum(ends + horizon, len(close) - 1)
    safe_past = np.maximum(ends - horizon, 0)
    eligible = (ends >= horizon) & (ends + horizon < len(close))
    eligible &= dates[safe_end] <= np.datetime64(cutoff)
    eligible &= np.isfinite(close[ends]) & np.isfinite(close[safe_end]) & np.isfinite(close[safe_past])
    eligible &= (close[ends] > 0) & (close[safe_end] > 0) & (close[safe_past] > 0)
    y = np.full(len(ends), np.nan, dtype=float)
    past_momentum = np.full(len(ends), np.nan, dtype=float)
    y[eligible] = np.log(close[safe_end[eligible]] / close[ends[eligible]])
    past_momentum[eligible] = np.log(close[ends[eligible]] / close[safe_past[eligible]])
    return y, past_momentum, eligible, dates[safe_end]


def purge(ids: np.ndarray, label_end: np.ndarray, first_forecast: np.datetime64) -> np.ndarray:
    kept = ids[label_end[ids] < first_forecast]
    if not len(kept) or label_end[kept].max() >= first_forecast:
        raise ValueError("Horizon labels leak into evaluation period")
    return kept


def fold_ids(data: Dataset, eligible: np.ndarray, label_end: np.ndarray, fold: str):
    if fold not in VALIDATION_FOLDS:
        raise ValueError("V11 horizon screen is development-only")
    evaluation = data.ids(fold, "validation")
    evaluation = evaluation[eligible[evaluation]]
    if not len(evaluation):
        raise ValueError("No eligible evaluation rows")
    first_eval = np.datetime64(data.index.forecast_date.iloc[evaluation[0]])
    training = data.ids(fold, "train")
    training = purge(training[eligible[training]], label_end, first_eval)
    stop_count = math.ceil(0.15 * len(training))
    stopping = training[-stop_count:]
    first_stop = np.datetime64(data.index.forecast_date.iloc[stopping[0]])
    inner_fit = purge(training[:-stop_count], label_end, first_stop)
    return inner_fit, stopping, training, evaluation


def ridge_prediction(x: np.ndarray, y: np.ndarray, fit_ids: np.ndarray,
                     eval_ids: np.ndarray, alpha: float) -> np.ndarray:
    scaler = Standardizer.fit(x[fit_ids], y[fit_ids])
    fitted = fit_ridge(scaler.x_transform(x[fit_ids]), scaler.y_transform(y[fit_ids]), alpha)
    return scaler.y_inverse(predict_ridge(scaler.x_transform(x[eval_ids]), fitted))


def scores(frame: pd.DataFrame) -> pd.Series:
    error = frame.y_pred.to_numpy() - frame.y_true.to_numpy()
    return pd.Series({"n": len(frame), "rmse": float(np.sqrt(np.mean(error ** 2))),
                      "mae": float(np.mean(np.abs(error)))})


def run() -> dict:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refuse to overwrite V11 results: {OUT}")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if tuple(contract["folds"]) != VALIDATION_FOLDS or tuple(contract["horizons_trading_days"]) != (20, 60):
        raise ValueError("Unexpected V11 horizon contract")
    data = Dataset(ROOT)
    cutoff = pd.Timestamp(contract["development_cutoff"])
    records, manifests, decisions = [], [], []
    for horizon in contract["horizons_trading_days"]:
        y, momentum, eligible, label_end = labels_and_momentum(data, horizon, cutoff)
        for fold in VALIDATION_FOLDS:
            inner_fit, stopping, training, evaluation = fold_ids(data, eligible, label_end, fold)
            trials = []
            for alpha in contract["alpha_grid"]:
                prediction = ridge_prediction(data.x, y, inner_fit, stopping, alpha)
                trials.append({"alpha": alpha, "rmse": float(np.sqrt(np.mean((prediction - y[stopping]) ** 2)))})
            selected = min(trials, key=lambda row: (row["rmse"], -row["alpha"]))
            predictions = {
                "Zero": np.zeros(len(evaluation)),
                "TrainMean": np.full(len(evaluation), float(y[training].mean())),
                "PastMomentum": momentum[evaluation],
                "Ridge": ridge_prediction(data.x, y, training, evaluation, selected["alpha"]),
            }
            for model, values in predictions.items():
                for window_id, value in zip(evaluation, values):
                    records.append({"horizon": horizon, "fold": fold, "model": model,
                                    "window_id": int(window_id),
                                    "forecast_date": str(data.index.forecast_date.iloc[window_id].date()),
                                    "label_end_date": str(pd.Timestamp(label_end[window_id]).date()),
                                    "y_true": float(y[window_id]), "y_pred": float(value)})
            manifests.append({"horizon": horizon, "fold": fold, "inner_fit_n": len(inner_fit),
                              "inner_stop_n": len(stopping), "outer_fit_n": len(training),
                              "evaluation_n": len(evaluation), "alpha_trials": trials,
                              "selected_alpha": selected["alpha"],
                              "last_evaluation_label_end": str(pd.Timestamp(label_end[evaluation[-1]]).date())})
    predictions = pd.DataFrame(records)
    if predictions.duplicated(["horizon", "fold", "model", "window_id"]).any():
        raise ValueError("Duplicate V11 prediction key")
    if (pd.to_datetime(predictions.label_end_date) > cutoff).any():
        raise ValueError("Final-period labels entered V11 screen")
    if not np.isfinite(predictions[["y_true", "y_pred"]].to_numpy()).all():
        raise ValueError("Nonfinite V11 forecasts")
    pooled = predictions.groupby(["horizon", "model"], sort=True).apply(scores, include_groups=False).reset_index()
    by_fold = predictions.groupby(["horizon", "fold", "model"], sort=True).apply(scores, include_groups=False).reset_index()
    for horizon in contract["horizons_trading_days"]:
        subset = pooled.loc[pooled.horizon.eq(horizon)]
        best = subset.loc[subset.model.ne("Ridge")].sort_values("rmse").iloc[0]
        candidate = subset.loc[subset.model.eq("Ridge")].iloc[0]
        paired = by_fold.loc[by_fold.horizon.eq(horizon) & by_fold.model.eq("Ridge")].merge(
            by_fold.loc[by_fold.horizon.eq(horizon) & by_fold.model.eq(best.model)],
            on=["horizon", "fold"], suffixes=("_ridge", "_baseline"), validate="one_to_one")
        fold_wins = int((paired.rmse_ridge < paired.rmse_baseline).sum())
        decisions.append({"horizon": horizon, "best_simple_baseline": str(best.model),
                          "baseline_rmse": float(best.rmse), "ridge_rmse": float(candidate.rmse),
                          "relative_rmse_improvement_pct": float(100 * (1 - candidate.rmse / best.rmse)),
                          "ridge_mae": float(candidate.mae), "baseline_mae": float(best.mae),
                          "fold_wins": fold_wins, "evaluation_rows_per_model": int(candidate.n),
                          "passes_heuristic": bool(candidate.rmse <= 0.99 * best.rmse
                                                   and candidate.mae <= 1.01 * best.mae and fold_wins >= 2),
                          "evidence": "development-only; overlapping labels; 60-day horizon descriptive"})
    OUT.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(OUT / "predictions.csv", index=False)
    pooled.to_csv(OUT / "pooled.csv", index=False)
    by_fold.to_csv(OUT / "by_fold.csv", index=False)
    (OUT / "manifest.json").write_text(json.dumps({"contract_sha256": digest(CONTRACT),
        "code_sha256": digest(Path(__file__)),
        "data_sha256": digest(ROOT / "data/processed/srt_daily_aligned.csv"),
        "window_sha256": digest(ROOT / "data/processed/srt_windows_unscaled.npz"),
        "fold_sha256": digest(ROOT / "data/processed/fold_assignments.csv"),
        "folds": manifests}, indent=2), encoding="utf-8")
    (OUT / "decision.json").write_text(json.dumps(decisions, indent=2), encoding="utf-8")
    return {"experiment_id": contract["experiment_id"], "decisions": decisions}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
