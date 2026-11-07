"""Same-calendar 20-day SRT return screen for an archived SGS rate signal."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.project_eval import Dataset, Standardizer, VALIDATION_FOLDS
from src.v2.baselines import fit_ridge, predict_ridge
from src.v2.screen_trend_rate import rate_features
from src.v11.horizon_screen import ROOT, digest, fold_ids, labels_and_momentum, scores


CONTRACT = ROOT / "docs/v11_rate_mechanism_contract_2026-11-06.json"
SGS = ROOT / "data/processed/mas_sgs10y_202111_202509.csv"
SGS_MANIFEST = ROOT / "data/source_sgs10y_manifest.json"
OUT = ROOT / "output/v11/rate_mechanism"
HORIZON = 20


def predict_design(x: np.ndarray, y: np.ndarray, fit_ids: np.ndarray,
                   eval_ids: np.ndarray, alpha: float) -> np.ndarray:
    scaler = Standardizer.fit(x[fit_ids, None, :], y[fit_ids])
    fitted = fit_ridge(scaler.x_transform(x[fit_ids, None, :]),
                       scaler.y_transform(y[fit_ids]), alpha)
    return scaler.y_inverse(predict_ridge(scaler.x_transform(x[eval_ids, None, :]), fitted))


def run() -> dict:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refuse to overwrite V11 rate results: {OUT}")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract["experiment_id"] != "V11-analyst-guided-20day-rate-screen-01" or tuple(contract["folds"]) != VALIDATION_FOLDS:
        raise ValueError("Unexpected V11 rate mechanism contract")
    data = Dataset(ROOT)
    cutoff = pd.Timestamp(contract["development_cutoff"])
    y, momentum, horizon_eligible, label_end = labels_and_momentum(data, HORIZON, cutoff)
    aligned = rate_features(data.daily, pd.read_csv(SGS))
    ends = data.index.daily_end_row.to_numpy(dtype=int)
    if not np.array_equal(data.daily.date.iloc[ends].to_numpy(), data.index.forecast_date.to_numpy()):
        raise ValueError("SRT forecast dates and rate feature endpoints disagree")
    rate_cols = ["sgs10y_minus_sora_pp", "sgs10y_change_5_srt_days_pp"]
    extra = aligned.loc[ends, rate_cols].to_numpy(dtype=float)
    eligible = horizon_eligible & np.isfinite(extra).all(axis=1)
    base = data.x.reshape(len(data.x), -1).astype(float)
    designs = {"BaseRidge": base, "RateRidge": np.column_stack((base, extra))}
    records, fits = [], []
    for fold in VALIDATION_FOLDS:
        inner_fit, stopping, training, evaluation = fold_ids(data, eligible, label_end, fold)
        predictions = {"Zero": np.zeros(len(evaluation)),
                       "TrainMean": np.full(len(evaluation), float(y[training].mean())),
                       "PastMomentum": momentum[evaluation]}
        for model, design in designs.items():
            trials = []
            for alpha in (10.0, 100.0, 1000.0):
                stop_pred = predict_design(design, y, inner_fit, stopping, alpha)
                trials.append({"alpha": alpha,
                               "rmse": float(np.sqrt(np.mean((stop_pred - y[stopping]) ** 2)))})
            selected = min(trials, key=lambda row: (row["rmse"], -row["alpha"]))
            predictions[model] = predict_design(design, y, training, evaluation, selected["alpha"])
            fits.append({"fold": fold, "model": model, "inner_fit_n": len(inner_fit),
                         "inner_stop_n": len(stopping), "outer_fit_n": len(training),
                         "evaluation_n": len(evaluation), "trials": trials,
                         "selected_alpha": selected["alpha"]})
        for model, values in predictions.items():
            for window_id, value in zip(evaluation, values):
                records.append({"fold": fold, "model": model, "window_id": int(window_id),
                                "forecast_date": str(data.index.forecast_date.iloc[window_id].date()),
                                "label_end_date": str(pd.Timestamp(label_end[window_id]).date()),
                                "y_true": float(y[window_id]), "y_pred": float(value)})
    predictions = pd.DataFrame(records)
    if predictions.duplicated(["fold", "model", "window_id"]).any():
        raise ValueError("Duplicate rate prediction key")
    if (pd.to_datetime(predictions.label_end_date) > cutoff).any():
        raise ValueError("Final-period target entered rate screen")
    if not np.isfinite(predictions[["y_true", "y_pred"]].to_numpy()).all():
        raise ValueError("Nonfinite rate predictions")
    pooled = predictions.groupby("model", sort=True).apply(scores, include_groups=False).reset_index()
    by_fold = predictions.groupby(["model", "fold"], sort=True).apply(scores, include_groups=False).reset_index()
    simple = pooled.loc[~pooled.model.isin(("BaseRidge", "RateRidge"))].sort_values("rmse").iloc[0]
    base = pooled.loc[pooled.model.eq("BaseRidge")].iloc[0]
    rate = pooled.loc[pooled.model.eq("RateRidge")].iloc[0]
    fold_scores = by_fold.pivot(index="fold", columns="model", values="rmse")
    wins_simple = int((fold_scores.RateRidge < fold_scores[str(simple.model)]).sum())
    wins_base = int((fold_scores.RateRidge < fold_scores.BaseRidge).sum())
    decision = {"best_simple_baseline": str(simple.model), "baseline_rmse": float(simple.rmse),
                "base_ridge_rmse": float(base.rmse), "rate_ridge_rmse": float(rate.rmse),
                "rate_gain_vs_simple_pct": float(100 * (1 - rate.rmse / simple.rmse)),
                "rate_gain_vs_base_pct": float(100 * (1 - rate.rmse / base.rmse)),
                "fold_wins_vs_simple": wins_simple, "fold_wins_vs_base": wins_base,
                "evaluation_rows_per_model": int(rate.n),
                "passes_heuristic": bool(rate.rmse <= .99 * simple.rmse and rate.rmse <= .99 * base.rmse
                                         and rate.mae <= 1.01 * simple.mae
                                         and wins_simple >= 2 and wins_base >= 2),
                "evidence": "development-only; overlapping labels; no fundamental or state data yet"}
    OUT.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(OUT / "predictions.csv", index=False)
    pooled.to_csv(OUT / "pooled.csv", index=False)
    by_fold.to_csv(OUT / "by_fold.csv", index=False)
    aligned.loc[aligned.date.le(cutoff), ["date", "sgs_source_date", "sgs10y_yield_pct",
        "sgs_stale_calendar_days", *rate_cols]].to_csv(OUT / "sgs_alignment.csv", index=False)
    (OUT / "manifest.json").write_text(json.dumps({"contract_sha256": digest(CONTRACT),
        "code_sha256": digest(Path(__file__)), "sgs_sha256": digest(SGS),
        "sgs_manifest_sha256": digest(SGS_MANIFEST),
        "srt_daily_sha256": digest(ROOT / "data/processed/srt_daily_aligned.csv"),
        "fold_sha256": digest(ROOT / "data/processed/fold_assignments.csv"),
        "fits": fits}, indent=2), encoding="utf-8")
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return decision


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
