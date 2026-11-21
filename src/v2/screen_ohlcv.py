"""Post-hoc development screen: stronger Ridge and separate causal OHLCV groups.

This entry point deliberately accepts only the three saved development folds.
It does not create or inspect predictions for the viewed final holdout.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.project_eval import Dataset, PREDICTION_COLUMNS, Standardizer, VALIDATION_FOLDS
from src.v2.baselines import fit_ridge, predict_ridge


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs/v2_h1b_h2_screen_contract_2026-11-04.json"
OUT = ROOT / "output/v2/h1b_h2_screen"
NAMES = {"base": "RidgeStrong", "price": "RidgePrice", "volume": "RidgeVolume"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def daily_feature_groups(daily: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return daily features available after that day's close, with no fill."""
    c = daily.srt_close.astype(float)
    o = daily.srt_open.astype(float)
    h = daily.srt_high.astype(float)
    l = daily.srt_low.astype(float)
    volume = daily.srt_volume.astype(float)
    if (c.dropna() <= 0).any() or (o.dropna() <= 0).any() or (h.dropna() <= 0).any() or (l.dropna() <= 0).any():
        raise ValueError("OHLC prices must be positive")
    if (volume.dropna() < 0).any():
        raise ValueError("Volume must be nonnegative")
    if ((h < l) & h.notna() & l.notna()).any():
        raise ValueError("Daily high below low")
    v = np.log1p(volume)
    price = np.column_stack((np.log(o / c.shift(1)), np.log(c / o), np.log(h / l)))
    volume_features = np.column_stack((
        v - v.shift(1).rolling(20, min_periods=20).mean(),
        v.rolling(5, min_periods=5).mean() - v.shift(5).rolling(5, min_periods=5).mean(),
    ))
    return {"price": price, "volume": volume_features}


def build_windows(data: Dataset) -> dict[str, np.ndarray]:
    groups = daily_feature_groups(data.daily)
    windows = {"base": data.x}
    starts = data.index.daily_start_row.to_numpy(dtype=int)
    ends = data.index.daily_end_row.to_numpy(dtype=int)
    if not np.array_equal(ends - starts + 1, np.full(len(starts), 20)):
        raise ValueError("Expected the original 20-day window indexing")
    if not np.array_equal(data.daily.date.iloc[ends].to_numpy(), data.index.forecast_date.to_numpy()):
        raise ValueError("Forecast dates do not match the last daily feature row")
    for group, daily_values in groups.items():
        incremental = np.stack([daily_values[start:end + 1] for start, end in zip(starts, ends)])
        windows[group] = np.concatenate((data.x, incremental.astype(np.float32)), axis=-1)
    for fold in VALIDATION_FOLDS:
        used = np.concatenate((data.ids(fold, "train"), data.ids(fold, "validation")))
        for group, window in windows.items():
            if not np.isfinite(window[used]).all():
                raise ValueError(f"Nonfinite {group} feature in {fold}; no date dropping is allowed")
    return windows


def fold_arrays(data: Dataset, x: np.ndarray, fold: str, inner_stop: bool):
    if fold not in VALIDATION_FOLDS:
        raise ValueError("Only development folds are permitted")
    train_ids = data.ids(fold, "train")
    if inner_stop:
        count = math.ceil(0.15 * len(train_ids))
        fit_ids, eval_ids = train_ids[:-count], train_ids[-count:]
    else:
        fit_ids, eval_ids = train_ids, data.ids(fold, "validation")
    if data.index.target_date.iloc[fit_ids].max() >= data.index.target_date.iloc[eval_ids].min():
        raise ValueError("Training label overlaps evaluation time")
    scaler = Standardizer.fit(x[fit_ids], data.y[fit_ids])
    return (fit_ids, eval_ids, scaler.x_transform(x[fit_ids]),
            scaler.y_transform(data.y[fit_ids]), scaler.x_transform(x[eval_ids]),
            data.y[eval_ids], scaler)


def fit_one(data: Dataset, x: np.ndarray, fold: str, alphas: tuple[float, ...]):
    fit_ids, stop_ids, x_fit, y_fit, x_stop, y_stop, inner_scaler = fold_arrays(data, x, fold, True)
    trials = []
    for alpha in alphas:
        fitted = fit_ridge(x_fit, y_fit, alpha)
        raw = inner_scaler.y_inverse(predict_ridge(x_stop, fitted))
        trials.append({"alpha": alpha, "inner_stop_rmse": float(np.sqrt(np.mean((raw - y_stop) ** 2)))})
    chosen = min(trials, key=lambda row: (row["inner_stop_rmse"], -row["alpha"]))
    outer_fit_ids, eval_ids, x_train, y_train, x_eval, _, outer_scaler = fold_arrays(data, x, fold, False)
    fitted = fit_ridge(x_train, y_train, chosen["alpha"])
    raw = outer_scaler.y_inverse(predict_ridge(x_eval, fitted))
    record = {"fold": fold, "inner_fit_rows": len(fit_ids), "inner_stop_rows": len(stop_ids),
              "outer_fit_rows": len(outer_fit_ids), "eval_rows": len(eval_ids),
              "alpha_trials": trials, "selected_alpha": chosen["alpha"],
              "outer_target_mean": outer_scaler.y_mean,
              "outer_weight_l2": float(np.linalg.norm(fitted[0]))}
    return eval_ids, raw, record


def make_predictions(model_id: str, fold: str, ids: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame([(model_id, "all", 0, fold, 0, int(i), float(y))
                         for i, y in zip(ids, y_pred)], columns=PREDICTION_COLUMNS)


def run() -> dict:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refuse to overwrite experiment results: {OUT}")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if tuple(contract["folds"]) != VALIDATION_FOLDS or tuple(contract["alpha_grid"]) != (10.0, 100.0, 1000.0):
        raise ValueError("Unexpected experiment contract")
    data = Dataset(ROOT)
    windows = build_windows(data)
    rows, manifests = [], []
    for group, model_id in NAMES.items():
        for fold in VALIDATION_FOLDS:
            ids, predictions, manifest = fit_one(data, windows[group], fold, tuple(contract["alpha_grid"]))
            rows.append(make_predictions(model_id, fold, ids, predictions))
            manifest.update({"group": group, "model_id": model_id, "features": windows[group].shape[-1]})
            manifests.append(manifest)
    prediction_table = pd.concat(rows, ignore_index=True)
    data.validate_predictions(prediction_table)
    reference = pd.concat([data.zero_predictions(), pd.read_csv(ROOT / "output/v2/h1/predictions.csv")], ignore_index=True)
    reference = reference.loc[reference.model_id.isin(("Z0", "Ridge"))]
    comparison = pd.concat([reference, prediction_table], ignore_index=True)
    pooled = data.pooled_validation_scores(comparison).sort_values("return_rmse")
    by_fold = data.forecast_scores(comparison).sort_values(["model_id", "fold"])
    z0 = pooled.loc[pooled.model_id.eq("Z0")].iloc[0]
    decisions = {}
    for group, model_id in NAMES.items():
        candidate = pooled.loc[pooled.model_id.eq(model_id)].iloc[0]
        merged = by_fold.loc[by_fold.model_id.eq(model_id), ["fold", "return_rmse"]].merge(
            by_fold.loc[by_fold.model_id.eq("Z0"), ["fold", "return_rmse"]],
            on="fold", suffixes=("_candidate", "_z0"), validate="one_to_one")
        fold_wins = int((merged.return_rmse_candidate < merged.return_rmse_z0).sum())
        decisions[group] = {"model_id": model_id, "rmse": float(candidate.return_rmse),
                            "mae": float(candidate.return_mae), "fold_wins_vs_z0": fold_wins,
                            "rmse_improvement_pct_vs_z0": float(100 * (1 - candidate.return_rmse / z0.return_rmse)),
                            "passes_screen": bool(candidate.return_rmse <= 0.99 * z0.return_rmse and
                                                  fold_wins >= 2 and candidate.return_mae <= 1.01 * z0.return_mae)}
    OUT.mkdir(parents=True, exist_ok=False)
    prediction_table.to_csv(OUT / "predictions.csv", index=False)
    pooled.to_csv(OUT / "comparison_pooled.csv", index=False)
    by_fold.to_csv(OUT / "comparison_by_fold.csv", index=False)
    (OUT / "manifests.json").write_text(json.dumps({"experiment_id": contract["experiment_id"],
        "contract_sha256": sha256(CONTRACT), "source_sha256": sha256(Path(__file__)),
        "source_data_sha256": sha256(ROOT / "data/processed/srt_daily_aligned.csv"),
        "base_windows_sha256": sha256(ROOT / "data/processed/srt_windows_unscaled.npz"),
        "folds_sha256": sha256(ROOT / "data/processed/fold_assignments.csv"),
        "fits": manifests}, indent=2), encoding="utf-8")
    (OUT / "decision.json").write_text(json.dumps(decisions, indent=2), encoding="utf-8")
    return decisions


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
