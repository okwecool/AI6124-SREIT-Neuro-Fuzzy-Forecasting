"""Bounded development-only screen for 50/200 trend and prior SGS 10Y yield."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.project_eval import Dataset, PREDICTION_COLUMNS, Standardizer, VALIDATION_FOLDS
from src.v2.baselines import fit_ridge, predict_ridge
from src.v2.run_residual_fuzzy import ResidualDataset
from src.v2.screen_distributions import EVENTS, announced_adjustments


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs/v2_trend_rate_screen_contract_2026-11-04.json"
SGS = ROOT / "data/processed/mas_sgs10y_202111_202509.csv"
SGS_MANIFEST = ROOT / "data/source_sgs10y_manifest.json"
OUT = ROOT / "output/v2/trend_rate_screen"
GROUPS = {
    "base": "RidgeResidualBaseMatched",
    "trend": "RidgeResidualTrend",
    "rate": "RidgeResidualSGS10Y",
    "trend_rate": "RidgeResidualTrendSGS10Y",
}
ALPHAS = (10.0, 100.0, 1000.0)
CUTOFF = pd.Timestamp("2025-09-30")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trend_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Causal gross-distribution index and two scale-free trend descriptors."""
    c = daily.srt_close.to_numpy(dtype=np.float64)
    dividend = daily.srt_dividend_sgd_on_date.to_numpy(dtype=np.float64)
    if (not np.isfinite(c).all() or (c <= 0).any() or not np.isfinite(dividend).all()
            or (dividend < 0).any() or (c + dividend <= 0).any()):
        raise ValueError("Invalid SRT close or distribution in daily data")
    gross = np.ones(len(c), dtype=np.float64)
    gross[1:] = (c[1:] + dividend[1:]) / c[:-1]
    total_return_index = pd.Series(np.cumprod(gross))
    sma50 = total_return_index.rolling(50, min_periods=50).mean()
    sma200 = total_return_index.rolling(200, min_periods=200).mean()
    spread = np.log(sma50 / sma200)
    return pd.DataFrame({"date": daily.date, "srt_total_return_index": total_return_index,
                         "ma50_200_log_ratio": spread,
                         "ma50_200_log_ratio_change_5": spread - spread.shift(5)})


def rate_features(daily: pd.DataFrame, sgs: pd.DataFrame) -> pd.DataFrame:
    """Use the last MAS date strictly before each SRT forecast date."""
    expected = {"date", "sgs10y_yield_pct"}
    if set(sgs.columns) != expected:
        raise ValueError("Unexpected SGS source columns")
    source = sgs.copy()
    source["date"] = pd.to_datetime(source.date)
    if (source.date.duplicated().any() or not source.date.is_monotonic_increasing
            or source.sgs10y_yield_pct.isna().any()
            or not source.sgs10y_yield_pct.between(0, 10).all()):
        raise ValueError("SGS dates or yields are invalid")
    left = daily[["date", "sora_rate_pct_asof"]].copy()
    right = source.rename(columns={"date": "sgs_source_date"})
    aligned = pd.merge_asof(left.sort_values("date"), right,
                            left_on="date", right_on="sgs_source_date",
                            direction="backward", allow_exact_matches=False)
    development = aligned.date.le(CUTOFF)
    if (aligned.loc[development, "sgs_source_date"].isna().any()
            or not (aligned.loc[development, "sgs_source_date"] < aligned.loc[development, "date"]).all()):
        raise ValueError("10-year yield not known before a development forecast date")
    aligned["sgs_stale_calendar_days"] = (aligned.date - aligned.sgs_source_date).dt.days
    if (aligned.loc[development, "sgs_stale_calendar_days"] > 10).any():
        raise ValueError("10-year yield is unexpectedly stale")
    aligned["sgs10y_minus_sora_pp"] = aligned.sgs10y_yield_pct - aligned.sora_rate_pct_asof
    aligned["sgs10y_change_5_srt_days_pp"] = aligned.sgs10y_yield_pct - aligned.sgs10y_yield_pct.shift(5)
    aligned.loc[~development, ["sgs10y_minus_sora_pp", "sgs10y_change_5_srt_days_pp"]] = np.nan
    return aligned


def design_matrices(data: ResidualDataset, trend: pd.DataFrame,
                    rate: pd.DataFrame) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if not data.daily.date.equals(trend.date) or not data.daily.date.equals(rate.date):
        raise ValueError("Daily feature calendars do not match SRT")
    ends = data.index.daily_end_row.to_numpy(dtype=int)
    if not np.array_equal(data.daily.date.iloc[ends].to_numpy(), data.index.forecast_date.to_numpy()):
        raise ValueError("Forecast dates do not match endpoint feature rows")
    base = data.x.reshape(len(data.x), -1).astype(np.float64)
    ma = trend.loc[ends, ["ma50_200_log_ratio", "ma50_200_log_ratio_change_5"]].to_numpy(dtype=np.float64)
    yield_values = rate.loc[ends, ["sgs10y_minus_sora_pp", "sgs10y_change_5_srt_days_pp"]].to_numpy(dtype=np.float64)
    development_target = data.index.target_date.le(CUTOFF).to_numpy()
    valid = (development_target & np.isfinite(base).all(axis=1)
             & np.isfinite(ma).all(axis=1) & np.isfinite(yield_values).all(axis=1))
    matrices = {
        "base": base,
        "trend": np.column_stack((base, ma)),
        "rate": np.column_stack((base, yield_values)),
        "trend_rate": np.column_stack((base, ma, yield_values)),
    }
    for fold in VALIDATION_FOLDS:
        if not valid[data.ids(fold, "validation")].all():
            raise ValueError(f"Missing new feature on {fold} evaluation dates")
    return matrices, valid


def fit_one(data: ResidualDataset, x: np.ndarray, valid: np.ndarray,
            fold: str) -> tuple[np.ndarray, np.ndarray, dict]:
    if fold not in VALIDATION_FOLDS:
        raise ValueError("Only development folds are permitted")
    train_ids = data.ids(fold, "train")
    train_ids = train_ids[valid[train_ids]]
    eval_ids = data.ids(fold, "validation")
    if len(train_ids) < 300 or not valid[eval_ids].all():
        raise ValueError("Insufficient matched training or evaluation data")
    stop_count = math.ceil(0.15 * len(train_ids))
    inner_fit, inner_stop = train_ids[:-stop_count], train_ids[-stop_count:]
    if data.index.target_date.iloc[inner_fit].max() >= data.index.target_date.iloc[inner_stop].min():
        raise ValueError("Inner fit/stop target dates overlap")
    if data.index.target_date.iloc[train_ids].max() >= data.index.target_date.iloc[eval_ids].min():
        raise ValueError("Outer train/eval target dates overlap")
    x3 = x[:, None, :]
    inner_scaler = Standardizer.fit(x3[inner_fit], data.y[inner_fit])
    x_fit = inner_scaler.x_transform(x3[inner_fit])
    y_fit = inner_scaler.y_transform(data.y[inner_fit])
    x_stop = inner_scaler.x_transform(x3[inner_stop])
    trials = []
    for alpha in ALPHAS:
        fitted = fit_ridge(x_fit, y_fit, alpha)
        raw = inner_scaler.y_inverse(predict_ridge(x_stop, fitted))
        trials.append({"alpha": alpha, "inner_stop_rmse": float(np.sqrt(np.mean((raw - data.y[inner_stop]) ** 2)))})
    chosen = min(trials, key=lambda row: (row["inner_stop_rmse"], -row["alpha"]))
    scaler = Standardizer.fit(x3[train_ids], data.y[train_ids])
    fitted = fit_ridge(scaler.x_transform(x3[train_ids]), scaler.y_transform(data.y[train_ids]), chosen["alpha"])
    pred = scaler.y_inverse(predict_ridge(scaler.x_transform(x3[eval_ids]), fitted))
    manifest = {"fold": fold, "train_rows": len(train_ids), "inner_fit_rows": len(inner_fit),
                "inner_stop_rows": len(inner_stop), "eval_rows": len(eval_ids),
                "alpha_trials": trials, "selected_alpha": chosen["alpha"],
                "feature_columns": x.shape[1], "outer_target_mean_residual": scaler.y_mean,
                "outer_weight_l2": float(np.linalg.norm(fitted[0]))}
    return eval_ids, pred, manifest


def run() -> dict:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refuse to overwrite results: {OUT}")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if (contract["experiment_id"] != "V2-H11-trend-SGS10Y-residual-ridge-01"
            or tuple(contract["folds"]) != VALIDATION_FOLDS):
        raise ValueError("Unexpected experiment contract")
    if tuple(contract["groups"]) != tuple(GROUPS) or tuple(contract["alpha_grid"]) != ALPHAS:
        raise ValueError("Unexpected group or alpha grid")
    raw = Dataset(ROOT)
    data = ResidualDataset(ROOT)
    sgs = pd.read_csv(SGS)
    trend = trend_features(data.daily)
    rate = rate_features(data.daily, sgs)
    matrices, valid = design_matrices(data, trend, rate)
    rows, fits = [], []
    adjustment = announced_adjustments(raw, pd.read_csv(EVENTS)).set_index("window_id").adjustment
    for group, model in GROUPS.items():
        for fold in VALIDATION_FOLDS:
            ids, residual_pred, manifest = fit_one(data, matrices[group], valid, fold)
            pred = residual_pred + adjustment.loc[ids].to_numpy(dtype=float)
            rows.extend((model, "all", 0, fold, 0, int(i), float(p)) for i, p in zip(ids, pred))
            fits.append({"group": group, "model_id": model, **manifest})
    predictions = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    raw.validate_predictions(predictions)
    schedule = pd.read_csv(ROOT / "output/v2/announced_distribution_screen/predictions.csv")
    schedule = schedule.loc[schedule.model_id.eq("DivSchedule"), PREDICTION_COLUMNS]
    if len(schedule) != 378 or len(predictions) != 4 * 378:
        raise ValueError("The fixed evaluation calendar changed")
    comparison = pd.concat((schedule, predictions), ignore_index=True)
    pooled = raw.pooled_validation_scores(comparison).sort_values("return_rmse")
    by_fold = raw.forecast_scores(comparison).sort_values(["model_id", "fold"])
    scores = pooled.set_index("model_id")
    reference = scores.loc[GROUPS["base"]]
    schedule_score = scores.loc["DivSchedule"]
    decisions = {}
    for group, model in GROUPS.items():
        if group == "base":
            continue
        candidate = scores.loc[model]
        pair = by_fold.loc[by_fold.model_id.eq(model), ["fold", "return_rmse"]].merge(
            by_fold.loc[by_fold.model_id.eq("DivSchedule"), ["fold", "return_rmse"]],
            on="fold", suffixes=("_candidate", "_schedule"), validate="one_to_one")
        fold_wins = int((pair.return_rmse_candidate < pair.return_rmse_schedule).sum())
        decisions[group] = {"model_id": model, "rmse": float(candidate.return_rmse),
                            "mae": float(candidate.return_mae),
                            "rmse_gain_vs_matched_base_pct": float(100 * (1 - candidate.return_rmse / reference.return_rmse)),
                            "rmse_gain_vs_schedule_pct": float(100 * (1 - candidate.return_rmse / schedule_score.return_rmse)),
                            "fold_wins_vs_schedule": fold_wins,
                            "passes_screen": bool(candidate.return_rmse <= .99 * reference.return_rmse
                                                  and candidate.return_rmse <= .99 * schedule_score.return_rmse
                                                  and fold_wins >= 2
                                                  and candidate.return_mae <= 1.01 * schedule_score.return_mae)}
    OUT.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(OUT / "predictions.csv", index=False)
    pooled.to_csv(OUT / "comparison_pooled.csv", index=False)
    by_fold.to_csv(OUT / "comparison_by_fold.csv", index=False)
    rate.loc[rate.date.le(CUTOFF), ["date", "sgs_source_date", "sgs10y_yield_pct",
                                     "sgs_stale_calendar_days", "sgs10y_minus_sora_pp",
                                     "sgs10y_change_5_srt_days_pp"]].to_csv(OUT / "sgs_development_alignment.csv", index=False)
    (OUT / "manifests.json").write_text(json.dumps({"experiment_id": contract["experiment_id"],
        "contract_sha256": sha256(CONTRACT), "source_sha256": sha256(Path(__file__)),
        "sgs_source_sha256": sha256(SGS), "sgs_manifest_sha256": sha256(SGS_MANIFEST),
        "srt_daily_sha256": sha256(ROOT / "data/processed/srt_daily_aligned.csv"),
        "v1_windows_sha256": sha256(ROOT / "data/processed/srt_windows_unscaled.npz"),
        "folds_sha256": sha256(ROOT / "data/processed/fold_assignments.csv"),
        "eligible_windows": int(valid.sum()), "fits": fits}, indent=2), encoding="utf-8")
    (OUT / "decision.json").write_text(json.dumps(decisions, indent=2), encoding="utf-8")
    return decisions


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
