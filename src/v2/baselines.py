"""H1: train-only mean and inner-selected Ridge on the original development folds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.project_eval import Dataset, PREDICTION_COLUMNS, VALIDATION_FOLDS


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/v2/h1"
CONTRACT = ROOT / "docs/v2_experiment_contract_2026-11-04.json"
ALPHAS = (0.1, 1.0, 10.0)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """Sum-SSE Ridge with unpenalized intercept, using float64 linear algebra."""
    design = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
    target = np.asarray(y, dtype=np.float64)
    x_mean = design.mean(axis=0)
    y_mean = float(target.mean())
    centered = design - x_mean
    weights = np.linalg.solve(centered.T @ centered + alpha * np.eye(centered.shape[1]),
                              centered.T @ (target - y_mean))
    intercept = y_mean - float(x_mean @ weights)
    return weights, intercept


def predict_ridge(x: np.ndarray, fitted: tuple[np.ndarray, float]) -> np.ndarray:
    weights, intercept = fitted
    return np.asarray(x, dtype=np.float64).reshape(len(x), -1) @ weights + intercept


def make_rows(model: str, fold: str, ids: np.ndarray, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame([(model, "all", 0, fold, 0, int(i), float(y))
                         for i, y in zip(ids, values)], columns=PREDICTION_COLUMNS)


def summarize(data: Dataset, predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    original = pd.read_csv(ROOT / "output/development/predictions.csv")
    original = original.loc[original.source_group.eq("all") & original.model_id.isin(
        ("LSTM32", "LSTM3", "Fuzzy2", "Gate2"))]
    combined = pd.concat([original, data.zero_predictions(), predictions], ignore_index=True)
    checked = data.validate_predictions(combined)
    scored = data.pooled_validation_scores(combined)
    pooled = scored.groupby("model_id", as_index=False).agg(
        seeds=("seed", "nunique"), mean_pooled_rmse=("return_rmse", "mean"),
        mean_pooled_mae=("return_mae", "mean")
    ).sort_values("mean_pooled_rmse")
    strongest = pooled.iloc[0].to_dict()
    by_fold = data.forecast_scores(combined).groupby(["model_id", "fold"], as_index=False).agg(
        seeds=("seed", "nunique"), mean_rmse=("return_rmse", "mean"),
        mean_mae=("return_mae", "mean"))
    bias = checked.groupby(["model_id", "fold", "seed"], as_index=False).agg(
        n=("window_id", "size"), actual_mean=("y_true", "mean"),
        actual_sd=("y_true", "std"), predicted_mean=("y_pred", "mean"),
        predicted_sd=("y_pred", "std"),
        positive_prediction_fraction=("y_pred", lambda s: float((s > 0).mean())))
    selection = {"status": "fixed_before_h3", "experiment_id": "V2-H1-simple-baselines",
                 "model_id": strongest["model_id"],
                 "mean_pooled_rmse": strongest["mean_pooled_rmse"],
                 "mean_pooled_mae": strongest["mean_pooled_mae"],
                 "comparison_rule": "lowest mean of seed-pooled RMSE among Z0, MeanTrain, Ridge, LSTM32, LSTM3, Fuzzy2, Gate2",
                 "contract_sha256": digest(CONTRACT)}
    return pooled, by_fold, bias, selection


def write_analysis(data: Dataset, predictions: pd.DataFrame) -> dict:
    pooled, by_fold, bias, selection = summarize(data, predictions)
    pooled.to_csv(OUT / "comparison_pooled.csv", index=False)
    by_fold.to_csv(OUT / "comparison_by_fold.csv", index=False)
    bias.to_csv(OUT / "bias_and_amplitude.csv", index=False)
    selection["h1_predictions_sha256"] = digest(OUT / "predictions.csv")
    (OUT / "selected_benchmark_pre_h3.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    return selection


def run() -> dict:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"H1 output already exists; do not overwrite: {OUT}")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract["h1"]["ridge_alphas"] != list(ALPHAS):
        raise ValueError("The pre-run H1 alpha grid changed")
    data = Dataset(ROOT)
    rows, manifests = [], []
    for fold in VALIDATION_FOLDS:
        inner = data.prepare(fold, "all", inner_stop=True)
        outer = data.prepare(fold, "all", inner_stop=False)
        choices = []
        for alpha in ALPHAS:
            fitted = fit_ridge(inner.x_train, inner.y_train, alpha)
            stop_raw = inner.scaler.y_inverse(predict_ridge(inner.x_eval, fitted))
            stop_rmse = float(np.sqrt(np.mean((stop_raw - inner.y_eval_raw) ** 2)))
            choices.append({"alpha": alpha, "inner_stop_rmse_raw": stop_rmse})
        selected = min(choices, key=lambda item: (item["inner_stop_rmse_raw"], -item["alpha"]))
        outer_fit = fit_ridge(outer.x_train, outer.y_train, selected["alpha"])
        ridge_raw = outer.scaler.y_inverse(predict_ridge(outer.x_eval, outer_fit))
        mean_raw = np.full(len(outer.eval_ids), outer.scaler.y_mean)
        rows.extend((make_rows("Ridge", fold, outer.eval_ids, ridge_raw),
                     make_rows("MeanTrain", fold, outer.eval_ids, mean_raw)))
        manifests.append({
            "fold": fold, "experiment_id": "V2-H1-simple-baselines",
            "inner_fit_rows": len(inner.train_ids), "inner_stop_rows": len(inner.eval_ids),
            "outer_fit_rows": len(outer.train_ids), "eval_rows": len(outer.eval_ids),
            "alpha_trials": choices, "selected_alpha": selected["alpha"],
            "outer_train_target_mean_raw": outer.scaler.y_mean,
            "ridge_weight_l2_standardized": float(np.linalg.norm(outer_fit[0])),
            "data_sha256": digest(ROOT / "data/processed/srt_windows_unscaled.npz"),
            "folds_sha256": digest(ROOT / "data/processed/fold_assignments.csv"),
            "contract_sha256": digest(CONTRACT), "source_sha256": digest(Path(__file__)),
        })
    predictions = pd.concat(rows, ignore_index=True)
    data.validate_predictions(predictions)
    OUT.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(OUT / "predictions.csv", index=False)
    (OUT / "manifests.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    selection = write_analysis(data, predictions)
    return {"status": "H1_complete", "ridge_fits": 12, "eval_rows": len(predictions),
            "selected_reference": selection["model_id"],
            "selected_reference_rmse": selection["mean_pooled_rmse"]}


def reanalyse_saved() -> dict:
    if (ROOT / "output/v2/h3").exists():
        raise RuntimeError("Cannot change the H1 reference after H3 begins")
    data = Dataset(ROOT)
    predictions = pd.read_csv(OUT / "predictions.csv")
    data.validate_predictions(predictions)
    previous = json.loads((OUT / "selected_benchmark_pre_h3.json").read_text(encoding="utf-8"))
    selection = write_analysis(data, predictions)
    correction = {"reason": "Initial H1 summary omitted the deterministic Z0 table because it is not in development/predictions.csv",
                  "fit_predictions_unchanged_sha256": digest(OUT / "predictions.csv"),
                  "old_selected_reference": previous["model_id"],
                  "corrected_selected_reference": selection["model_id"],
                  "refits": 0, "source_at_correction_sha256": digest(Path(__file__))}
    (OUT / "analysis_correction.json").write_text(json.dumps(correction, indent=2), encoding="utf-8")
    return correction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reanalyse-saved", action="store_true")
    args = parser.parse_args()
    print(json.dumps(reanalyse_saved() if args.reanalyse_saved else run(), indent=2))
