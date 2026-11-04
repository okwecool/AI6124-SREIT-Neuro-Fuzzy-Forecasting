"""Read-only, post-hoc diagnosis of frozen development and final exports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.project_eval import Dataset


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/diagnostics_2026-11-04"
MODELS = ("LSTM32", "LSTM3", "Fuzzy2", "Gate2", "Z0")


def load_predictions(data: Dataset) -> pd.DataFrame:
    frames = [pd.read_csv(ROOT / f"output/{stage}/predictions.csv")
              for stage in ("development", "final")]
    predictions = pd.concat(frames, ignore_index=True)
    predictions = predictions.loc[predictions.source_group.eq("all") &
                                  predictions.model_id.isin(MODELS)].copy()
    return data.validate_predictions(predictions)


def forecast_diagnostics(data: Dataset, predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (fold, model, seed), group in predictions.groupby(["fold", "model_id", "seed"]):
        y = group.y_true.to_numpy(dtype=float)
        p = group.y_pred.to_numpy(dtype=float)
        train_mean = float(data.y[data.ids(fold, "train")].mean())
        mse = float(np.mean((y - p) ** 2))
        zero_mse = float(np.mean(y ** 2))
        mean_mse = float(np.mean((y - train_mean) ** 2))
        own_mean_mse = float(np.mean((y - p.mean()) ** 2))
        records.append({
            "fold": fold, "model_id": model, "seed": int(seed), "n": len(y),
            "train_target_mean": train_mean, "actual_mean": float(y.mean()),
            "actual_sd": float(y.std()), "predicted_mean": float(p.mean()),
            "predicted_sd": float(p.std()),
            "pred_to_actual_sd_ratio": float(p.std() / y.std()),
            "positive_prediction_fraction": float((p > 0).mean()),
            "positive_actual_fraction": float((y > 0).mean()),
            "rmse": float(np.sqrt(mse)), "zero_rmse_same_dates": float(np.sqrt(zero_mse)),
            "train_mean_rmse_same_dates": float(np.sqrt(mean_mse)),
            "own_prediction_mean_rmse_descriptive_only": float(np.sqrt(own_mean_mse)),
            "constant_component_mse_gain_vs_zero": zero_mse - own_mean_mse,
            "dynamic_component_mse_gain_vs_own_mean": own_mean_mse - mse,
            "mse_skill_vs_zero": float(1 - mse / zero_mse),
            "pred_true_corr": float(np.corrcoef(p, y)[0, 1]) if p.std() > 1e-12 else None,
            "mean_p_times_y": float(np.mean(p * y)),
            "mean_p_squared": float(np.mean(p ** 2)),
        })
    return pd.DataFrame(records).sort_values(["fold", "model_id", "seed"])


def expert_diagnostics() -> pd.DataFrame:
    frames = [pd.read_csv(ROOT / f"output/{stage}/rules.csv")
              for stage in ("development", "final")]
    rules = pd.concat(frames, ignore_index=True)
    rules = rules.loc[rules.model_id.eq("Fuzzy2") & rules.source_group.eq("all")].copy()
    key = ["fold", "seed", "window_id"]
    wide = rules.pivot(index=key, columns="rule_id", values=["alpha", "expert_raw"])
    wide.columns = [f"{name}_{rule}" for name, rule in wide.columns]
    wide = wide.reset_index()
    c1 = wide.alpha_1 * wide.expert_raw_1
    c2 = wide.alpha_2 * wide.expert_raw_2
    mix = c1 + c2
    wide["max_alpha"] = wide[["alpha_1", "alpha_2"]].max(axis=1)
    wide["opposite_expert_sign"] = wide.expert_raw_1 * wide.expert_raw_2 < 0
    wide["cancellation_fraction"] = 1 - mix.abs() / (c1.abs() + c2.abs() + 1e-15)
    wide["abs_mix"] = mix.abs()
    wide["mean_abs_expert"] = (wide.expert_raw_1.abs() + wide.expert_raw_2.abs()) / 2
    return wide.groupby(["fold", "seed"], as_index=False).agg(
        n=("window_id", "size"), mean_max_alpha=("max_alpha", "mean"),
        fraction_max_alpha_gt_08=("max_alpha", lambda x: float((x > .8).mean())),
        opposite_expert_sign_fraction=("opposite_expert_sign", "mean"),
        mean_cancellation_fraction=("cancellation_fraction", "mean"),
        mean_abs_mix=("abs_mix", "mean"),
        mean_abs_expert=("mean_abs_expert", "mean"),
    ).sort_values(["fold", "seed"])


def stopping_diagnostics() -> pd.DataFrame:
    records = []
    for fold in ("validation_1", "validation_2", "validation_3"):
        for model in ("LSTM32", "LSTM3", "Fuzzy2", "Gate2"):
            for seed in (11, 23, 37):
                path = ROOT / "output/runs/development" / fold / "all" / model / f"seed_{seed}/manifest.json"
                manifest = json.loads(path.read_text(encoding="utf-8"))
                trace = pd.DataFrame(manifest["inner_trace"])
                best = int(manifest["best_epoch"])
                records.append({
                    "fold": fold, "model_id": model, "seed": seed,
                    "best_epoch": best, "epochs_run": len(trace),
                    "stop_mse_epoch_1": float(trace.stop_mse_scaled.iloc[0]),
                    "stop_mse_best": float(trace.loc[trace.epoch.eq(best), "stop_mse_scaled"].iloc[0]),
                    "stop_mse_last": float(trace.stop_mse_scaled.iloc[-1]),
                    "train_mse_best": float(trace.loc[trace.epoch.eq(best), "train_mse_scaled"].iloc[0]),
                    "train_mse_last": float(trace.train_mse_scaled.iloc[-1]),
                })
    return pd.DataFrame(records).sort_values(["model_id", "fold", "seed"])


def circular_block_interval(daily_differences: np.ndarray, block: int,
                            *, draws: int = 4000) -> tuple[float, float]:
    """Descriptive sensitivity interval; no model selection or p-value."""
    rng = np.random.default_rng(6124 + block)
    n = len(daily_differences)
    blocks_needed = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(draws, blocks_needed))
    offsets = np.arange(block)
    indices = (starts[:, :, None] + offsets[None, None, :]) % n
    sample_means = daily_differences[indices.reshape(draws, -1)[:, :n]].mean(axis=1)
    return tuple(np.quantile(sample_means, [.025, .975]).astype(float))


def final_loss_intervals(predictions: pd.DataFrame) -> pd.DataFrame:
    final = predictions.loc[predictions.fold.eq("final_holdout")].copy()
    fuzzy = final.loc[final.model_id.eq("Fuzzy2")]
    baseline = final.loc[final.model_id.eq("LSTM32")]
    zero = final.loc[final.model_id.eq("Z0")]
    records = []
    for label, other in (("Z0", zero), ("LSTM32", baseline)):
        left = fuzzy[["seed", "window_id", "y_true", "y_pred"]]
        if label == "Z0":
            right = other[["window_id", "y_pred"]]
            paired = left.merge(right, on="window_id", suffixes=("_fuzzy", "_other"), validate="many_to_one")
        else:
            right = other[["seed", "window_id", "y_pred"]]
            paired = left.merge(right, on=["seed", "window_id"], suffixes=("_fuzzy", "_other"), validate="one_to_one")
        paired["squared_loss_difference"] = ((paired.y_true - paired.y_pred_fuzzy) ** 2 -
                                               (paired.y_true - paired.y_pred_other) ** 2)
        daily = paired.groupby("window_id", sort=True).squared_loss_difference.mean().to_numpy()
        for block in (5, 10, 20):
            low, high = circular_block_interval(daily, block)
            records.append({"challenger": "Fuzzy2", "benchmark": label, "days": len(daily),
                            "seed_aggregation": "mean daily paired loss across 3 shared-date seeds",
                            "mean_squared_loss_difference": float(daily.mean()),
                            "block_days": block, "descriptive_95pct_low": low,
                            "descriptive_95pct_high": high})
    return pd.DataFrame(records)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = Dataset(ROOT)
    predictions = load_predictions(data)
    tables = {
        "forecast_diagnostics.csv": forecast_diagnostics(data, predictions),
        "expert_diagnostics.csv": expert_diagnostics(),
        "stopping_diagnostics.csv": stopping_diagnostics(),
        "final_paired_loss_block_intervals.csv": final_loss_intervals(predictions),
    }
    for name, frame in tables.items():
        frame.to_csv(OUT / name, index=False)
    print(json.dumps({"status": "posthoc_diagnostic_only", "tables": {name: len(frame)
                       for name, frame in tables.items()}}, indent=2))


if __name__ == "__main__":
    main()
