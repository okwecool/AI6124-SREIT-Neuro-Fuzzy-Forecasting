"""Ex-post convex-mixture lower error bound for the saved V1 Fuzzy2 experts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.project_eval import Dataset, FORECAST_KEY, VALIDATION_FOLDS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/v12/oracle_expert_headroom"
CONTRACT = ROOT / "docs/v12_oracle_expert_contract_2026-11-07.json"
SOURCE = ROOT / "output/development"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def oracle_convex(y: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best hindsight convex mix of two *fixed* experts, never an inference rule."""
    y, e1, e2 = (np.asarray(v, dtype=float) for v in (y, e1, e2))
    if y.shape != e1.shape or y.shape != e2.shape or not all(np.isfinite(v).all() for v in (y, e1, e2)):
        raise ValueError("Oracle inputs must be finite and share a shape")
    difference = e1 - e2
    alpha = np.full(y.shape, 0.5, dtype=float)
    np.divide(y - e2, difference, out=alpha, where=difference != 0)
    alpha = np.clip(alpha, 0.0, 1.0)
    return alpha, alpha * e1 + (1 - alpha) * e2


def score(frame: pd.DataFrame, column: str) -> dict:
    error = frame[column].to_numpy(dtype=float) - frame.y_true.to_numpy(dtype=float)
    return {"n": len(frame), "rmse": float(np.sqrt(np.mean(error * error))),
            "mae": float(np.mean(np.abs(error)))}


def run() -> dict:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refuse to overwrite saved oracle diagnostic: {OUT}")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract["experiment_id"] != "V12-fixed-expert-oracle-headroom-01":
        raise ValueError("Unexpected oracle contract")
    data = Dataset(ROOT)
    predictions = pd.read_csv(SOURCE / "predictions.csv")
    cases = pd.read_csv(SOURCE / "cases.csv")
    rules = pd.read_csv(SOURCE / "rules.csv")
    data.validate_fuzzy_diagnostics(predictions, cases, rules)
    p = predictions.loc[(predictions.model_id == "Fuzzy2") & (predictions.source_group == "all")
                        & (predictions.rule_count == 2)].copy()
    r = rules.loc[(rules.model_id == "Fuzzy2") & (rules.source_group == "all")
                  & (rules.rule_count == 2)].copy()
    if set(p.fold) != set(VALIDATION_FOLDS) or set(p.seed) != {11, 23, 37}:
        raise ValueError("Unexpected Fuzzy2 development runs")
    wide = r.pivot(index=FORECAST_KEY, columns="rule_id", values=["alpha", "expert_raw"])
    if set(wide["expert_raw"].columns) != {1, 2}:
        raise ValueError("Expected exactly two expert columns")
    wide.columns = [f"{name}_{rule_id}" for name, rule_id in wide.columns]
    frame = p.merge(wide.reset_index(), on=FORECAST_KEY, validate="one_to_one")
    if len(frame) != len(p) or frame[FORECAST_KEY].duplicated().any():
        raise ValueError("Missing or duplicated expert predictions")
    frame["y_true"] = data.y[frame.window_id.to_numpy(dtype=int)]
    y = frame.y_true.to_numpy(dtype=float)
    e1 = frame.expert_raw_1.to_numpy(dtype=float)
    e2 = frame.expert_raw_2.to_numpy(dtype=float)
    a, optimal = oracle_convex(y, e1, e2)
    frame["oracle_alpha_1"] = a
    frame["oracle_convex"] = optimal
    frame["equal_mix"] = 0.5 * (e1 + e2)
    frame["best_hard"] = np.where(np.abs(y - e1) <= np.abs(y - e2), e1, e2)
    frame["zero"] = 0.0
    frame["expert_spread_abs"] = np.abs(e1 - e2)
    if not np.allclose(frame.y_pred, frame.alpha_1 * e1 + frame.alpha_2 * e2, atol=1e-6, rtol=0):
        raise ValueError("Saved mixture does not match its experts")
    if (np.abs(y - optimal) > np.abs(y - frame.y_pred.to_numpy(dtype=float)) + 1e-10).any():
        raise ValueError("Oracle cannot be worse on any row")
    methods = {"ActualFuzzy2": "y_pred", "OracleConvex": "oracle_convex",
               "OracleHard": "best_hard", "EqualMix": "equal_mix", "Expert1": "expert_raw_1",
               "Expert2": "expert_raw_2", "Zero": "zero"}
    records = []
    for (fold, seed), part in frame.groupby(["fold", "seed"], sort=True):
        for name, column in methods.items():
            records.append({"fold": fold, "seed": int(seed), "model": name, **score(part, column)})
    by_fold_seed = pd.DataFrame(records)
    pooled = []
    for seed, part in frame.groupby("seed", sort=True):
        for name, column in methods.items():
            pooled.append({"seed": int(seed), "model": name, **score(part, column)})
    pooled = pd.DataFrame(pooled)
    summary = pooled.groupby("model", sort=True)[["rmse", "mae"]].agg(["mean", "min", "max"])
    summary.columns = ["_".join(col) for col in summary.columns]
    summary = summary.reset_index()
    z = float(summary.loc[summary.model.eq("Zero"), "rmse_mean"].iloc[0])
    o = float(summary.loc[summary.model.eq("OracleConvex"), "rmse_mean"].iloc[0])
    actual = float(summary.loc[summary.model.eq("ActualFuzzy2"), "rmse_mean"].iloc[0])
    decision = {"oracle_rmse_gain_vs_zero_pct": 100 * (1 - o / z),
                "oracle_rmse_gain_vs_actual_pct": 100 * (1 - o / actual),
                "oracle_beats_zero_by_five_pct": bool(o <= 0.95 * z),
                "oracle_interior_fraction": float(np.mean((a > 0) & (a < 1))),
                "expert_spread_abs_median": float(np.median(frame.expert_spread_abs)),
                "expert_spread_abs_mean": float(frame.expert_spread_abs.mean()),
                "n_dates_per_seed": int(frame.groupby("seed").size().iloc[0]),
                "note": "Ex-post unattainable bound on saved experts; development labels used only for diagnosis"}
    OUT.mkdir(parents=True, exist_ok=False)
    frame.sort_values(["fold", "seed", "window_id"]).to_csv(OUT / "daily.csv", index=False)
    by_fold_seed.to_csv(OUT / "by_fold_seed.csv", index=False)
    pooled.to_csv(OUT / "pooled_by_seed.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps({
        "contract_sha256": sha(CONTRACT), "code_sha256": sha(Path(__file__)),
        "predictions_sha256": sha(SOURCE / "predictions.csv"),
        "cases_sha256": sha(SOURCE / "cases.csv"),
        "rules_sha256": sha(SOURCE / "rules.csv"),
        "target_sha256": sha(ROOT / "data/processed/srt_windows_unscaled.npz")}, indent=2), encoding="utf-8")
    return decision


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
