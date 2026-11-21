"""Weakly align the original fuzzy/gate rules to known historical volatility."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.export_contract import fuzzy_batch_tables
from src.models import build_model
from src.project_eval import Dataset, PREDICTION_COLUMNS, VALIDATION_FOLDS
from src.train import (BATCH_SIZE, MAX_EPOCHS, MIN_IMPROVEMENT, PATIENCE,
                       SEEDS, evaluation_loss, seed_everything, train_epoch)


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs/v8_regime_supervision_contract_2026-11-05.json"
OUT = ROOT / "output/v8/regime_supervision"
AUX_WEIGHT = 0.05
ARMS = {"V8FuzzyControl": ("Fuzzy2", False),
        "V8FuzzyState": ("Fuzzy2", True),
        "V8GateControl": ("Gate2", False),
        "V8GateState": ("Gate2", True)}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def volatility_regime(data: Dataset, train_ids: np.ndarray,
                      other_ids: np.ndarray) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Use only observed 20-day volatility at the last day of each saved window."""
    if not len(train_ids) or not len(other_ids):
        raise ValueError("Nonempty train and comparison windows required")
    threshold = float(np.median(data.x[train_ids, -1, 2]))
    train = torch.from_numpy((data.x[train_ids, -1, 2] >= threshold).astype(np.float32))
    other = torch.from_numpy((data.x[other_ids, -1, 2] >= threshold).astype(np.float32))
    return threshold, train, other


def train_epoch_state(model: nn.Module, optimizer: torch.optim.Optimizer,
                      x: torch.Tensor, y: torch.Tensor, state: torch.Tensor,
                      *, seed: int, epoch: int) -> tuple[float, float, float]:
    if len(x) != len(y) or len(y) != len(state):
        raise ValueError("Unaligned regime-supervision training arrays")
    model.train()
    generator = torch.Generator().manual_seed(seed + epoch)
    totals = np.zeros(3, dtype=float)
    for ids in torch.randperm(len(x), generator=generator).split(BATCH_SIZE):
        optimizer.zero_grad(set_to_none=True)
        pred, diag = model(x[ids], return_diagnostics=True)
        return_loss = nn.functional.mse_loss(pred, y[ids])
        alpha = diag["alpha"]
        regime_logit = torch.log(alpha[:, 1].clamp_min(1e-9)) - torch.log(alpha[:, 0].clamp_min(1e-9))
        state_loss = nn.functional.binary_cross_entropy_with_logits(regime_logit, state[ids])
        loss = return_loss + AUX_WEIGHT * state_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite V8 training objective")
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters() if p.requires_grad):
            raise FloatingPointError("Missing/nonfinite V8 gradient")
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        totals += np.array([float(return_loss.detach()), float(state_loss.detach()), float(loss.detach())]) * len(ids)
    return tuple((totals / len(x)).tolist())


def fit_arm(data: Dataset, arm: str, fold: str, seed: int):
    if arm not in ARMS or fold not in VALIDATION_FOLDS or seed not in SEEDS:
        raise ValueError("Only fixed V8 development arms, folds and seeds are permitted")
    model_id, state_supervised = ARMS[arm]
    inner = data.prepare(fold, "all", inner_stop=True)
    inner_threshold, train_state, stop_state = volatility_regime(data, inner.train_ids, inner.eval_ids)
    seed_everything(seed)
    model = build_model(model_id, 6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    x_fit = torch.from_numpy(inner.x_train)
    y_fit = torch.from_numpy(inner.y_train)
    x_stop = torch.from_numpy(inner.x_eval)
    y_stop = torch.from_numpy(inner.scaler.y_transform(inner.y_eval_raw))
    best, best_epoch, stale = float("inf"), 0, 0
    trace = []
    started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        if state_supervised:
            return_loss, state_loss, total_loss = train_epoch_state(
                model, optimizer, x_fit, y_fit, train_state, seed=seed, epoch=epoch)
        else:
            return_loss = train_epoch(model, optimizer, x_fit, y_fit, seed=seed, epoch=epoch)
            state_loss, total_loss = None, return_loss
        stop_loss = evaluation_loss(model, x_stop, y_stop)
        if not math.isfinite(stop_loss):
            raise FloatingPointError("Nonfinite V8 stop loss")
        trace.append({"epoch": epoch, "train_return_mse_scaled": return_loss,
                      "train_regime_bce": state_loss, "train_total": total_loss,
                      "stop_return_mse_scaled": stop_loss})
        if stop_loss <= best - MIN_IMPROVEMENT:
            best, best_epoch, stale = stop_loss, epoch, 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    inner_seconds = time.perf_counter() - started
    outer = data.prepare(fold, "all", inner_stop=False)
    outer_threshold, outer_state, eval_state = volatility_regime(data, outer.train_ids, outer.eval_ids)
    seed_everything(seed)
    model = build_model(model_id, 6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    x_outer = torch.from_numpy(outer.x_train)
    y_outer = torch.from_numpy(outer.y_train)
    started = time.perf_counter()
    refit_trace = []
    for epoch in range(1, best_epoch + 1):
        if state_supervised:
            refit_trace.append(train_epoch_state(model, optimizer, x_outer, y_outer, outer_state,
                                                 seed=seed, epoch=epoch))
        else:
            refit_trace.append((train_epoch(model, optimizer, x_outer, y_outer,
                                            seed=seed, epoch=epoch), None, None))
    outer_seconds = time.perf_counter() - started
    model.eval()
    predictions, diagnostics, cases, rules = [], [], [], []
    x_eval = torch.from_numpy(outer.x_eval)
    with torch.no_grad():
        for start in range(0, len(x_eval), BATCH_SIZE):
            end = start + BATCH_SIZE
            values, diag = model(x_eval[start:end], return_diagnostics=True)
            ids = outer.eval_ids[start:end]
            alpha = diag["alpha"].numpy()
            expert = outer.scaler.y_inverse(diag["expert_predictions"].numpy())
            z = diag["z"].numpy()
            for i, window_id in enumerate(ids):
                diagnostics.append({"arm": arm, "fold": fold, "seed": seed,
                    "window_id": int(window_id), "regime_high": int(eval_state[start + i]),
                    "alpha_low": float(alpha[i, 0]), "alpha_high": float(alpha[i, 1]),
                    "expert_low_raw": float(expert[i, 0]), "expert_high_raw": float(expert[i, 1]),
                    "z1": float(z[i, 0]), "z2": float(z[i, 1]), "z3": float(z[i, 2])})
            if model_id == "Fuzzy2":
                pred, case, rule = fuzzy_batch_tables(model_id="Fuzzy2", source_group="all",
                    fold=fold, seed=seed, window_ids=ids, prediction=values,
                    diagnostics=diag, scaler=outer.scaler)
                predictions.append(pred)
                cases.append(case)
                rules.append(rule)
            else:
                raw = outer.scaler.y_inverse(values.numpy())
                predictions.append(pd.DataFrame([("Gate2", "all", 0, fold, seed, int(i), float(v))
                    for i, v in zip(ids, raw)], columns=PREDICTION_COLUMNS))
    base_prediction = pd.concat(predictions, ignore_index=True)
    if cases:
        case_table, rule_table = pd.concat(cases, ignore_index=True), pd.concat(rules, ignore_index=True)
        data.validate_fuzzy_diagnostics(base_prediction, case_table, rule_table)
    else:
        case_table = rule_table = None
    base_prediction["model_id"] = arm
    base_prediction["rule_count"] = 0
    data.validate_predictions(base_prediction)
    diagnostic_table = pd.DataFrame(diagnostics)
    manifest = {"arm": arm, "base_model": model_id, "state_supervised": state_supervised,
                "fold": fold, "seed": seed, "inner_threshold": inner_threshold,
                "outer_threshold": outer_threshold,
                "inner_fit_rows": len(inner.train_ids), "inner_stop_rows": len(inner.eval_ids),
                "outer_fit_rows": len(outer.train_ids), "eval_rows": len(outer.eval_ids),
                "best_epoch": best_epoch, "best_inner_stop_return_mse_scaled": best,
                "inner_trace": trace, "outer_refit_trace": refit_trace,
                "inner_seconds": inner_seconds, "outer_seconds": outer_seconds,
                "eval_regime_accuracy": float(np.mean((diagnostic_table.alpha_high.to_numpy() >= 0.5)
                                                      == diagnostic_table.regime_high.to_numpy())),
                "mean_max_alpha": float(np.maximum(diagnostic_table.alpha_low,
                                                    diagnostic_table.alpha_high).mean())}
    return base_prediction, diagnostic_table, case_table, rule_table, manifest


def summarize(data: Dataset, predictions: pd.DataFrame, manifests: list[dict]):
    historic = pd.read_csv(ROOT / "output/development/predictions.csv")
    historic = historic.loc[historic.source_group.eq("all") &
        historic.model_id.isin(("Fuzzy2", "Gate2", "LSTM3")), PREDICTION_COLUMNS]
    comparison = pd.concat((predictions, historic, data.zero_predictions()), ignore_index=True)
    data.validate_predictions(comparison)
    fold_seed = data.forecast_scores(comparison)
    pooled_seed = data.pooled_validation_scores(comparison)
    pooled = pooled_seed.groupby("model_id", as_index=False).agg(
        rmse=("return_rmse", "mean"), mae=("return_mae", "mean"), seeds=("seed", "nunique"))
    folds = fold_seed.groupby(["model_id", "fold"], as_index=False).agg(
        rmse=("return_rmse", "mean"), mae=("return_mae", "mean"))
    score = pooled.set_index("model_id")
    candidate, control, zero = (score.loc[name] for name in ("V8FuzzyState", "V8FuzzyControl", "Z0"))
    pair = fold_seed.loc[fold_seed.model_id.eq("V8FuzzyState"), ["fold", "seed", "return_rmse"]].merge(
        fold_seed.loc[fold_seed.model_id.eq("V8FuzzyControl"), ["fold", "seed", "return_rmse"]],
        on=["fold", "seed"], suffixes=("_candidate", "_control"), validate="one_to_one")
    fold_wins = {}
    for reference in ("V8FuzzyControl", "Z0"):
        joined = folds.loc[folds.model_id.eq("V8FuzzyState"), ["fold", "rmse"]].merge(
            folds.loc[folds.model_id.eq(reference), ["fold", "rmse"]], on="fold",
            suffixes=("_candidate", "_reference"), validate="one_to_one")
        fold_wins[reference] = int((joined.rmse_candidate < joined.rmse_reference).sum())
    control_checks = {}
    for arm, original in (("V8FuzzyControl", "Fuzzy2"), ("V8GateControl", "Gate2")):
        joined = comparison.loc[comparison.model_id.eq(arm), ["fold", "seed", "window_id", "y_pred"]].merge(
            comparison.loc[comparison.model_id.eq(original), ["fold", "seed", "window_id", "y_pred"]],
            on=["fold", "seed", "window_id"], suffixes=("_control", "_v1"), validate="one_to_one")
        if len(joined) != len(comparison.loc[comparison.model_id.eq(arm)]):
            raise ValueError("V8 control and V1 have different evaluation keys")
        control_checks[arm] = float(np.abs(joined.y_pred_control - joined.y_pred_v1).max())
    if max(control_checks.values()) > 1e-6:
        raise ValueError(f"V8 control diverges from V1: {control_checks}")
    decision = {"experiment_id": "V8-H20-train-only-volatility-rule-supervision-01",
        "fuzzy_state_rmse": float(candidate.rmse), "fuzzy_control_rmse": float(control.rmse),
        "z0_rmse": float(zero.rmse),
        "gain_vs_fuzzy_control_pct": float(100 * (1 - candidate.rmse / control.rmse)),
        "gain_vs_z0_pct": float(100 * (1 - candidate.rmse / zero.rmse)),
        "paired_wins_vs_fuzzy_control": int((pair.return_rmse_candidate < pair.return_rmse_control).sum()),
        "fold_wins": fold_wins, "control_max_abs_diff_from_v1": control_checks,
        "candidate_gate": bool(candidate.rmse <= 0.99 * min(control.rmse, zero.rmse)
            and all(v >= 2 for v in fold_wins.values())
            and int((pair.return_rmse_candidate < pair.return_rmse_control).sum()) >= 6
            and candidate.mae <= 1.01 * zero.mae),
        "substantial_goal": bool(candidate.rmse <= 0.97 * min(float(score.loc[n].rmse)
            for n in ("Z0", "Fuzzy2", "Gate2", "LSTM3"))),
        "scope": "development-only; no final-period evaluation"}
    return pooled.sort_values("rmse"), folds.sort_values(["model_id", "fold"]), fold_seed, decision


def run() -> dict:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refuse to overwrite V8: {OUT}")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if (contract["experiment_id"] != "V8-H20-train-only-volatility-rule-supervision-01"
            or tuple(contract["fixed_task"]["folds"]) != VALIDATION_FOLDS
            or tuple(contract["fixed_task"]["seeds"]) != SEEDS
            or tuple(contract["arms"]) != tuple(ARMS)):
        raise ValueError("Unexpected V8 contract")
    torch.set_num_threads(1)
    data = Dataset(ROOT)
    predictions, diagnostics, cases_by_run, manifests = [], [], [], []
    for fold in VALIDATION_FOLDS:
        for seed in SEEDS:
            for arm in ARMS:
                pred, diag, cases, rules, manifest = fit_arm(data, arm, fold, seed)
                predictions.append(pred)
                diagnostics.append(diag)
                manifests.append(manifest)
                if cases is not None:
                    cases_by_run.append((arm, fold, seed, cases, rules))
                print(json.dumps({"arm": arm, "fold": fold, "seed": seed,
                                  "best_epoch": manifest["best_epoch"]}), flush=True)
    prediction_table = pd.concat(predictions, ignore_index=True)
    data.validate_predictions(prediction_table)
    pooled, folds, fold_seed, decision = summarize(data, prediction_table, manifests)
    dependencies = [CONTRACT, Path(__file__), ROOT / "src/models.py", ROOT / "src/train.py",
                    ROOT / "src/project_eval.py", ROOT / "data/processed/srt_windows_unscaled.npz",
                    ROOT / "data/processed/fold_assignments.csv",
                    ROOT / "output/development/predictions.csv"]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in dependencies}
    OUT.mkdir(parents=True, exist_ok=False)
    prediction_table.to_csv(OUT / "predictions.csv", index=False)
    pd.concat(diagnostics, ignore_index=True).to_csv(OUT / "regime_diagnostics.csv", index=False)
    pooled.to_csv(OUT / "comparison_pooled.csv", index=False)
    folds.to_csv(OUT / "comparison_by_fold.csv", index=False)
    fold_seed.to_csv(OUT / "comparison_by_fold_seed.csv", index=False)
    for arm, fold, seed, cases, rules in cases_by_run:
        folder = OUT / "fuzzy_diagnostics" / arm / fold / f"seed_{seed}"
        folder.mkdir(parents=True, exist_ok=False)
        cases.to_csv(folder / "cases.csv", index=False)
        rules.to_csv(folder / "rules.csv", index=False)
    (OUT / "manifests.json").write_text(json.dumps({"dependency_sha256": hashes,
        "runs": manifests}, indent=2), encoding="utf-8")
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return decision


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
