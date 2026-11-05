"""Test within-window SRT masking as a training-only 3D representation task."""

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
from src.models import FuzzyK, build_model
from src.project_eval import Dataset, PREDICTION_COLUMNS, VALIDATION_FOLDS
from src.train import (BATCH_SIZE, MAX_EPOCHS, MIN_IMPROVEMENT, PATIENCE,
                       SEEDS, evaluation_loss, seed_everything, train_epoch)


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs/v6_representation_contract_2026-11-05.json"
OUT = ROOT / "output/v6/representation_pretrain"
PRETRAIN_EPOCHS = 20
PRETRAIN_LR = 0.001
ARMS = {
    "V6RandomFrozenF2": ("Fuzzy2", False, True),
    "V6PreFrozenF2": ("Fuzzy2", True, True),
    "V6PreF2": ("Fuzzy2", True, False),
    "V6PreL3": ("LSTM3", True, False),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def masked_history(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Hide only the final five *historical* SRT-derived rows."""
    if x.ndim != 3 or x.shape[1:] != (20, 6):
        raise ValueError("Expected original 20x6 history")
    target = x[:, -5:, 0].clone()
    masked = x.clone()
    masked[:, -5:, :3] = 0.0
    return masked, target


def pretrain_representation(x_train: np.ndarray, seed: int) -> tuple[dict, dict]:
    """Uses only scaled training X; no primary target or validation window is accepted."""
    seed_everything(seed)
    model = FuzzyK(6, 2)
    decoder = nn.Linear(3, 5)
    optimizer = torch.optim.AdamW(
        [*model.encoder.parameters(), *model.projection.parameters(), *decoder.parameters()],
        lr=PRETRAIN_LR, weight_decay=0.0001)
    x = torch.from_numpy(x_train)
    masked, target = masked_history(x)
    losses = []
    started = time.perf_counter()
    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        model.train()
        decoder.train()
        total = 0.0
        generator = torch.Generator().manual_seed(seed + 10000 + epoch)
        for ids in torch.randperm(len(x), generator=generator).split(BATCH_SIZE):
            optimizer.zero_grad(set_to_none=True)
            z = torch.tanh(model.projection(model.encoder(masked[ids])))
            loss = nn.functional.mse_loss(decoder(z), target[ids])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite auxiliary pretraining loss")
            loss.backward()
            nn.utils.clip_grad_norm_(
                [*model.encoder.parameters(), *model.projection.parameters(), *decoder.parameters()], 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        losses.append(total / len(x))
    state = {"encoder": {name: value.detach().clone() for name, value in model.encoder.state_dict().items()},
             "projection": {name: value.detach().clone() for name, value in model.projection.state_dict().items()}}
    return state, {"train_rows": len(x), "epochs": PRETRAIN_EPOCHS,
                   "loss_first": losses[0], "loss_last": losses[-1],
                   "loss_trace": losses, "seconds": time.perf_counter() - started}


def make_model(model_id: str, seed: int, state: dict | None, frozen: bool) -> nn.Module:
    seed_everything(seed)
    model = build_model(model_id, 6)
    if state is not None:
        model.encoder.load_state_dict(state["encoder"])
        model.projection.load_state_dict(state["projection"])
    if frozen:
        for part in (model.encoder, model.projection):
            for parameter in part.parameters():
                parameter.requires_grad_(False)
    return model


def fit_arm(data: Dataset, arm: str, fold: str, seed: int,
            pre_inner: dict, pre_outer: dict) -> tuple[pd.DataFrame, pd.DataFrame | None,
                                                        pd.DataFrame | None, dict]:
    if arm not in ARMS or fold not in VALIDATION_FOLDS or seed not in SEEDS:
        raise ValueError("Only fixed development arms, folds and seeds are permitted")
    model_id, use_pretraining, frozen = ARMS[arm]
    inner = data.prepare(fold, "all", inner_stop=True)
    outer = data.prepare(fold, "all", inner_stop=False)
    model = make_model(model_id, seed, pre_inner if use_pretraining else None, frozen)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                   lr=0.001, weight_decay=0.0001)
    x_fit = torch.from_numpy(inner.x_train)
    y_fit = torch.from_numpy(inner.y_train)
    x_stop = torch.from_numpy(inner.x_eval)
    y_stop = torch.from_numpy(inner.scaler.y_transform(inner.y_eval_raw))
    best_loss, best_epoch, stale = float("inf"), 0, 0
    trace = []
    began = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_epoch(model, optimizer, x_fit, y_fit, seed=seed, epoch=epoch)
        stop_loss = evaluation_loss(model, x_stop, y_stop)
        if not math.isfinite(stop_loss):
            raise FloatingPointError("Nonfinite stopping loss")
        trace.append({"epoch": epoch, "train_mse_scaled": train_loss, "stop_mse_scaled": stop_loss})
        if stop_loss <= best_loss - MIN_IMPROVEMENT:
            best_loss, best_epoch, stale = stop_loss, epoch, 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    inner_seconds = time.perf_counter() - began
    model = make_model(model_id, seed, pre_outer if use_pretraining else None, frozen)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                   lr=0.001, weight_decay=0.0001)
    x_outer = torch.from_numpy(outer.x_train)
    y_outer = torch.from_numpy(outer.y_train)
    began = time.perf_counter()
    refit_losses = [train_epoch(model, optimizer, x_outer, y_outer, seed=seed, epoch=epoch)
                    for epoch in range(1, best_epoch + 1)]
    refit_seconds = time.perf_counter() - began
    model.eval()
    with torch.no_grad():
        x_eval = torch.from_numpy(outer.x_eval)
        if model_id == "Fuzzy2":
            values, diagnostics = model(x_eval, return_diagnostics=True)
            fuzzy_pred, cases, rules = fuzzy_batch_tables(model_id="Fuzzy2", source_group="all",
                fold=fold, seed=seed, window_ids=outer.eval_ids, prediction=values,
                diagnostics=diagnostics, scaler=outer.scaler)
            data.validate_fuzzy_diagnostics(fuzzy_pred, cases, rules, complete=False)
            raw = fuzzy_pred.y_pred.to_numpy(dtype=float)
            mean_max_alpha = float(np.max(diagnostics["alpha"].numpy(), axis=1).mean())
        else:
            raw = outer.scaler.y_inverse(model(x_eval).numpy())
            cases = rules = None
            mean_max_alpha = None
    predictions = pd.DataFrame([(arm, "all", 0, fold, seed, int(i), float(v))
                                for i, v in zip(outer.eval_ids, raw)], columns=PREDICTION_COLUMNS)
    manifest = {"arm": arm, "base_model": model_id, "fold": fold, "seed": seed,
                "uses_pretraining": use_pretraining, "representation_frozen": frozen,
                "inner_fit_rows": len(inner.train_ids), "inner_stop_rows": len(inner.eval_ids),
                "outer_fit_rows": len(outer.train_ids), "eval_rows": len(outer.eval_ids),
                "best_epoch": best_epoch, "best_inner_stop_mse_scaled": best_loss,
                "inner_trace": trace, "outer_refit_train_mse_scaled": refit_losses,
                "inner_seconds": inner_seconds, "refit_seconds": refit_seconds,
                "mean_max_alpha": mean_max_alpha}
    return predictions, cases, rules, manifest


def summarize(data: Dataset, predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame,
                                                                   pd.DataFrame, dict]:
    historical = pd.read_csv(ROOT / "output/development/predictions.csv")
    historical = historical.loc[historical.source_group.eq("all")
        & historical.model_id.isin(("Fuzzy2", "LSTM3", "Gate2")), PREDICTION_COLUMNS]
    comparison = pd.concat((historical, data.zero_predictions(), predictions), ignore_index=True)
    data.validate_predictions(comparison)
    seed_scores = data.pooled_validation_scores(comparison)
    pooled = seed_scores.groupby("model_id", as_index=False).agg(
        rmse=("return_rmse", "mean"), mae=("return_mae", "mean"), seeds=("seed", "nunique"))
    fold_seed = data.forecast_scores(comparison)
    by_fold = fold_seed.groupby(["model_id", "fold"], as_index=False).agg(
        rmse=("return_rmse", "mean"), mae=("return_mae", "mean"))
    scores = pooled.set_index("model_id")
    candidate, old, zero = scores.loc["V6PreF2"], scores.loc["Fuzzy2"], scores.loc["Z0"]
    pair = fold_seed.loc[fold_seed.model_id.eq("V6PreF2"), ["fold", "seed", "return_rmse"]].merge(
        fold_seed.loc[fold_seed.model_id.eq("Fuzzy2"), ["fold", "seed", "return_rmse"]],
        on=["fold", "seed"], suffixes=("_candidate", "_v1"), validate="one_to_one")
    folds = by_fold.loc[by_fold.model_id.eq("V6PreF2"), ["fold", "rmse"]].merge(
        by_fold.loc[by_fold.model_id.eq("Z0"), ["fold", "rmse"]],
        on="fold", suffixes=("_candidate", "_z0"), validate="one_to_one")
    paired_wins = int((pair.return_rmse_candidate < pair.return_rmse_v1).sum())
    fold_wins = int((folds.rmse_candidate < folds.rmse_z0).sum())
    strongest_fixed = min(float(scores.loc[name].rmse) for name in ("Fuzzy2", "LSTM3", "Gate2", "Z0"))
    decision = {"experiment_id": "V6-H18-within-window-SRT-mask-pretraining-01",
                "pretrained_fuzzy_rmse": float(candidate.rmse), "v1_fuzzy_rmse": float(old.rmse),
                "z0_rmse": float(zero.rmse),
                "gain_vs_v1_fuzzy_pct": float(100 * (1 - candidate.rmse / old.rmse)),
                "gain_vs_z0_pct": float(100 * (1 - candidate.rmse / zero.rmse)),
                "paired_wins_vs_v1_fuzzy": paired_wins, "fold_wins_vs_z0": fold_wins,
                "candidate_gate": bool(candidate.rmse <= .99 * min(old.rmse, zero.rmse)
                                       and fold_wins >= 2 and paired_wins >= 6
                                       and candidate.mae <= 1.01 * zero.mae),
                "substantial_goal": bool(candidate.rmse <= .97 * strongest_fixed),
                "scope": "development-only; no final holdout evaluation"}
    return pooled.sort_values("rmse"), by_fold.sort_values(["model_id", "fold"]), fold_seed, decision


def run() -> dict:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refuse to overwrite V6 results: {OUT}")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if (contract["experiment_id"] != "V6-H18-within-window-SRT-mask-pretraining-01"
            or tuple(contract["fixed_primary_task"]["folds"]) != VALIDATION_FOLDS
            or tuple(contract["fixed_primary_task"]["seeds"]) != SEEDS
            or tuple(contract["arms"]) != tuple(ARMS)):
        raise ValueError("Unexpected representation experiment contract")
    data = Dataset(ROOT)
    predictions, manifests, auxiliary, diagnostics = [], [], [], []
    for fold in VALIDATION_FOLDS:
        for seed in SEEDS:
            inner = data.prepare(fold, "all", inner_stop=True)
            outer = data.prepare(fold, "all", inner_stop=False)
            pre_inner, info_inner = pretrain_representation(inner.x_train, seed)
            pre_outer, info_outer = pretrain_representation(outer.x_train, seed)
            auxiliary.append({"fold": fold, "seed": seed, "inner": info_inner, "outer": info_outer})
            for arm in ARMS:
                pred, cases, rules, manifest = fit_arm(data, arm, fold, seed, pre_inner, pre_outer)
                predictions.append(pred)
                manifests.append(manifest)
                if cases is not None:
                    diagnostics.append((arm, fold, seed, cases, rules))
    prediction_table = pd.concat(predictions, ignore_index=True)
    pooled, by_fold, fold_seed, decision = summarize(data, prediction_table)
    OUT.mkdir(parents=True, exist_ok=False)
    prediction_table.to_csv(OUT / "predictions.csv", index=False)
    pooled.to_csv(OUT / "comparison_pooled.csv", index=False)
    by_fold.to_csv(OUT / "comparison_by_fold.csv", index=False)
    fold_seed.to_csv(OUT / "comparison_by_fold_seed.csv", index=False)
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    for arm, fold, seed, cases, rules in diagnostics:
        path = OUT / "diagnostics" / arm / fold / f"seed_{seed}"
        path.mkdir(parents=True, exist_ok=False)
        cases.to_csv(path / "cases.csv", index=False)
        rules.to_csv(path / "rules.csv", index=False)
    (OUT / "manifests.json").write_text(json.dumps({"experiment_id": contract["experiment_id"],
        "contract_sha256": digest(CONTRACT), "source_sha256": digest(Path(__file__)),
        "model_source_sha256": digest(ROOT / "src/models.py"),
        "training_source_sha256": digest(ROOT / "src/train.py"),
        "dataset_sha256": digest(ROOT / "data/processed/srt_windows_unscaled.npz"),
        "folds_sha256": digest(ROOT / "data/processed/fold_assignments.csv"),
        "historical_predictions_sha256": digest(ROOT / "output/development/predictions.csv"),
        "auxiliary": auxiliary, "fits": manifests}, indent=2), encoding="utf-8")
    return decision


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
