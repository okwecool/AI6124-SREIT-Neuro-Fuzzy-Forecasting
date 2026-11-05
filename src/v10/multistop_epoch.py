"""Compare two forward stopping blocks with the original V1 single stop."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.export_contract import fuzzy_batch_tables
from src.models import build_model
from src.project_eval import Dataset, PREDICTION_COLUMNS, Standardizer, VALIDATION_FOLDS
from src.train import (BATCH_SIZE, MAX_EPOCHS, SEEDS, evaluation_loss,
                       seed_everything, train_epoch)


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs/v10_multistop_epoch_contract_2026-11-05.json"
OUT = ROOT / "output/v10/multistop_epoch"
MODELS = {"Fuzzy2": "V10Fuzzy2", "Gate2": "V10Gate2", "LSTM3": "V10LSTM3"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def forward_blocks(data: Dataset, fold: str):
    if fold not in VALIDATION_FOLDS:
        raise ValueError("V10 accepts only original development folds")
    train = data.ids(fold, "train")
    width = math.ceil(0.15 * len(train))
    blocks = (("A", train[:len(train) - 2 * width], train[len(train) - 2 * width:len(train) - width]),
              ("B", train[:len(train) - width], train[len(train) - width:]))
    if min(len(x) for _, fit, stop in blocks for x in (fit, stop)) < 1:
        raise ValueError("Empty forward block")
    for _, fit, stop in blocks:
        if len(set(fit) & set(stop)) or data.index.target_date.iloc[fit].max() > data.index.forecast_date.iloc[stop].min():
            raise ValueError("Forward block chronology is invalid")
    if len(set(blocks[0][2]) & set(blocks[1][2])):
        raise ValueError("Forward stop blocks overlap")
    return blocks


def block_arrays(data: Dataset, fit: np.ndarray, stop: np.ndarray):
    scale = Standardizer.fit(data.x[fit], data.y[fit])
    x_fit = torch.from_numpy(scale.x_transform(data.x[fit]))
    y_fit = torch.from_numpy(scale.y_transform(data.y[fit]))
    x_stop = torch.from_numpy(scale.x_transform(data.x[stop]))
    y_stop = torch.from_numpy(scale.y_transform(data.y[stop]))
    zero = torch.from_numpy(scale.y_transform(np.zeros(len(stop), dtype=float)))
    zero_mse = float(((zero - y_stop) ** 2).mean())
    if zero_mse <= 0:
        raise ValueError("Zero-return denominator must be positive")
    return x_fit, y_fit, x_stop, y_stop, zero_mse


def trace_block(data: Dataset, model_id: str, fold: str, seed: int,
                block: str, fit: np.ndarray, stop: np.ndarray) -> pd.DataFrame:
    x_fit, y_fit, x_stop, y_stop, zero_mse = block_arrays(data, fit, stop)
    seed_everything(seed)
    model = build_model(model_id, 6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    rows = []
    for epoch in range(MAX_EPOCHS + 1):
        train_loss = (evaluation_loss(model, x_fit, y_fit) if epoch == 0 else
                      train_epoch(model, optimizer, x_fit, y_fit, seed=seed, epoch=epoch))
        stop_loss = evaluation_loss(model, x_stop, y_stop)
        if not (np.isfinite(train_loss) and np.isfinite(stop_loss)):
            raise FloatingPointError("Nonfinite V10 epoch curve")
        rows.append({"model_id": model_id, "fold": fold, "seed": seed, "block": block,
                     "epoch": epoch, "n_fit": len(fit), "n_stop": len(stop),
                     "train_mse_scaled": train_loss, "stop_mse_scaled": stop_loss,
                     "zero_mse_scaled": zero_mse, "stop_to_zero_mse": stop_loss / zero_mse})
    return pd.DataFrame(rows)


def original_manifest(model_id: str, fold: str, seed: int) -> dict:
    path = ROOT / "output/runs/development" / fold / "all" / model_id / f"seed_{seed}" / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def select_epoch(curves: pd.DataFrame, old_manifest: dict) -> dict:
    if set(curves.block) != {"A", "B"} or set(curves.epoch) != set(range(MAX_EPOCHS + 1)):
        raise ValueError("Incomplete forward-block epoch curves")
    a = curves.loc[curves.block.eq("A")].sort_values("epoch")
    b = curves.loc[curves.block.eq("B")].sort_values("epoch")
    old = old_manifest["inner_trace"]
    old_stop = np.array([row["stop_mse_scaled"] for row in old])
    old_train = np.array([row["train_mse_scaled"] for row in old])
    replay_stop_max_diff = float(np.max(np.abs(b.stop_mse_scaled.to_numpy()[1:len(old) + 1] - old_stop)))
    replay_train_max_diff = float(np.max(np.abs(b.train_mse_scaled.to_numpy()[1:len(old) + 1] - old_train)))
    if max(replay_stop_max_diff, replay_train_max_diff) > 1e-6:
        raise RuntimeError("Latest V10 block does not replay original V1 training")
    mean_ratio = (a.stop_to_zero_mse.to_numpy() + b.stop_to_zero_mse.to_numpy()) / 2
    chosen = int(np.argmin(mean_ratio[1:]) + 1)
    best_a = int(np.argmin(a.stop_to_zero_mse.to_numpy()[1:]) + 1)
    best_b = int(np.argmin(b.stop_to_zero_mse.to_numpy()[1:]) + 1)
    return {"model_id": str(a.model_id.iloc[0]), "fold": str(a.fold.iloc[0]),
            "seed": int(a.seed.iloc[0]), "selected_epoch": chosen,
            "v1_selected_epoch": int(old_manifest["best_epoch"]),
            "block_A_best_epoch": best_a, "block_B_best_epoch_full_150": best_b,
            "block_A_epoch0_beats_all_trained": bool(a.stop_to_zero_mse.iloc[0] < a.stop_to_zero_mse.iloc[best_a]),
            "block_B_epoch0_beats_all_trained": bool(b.stop_to_zero_mse.iloc[0] < b.stop_to_zero_mse.iloc[best_b]),
            "mean_ratio_at_selection": float(mean_ratio[chosen]),
            "mean_ratio_at_epoch0": float(mean_ratio[0]),
            "block_A_ratio_at_selection": float(a.stop_to_zero_mse.iloc[chosen]),
            "block_B_ratio_at_selection": float(b.stop_to_zero_mse.iloc[chosen]),
            "v1_replay_stop_max_abs_diff": replay_stop_max_diff,
            "v1_replay_train_max_abs_diff": replay_train_max_diff}


def outer_refit(data: Dataset, model_id: str, fold: str, seed: int, epoch_count: int):
    if epoch_count < 1 or epoch_count > MAX_EPOCHS:
        raise ValueError("Only trained epochs 1..150 are selectable")
    outer = data.prepare(fold, "all", inner_stop=False)
    seed_everything(seed)
    model = build_model(model_id, 6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    x_train, y_train = torch.from_numpy(outer.x_train), torch.from_numpy(outer.y_train)
    training = [train_epoch(model, optimizer, x_train, y_train, seed=seed, epoch=epoch)
                for epoch in range(1, epoch_count + 1)]
    model.eval()
    x_eval = torch.from_numpy(outer.x_eval)
    predictions, cases, rules = [], [], []
    with torch.no_grad():
        for start in range(0, len(x_eval), BATCH_SIZE):
            end = start + BATCH_SIZE
            ids = outer.eval_ids[start:end]
            batch = x_eval[start:end]
            if model_id == "Fuzzy2":
                values, diagnostics = model(batch, return_diagnostics=True)
                pred, case, rule = fuzzy_batch_tables(model_id="Fuzzy2", source_group="all",
                    fold=fold, seed=seed, window_ids=ids, prediction=values,
                    diagnostics=diagnostics, scaler=outer.scaler)
                predictions.append(pred)
                cases.append(case)
                rules.append(rule)
            else:
                raw = outer.scaler.y_inverse(model(batch).numpy())
                predictions.append(pd.DataFrame([(model_id, "all", 0, fold, seed, int(i), float(v))
                    for i, v in zip(ids, raw)], columns=PREDICTION_COLUMNS))
    base_prediction = pd.concat(predictions, ignore_index=True)
    data.validate_predictions(base_prediction)
    if model_id == "Fuzzy2":
        case_table, rule_table = pd.concat(cases, ignore_index=True), pd.concat(rules, ignore_index=True)
        data.validate_fuzzy_diagnostics(base_prediction, case_table, rule_table)
    else:
        case_table = rule_table = None
    base_prediction["model_id"] = MODELS[model_id]
    base_prediction["rule_count"] = 0
    return base_prediction, case_table, rule_table, training


def compare(data: Dataset, predictions: pd.DataFrame):
    historic = pd.read_csv(ROOT / "output/development/predictions.csv")
    historic = historic.loc[historic.source_group.eq("all") & historic.model_id.isin(MODELS), PREDICTION_COLUMNS]
    joined = pd.concat((predictions, historic, data.zero_predictions()), ignore_index=True)
    data.validate_predictions(joined)
    fold_seed = data.forecast_scores(joined)
    pooled_seed = data.pooled_validation_scores(joined)
    pooled = pooled_seed.groupby("model_id", as_index=False).agg(
        rmse=("return_rmse", "mean"), mae=("return_mae", "mean"), seeds=("seed", "nunique"))
    folds = fold_seed.groupby(["model_id", "fold"], as_index=False).agg(
        rmse=("return_rmse", "mean"), mae=("return_mae", "mean"))
    scores = pooled.set_index("model_id")
    candidate, original, zero = (scores.loc[name] for name in ("V10Fuzzy2", "Fuzzy2", "Z0"))
    pair = fold_seed.loc[fold_seed.model_id.eq("V10Fuzzy2"), ["fold", "seed", "return_rmse"]].merge(
        fold_seed.loc[fold_seed.model_id.eq("Fuzzy2"), ["fold", "seed", "return_rmse"]],
        on=["fold", "seed"], suffixes=("_v10", "_v1"), validate="one_to_one")
    fold_wins = {}
    for reference in ("Fuzzy2", "Z0"):
        table = folds.loc[folds.model_id.eq("V10Fuzzy2"), ["fold", "rmse"]].merge(
            folds.loc[folds.model_id.eq(reference), ["fold", "rmse"]],
            on="fold", suffixes=("_v10", "_reference"), validate="one_to_one")
        fold_wins[reference] = int((table.rmse_v10 < table.rmse_reference).sum())
    paired_wins = int((pair.return_rmse_v10 < pair.return_rmse_v1).sum())
    decision = {"experiment_id": "V10-H22-two-forward-stop-epoch-selection-01",
        "v10_fuzzy_rmse": float(candidate.rmse), "v1_fuzzy_rmse": float(original.rmse),
        "z0_rmse": float(zero.rmse),
        "gain_vs_v1_fuzzy_pct": float(100 * (1 - candidate.rmse / original.rmse)),
        "gain_vs_z0_pct": float(100 * (1 - candidate.rmse / zero.rmse)),
        "fold_wins": fold_wins, "paired_wins_vs_v1_fuzzy": paired_wins,
        "mae_within_z0_1pct": bool(candidate.mae <= 1.01 * zero.mae),
        "candidate_gate": bool(candidate.rmse <= 0.99 * min(original.rmse, zero.rmse)
            and all(value >= 2 for value in fold_wins.values()) and paired_wins >= 6
            and candidate.mae <= 1.01 * zero.mae),
        "substantial_goal": bool(candidate.rmse <= 0.97 * min(float(scores.loc[name].rmse)
            for name in ("Fuzzy2", "Gate2", "LSTM3", "Z0"))),
        "scope": "development-only; no final-period score"}
    return pooled.sort_values("rmse"), folds.sort_values(["model_id", "fold"]), fold_seed, decision


def run():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refuse to overwrite V10: {OUT}")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if (contract["experiment_id"] != "V10-H22-two-forward-stop-epoch-selection-01"
            or tuple(contract["fixed_task"]["folds"]) != VALIDATION_FOLDS
            or tuple(contract["fixed_task"]["seeds"]) != SEEDS
            or tuple(contract["arms"]) != tuple(MODELS.values())):
        raise ValueError("Unexpected V10 contract")
    torch.set_num_threads(1)
    data = Dataset(ROOT)
    paths = [CONTRACT, Path(__file__), ROOT / "src/models.py", ROOT / "src/train.py",
        ROOT / "src/project_eval.py", ROOT / "src/export_contract.py",
        ROOT / "data/processed/srt_windows_unscaled.npz",
        ROOT / "data/processed/fold_assignments.csv",
        ROOT / "output/development/predictions.csv"]
    paths.extend(ROOT / "output/runs/development" / fold / "all" / model_id / f"seed_{seed}" / "manifest.json"
                 for model_id in MODELS for fold in VALIDATION_FOLDS for seed in SEEDS)
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    curves, selections, predictions, diagnostics, refit_traces = [], [], [], [], []
    started = time.perf_counter()
    for model_id in MODELS:
        for fold in VALIDATION_FOLDS:
            blocks = forward_blocks(data, fold)
            for seed in SEEDS:
                frame = pd.concat([trace_block(data, model_id, fold, seed, name, fit, stop)
                                   for name, fit, stop in blocks], ignore_index=True)
                selection = select_epoch(frame, original_manifest(model_id, fold, seed))
                pred, cases, rules, training = outer_refit(data, model_id, fold, seed,
                                                            selection["selected_epoch"])
                curves.append(frame)
                selections.append(selection)
                predictions.append(pred)
                refit_traces.append({"model_id": model_id, "fold": fold, "seed": seed,
                                     "selected_epoch": selection["selected_epoch"],
                                     "train_mse_scaled": training})
                if cases is not None:
                    diagnostics.append((fold, seed, cases, rules))
                print(json.dumps({"model": model_id, "fold": fold, "seed": seed,
                    "epoch_v1": selection["v1_selected_epoch"],
                    "epoch_v10": selection["selected_epoch"],
                    "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)
    prediction_table = pd.concat(predictions, ignore_index=True)
    data.validate_predictions(prediction_table)
    pooled, folds, fold_seed, decision = compare(data, prediction_table)
    if hashes != {str(path.relative_to(ROOT)): sha(path) for path in paths}:
        raise RuntimeError("V10 dependency changed during fitting")
    OUT.mkdir(parents=True, exist_ok=False)
    pd.concat(curves, ignore_index=True).to_csv(OUT / "epoch_curves.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "selected_epochs.csv", index=False)
    prediction_table.to_csv(OUT / "predictions.csv", index=False)
    pooled.to_csv(OUT / "comparison_pooled.csv", index=False)
    folds.to_csv(OUT / "comparison_by_fold.csv", index=False)
    fold_seed.to_csv(OUT / "comparison_by_fold_seed.csv", index=False)
    for fold, seed, cases, rules in diagnostics:
        folder = OUT / "fuzzy_diagnostics" / fold / f"seed_{seed}"
        folder.mkdir(parents=True, exist_ok=False)
        cases.to_csv(folder / "cases.csv", index=False)
        rules.to_csv(folder / "rules.csv", index=False)
    (OUT / "manifests.json").write_text(json.dumps({"dependency_sha256": hashes,
        "outer_refits": refit_traces, "elapsed_seconds": time.perf_counter() - started}, indent=2), encoding="utf-8")
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return decision


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
