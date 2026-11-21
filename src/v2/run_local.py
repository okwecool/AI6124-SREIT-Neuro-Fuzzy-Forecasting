"""H3: same Fuzzy2 encoder/gate with one fixed local-expert loss on development folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.export_contract import fuzzy_batch_tables
from src.models import build_model, parameter_count
from src.project_eval import Dataset, VALIDATION_FOLDS
from src.train import (BATCH_SIZE, MAX_EPOCHS, MIN_IMPROVEMENT, PATIENCE,
                       SEEDS, evaluation_loss, seed_everything)
from src.v2.local_loss import loss_components


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs/v2_experiment_contract_2026-11-04.json"
H1_BENCHMARK = ROOT / "output/v2/h1/selected_benchmark_pre_h3.json"
DEFAULT_OUT = ROOT / "output/v2/h3"
EXPERIMENT_ID = "V2-H3-local-loss-01"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, x: torch.Tensor,
                y: torch.Tensor, *, seed: int, epoch: int,
                local_weight: float) -> dict:
    model.train()
    generator = torch.Generator().manual_seed(seed + epoch)
    totals = np.zeros(3, dtype=float)
    for ids in torch.randperm(len(x), generator=generator).split(BATCH_SIZE):
        ids = ids.to(x.device)
        optimizer.zero_grad(set_to_none=True)
        if local_weight == 0:
            prediction = model(x[ids])
            loss = nn.functional.mse_loss(prediction, y[ids])
            mixed, local = loss, loss.new_zeros(())
        else:
            prediction, diagnostic = model(x[ids], return_diagnostics=True)
            loss, mixed, local = loss_components(
                prediction, diagnostic["expert_predictions"], diagnostic["alpha"],
                y[ids], local_weight=local_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite V2 local training loss")
        loss.backward()
        if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters() if parameter.requires_grad):
            raise FloatingPointError("Missing/nonfinite V2 gradient")
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        totals += np.array([float(loss.detach()), float(mixed.detach()),
                            float(local.detach())]) * len(ids)
    mean = totals / len(x)
    return {"train_total_scaled": float(mean[0]),
            "train_mixed_mse_scaled": float(mean[1]),
            "train_local_mse_scaled": float(mean[2])}


@torch.no_grad()
def training_rule_cuts(model: nn.Module, x_train: np.ndarray,
                       device: torch.device) -> dict:
    model.eval()
    weights = []
    for start in range(0, len(x_train), BATCH_SIZE):
        batch = torch.from_numpy(x_train[start:start + BATCH_SIZE]).to(device)
        _, diagnostic = model(batch, return_diagnostics=True)
        weights.append(diagnostic["alpha"].cpu().numpy())
    alpha = np.concatenate(weights)
    centers = model.centers.detach().cpu().numpy()
    scales = (0.1 + 1.9 * torch.sigmoid(model.scale_logits)).detach().cpu().numpy()
    return {"outer_train_rows": len(alpha),
            "alpha_q75_by_rule": np.quantile(alpha, 0.75, axis=0).tolist(),
            "alpha_mean_by_rule": alpha.mean(axis=0).tolist(),
            "effective_sample_size_by_rule": ((alpha.sum(axis=0) ** 2) /
                                              (alpha ** 2).sum(axis=0)).tolist(),
            "centers": centers.tolist(), "scales": scales.tolist()}


def run_one(data: Dataset, fold: str, seed: int, output_root: Path,
            *, local_weight: float = 0.1, device: str = "cpu") -> dict:
    if fold not in VALIDATION_FOLDS or seed not in SEEDS:
        raise ValueError("V2-H3 accepts only the three development folds and seeds")
    if local_weight not in (0.0, 0.1):
        raise ValueError("H3 only permits the fixed 0.1 weight or zero engineering check")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    run_dir = output_root / fold / "all" / "Fuzzy2" / f"seed_{seed}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing H3 run: {run_dir}")
    device_obj = torch.device(device)
    inner = data.prepare(fold, "all", inner_stop=True)
    seed_everything(seed)
    model = build_model("Fuzzy2", 6).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    x_fit = torch.from_numpy(inner.x_train).to(device_obj)
    y_fit = torch.from_numpy(inner.y_train).to(device_obj)
    x_stop = torch.from_numpy(inner.x_eval).to(device_obj)
    y_stop = torch.from_numpy(inner.scaler.y_transform(inner.y_eval_raw)).to(device_obj)
    started = time.perf_counter()
    trace = []
    best_loss, best_epoch, stale = float("inf"), 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        step = train_epoch(model, optimizer, x_fit, y_fit, seed=seed, epoch=epoch,
                           local_weight=local_weight)
        stop_loss = evaluation_loss(model, x_stop, y_stop)
        if not np.isfinite(stop_loss):
            raise FloatingPointError("Nonfinite V2 stopping loss")
        trace.append({"epoch": epoch, **step, "stop_mixed_mse_scaled": stop_loss})
        if stop_loss <= best_loss - MIN_IMPROVEMENT:
            best_loss, best_epoch, stale = stop_loss, epoch, 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    inner_seconds = time.perf_counter() - started

    outer = data.prepare(fold, "all", inner_stop=False)
    seed_everything(seed)
    model = build_model("Fuzzy2", 6).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    x_outer = torch.from_numpy(outer.x_train).to(device_obj)
    y_outer = torch.from_numpy(outer.y_train).to(device_obj)
    started = time.perf_counter()
    refit_trace = [train_epoch(model, optimizer, x_outer, y_outer,
                               seed=seed, epoch=epoch, local_weight=local_weight)
                   for epoch in range(1, best_epoch + 1)]
    refit_seconds = time.perf_counter() - started
    cuts = training_rule_cuts(model, outer.x_train, device_obj)

    model.eval()
    predictions, cases, rules = [], [], []
    with torch.no_grad():
        for start in range(0, len(outer.eval_ids), BATCH_SIZE):
            ids = outer.eval_ids[start:start + BATCH_SIZE]
            batch = torch.from_numpy(outer.x_eval[start:start + BATCH_SIZE]).to(device_obj)
            values, diagnostic = model(batch, return_diagnostics=True)
            pred, case, rule = fuzzy_batch_tables(
                model_id="Fuzzy2", source_group="all", fold=fold, seed=seed,
                window_ids=ids, prediction=values, diagnostics=diagnostic,
                scaler=outer.scaler)
            predictions.append(pred)
            cases.append(case)
            rules.append(rule)
    prediction_table = pd.concat(predictions, ignore_index=True)
    case_table = pd.concat(cases, ignore_index=True)
    rule_table = pd.concat(rules, ignore_index=True)
    data.validate_predictions(prediction_table)
    data.validate_fuzzy_diagnostics(prediction_table, case_table, rule_table)
    run_dir.mkdir(parents=True, exist_ok=True)
    prediction_table.to_csv(run_dir / "predictions.csv", index=False)
    case_table.to_csv(run_dir / "cases.csv", index=False)
    rule_table.to_csv(run_dir / "rules.csv", index=False)
    torch.save({"model_id": "Fuzzy2", "experiment_id": EXPERIMENT_ID,
                "state_dict": model.cpu().state_dict(), "best_epoch": best_epoch,
                "scaler": {"x_mean": outer.scaler.x_mean.tolist(),
                           "x_std": outer.scaler.x_std.tolist(),
                           "y_mean": outer.scaler.y_mean, "y_std": outer.scaler.y_std}},
               run_dir / "checkpoint.pt")
    (run_dir / "training_rule_cuts.json").write_text(json.dumps(cuts, indent=2), encoding="utf-8")
    manifest = {
        "status": "v2_h3_development" if local_weight == 0.1 else "engineering_zero_loss_equivalence",
        "experiment_id": EXPERIMENT_ID if local_weight == 0.1 else "V2-H3-lambda-zero-engineering",
        "model_id": "Fuzzy2", "source_group": "all", "rule_count": 2,
        "fold": fold, "seed": seed, "device": device,
        "local_weight": local_weight, "best_epoch": best_epoch,
        "best_inner_stop_mse_scaled": best_loss, "inner_trace": trace,
        "outer_refit_trace": refit_trace, "inner_seconds": inner_seconds,
        "refit_seconds": refit_seconds, "parameters": parameter_count(model),
        "inner_fit_rows": len(inner.train_ids), "inner_stop_rows": len(inner.eval_ids),
        "outer_fit_rows": len(outer.train_ids), "outer_eval_rows": len(outer.eval_ids),
        "contract_sha256": digest(CONTRACT), "h1_benchmark_sha256": digest(H1_BENCHMARK),
        "dataset_sha256": digest(ROOT / "data/processed/srt_windows_unscaled.npz"),
        "folds_sha256": digest(ROOT / "data/processed/fold_assignments.csv"),
        "v1_model_source_sha256": digest(ROOT / "src/models.py"),
        "v1_train_source_sha256": digest(ROOT / "src/train.py"),
        "v2_runner_source_sha256": digest(Path(__file__)),
        "v2_loss_source_sha256": digest(Path(__file__).with_name("local_loss.py")),
        "prediction_file_sha256": digest(run_dir / "predictions.csv"),
        "cases_file_sha256": digest(run_dir / "cases.csv"),
        "rules_file_sha256": digest(run_dir / "rules.csv"),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"fold": fold, "seed": seed, "best_epoch": best_epoch,
            "prediction_rows": len(prediction_table), "run_dir": str(run_dir)}


def check_zero_equivalence(data: Dataset) -> dict:
    output = ROOT / "output/runs/tmp/v2_zero_equivalence"
    new = run_one(data, "validation_1", 11, output, local_weight=0.0)
    previous = ROOT / "output/runs/development/validation_1/all/Fuzzy2/seed_11"
    old_manifest = json.loads((previous / "manifest.json").read_text(encoding="utf-8"))
    new_manifest = json.loads((Path(new["run_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    old_pred = pd.read_csv(previous / "predictions.csv").sort_values("window_id")
    new_pred = pd.read_csv(Path(new["run_dir"]) / "predictions.csv").sort_values("window_id")
    max_difference = float(np.max(np.abs(old_pred.y_pred.to_numpy() - new_pred.y_pred.to_numpy())))
    if (old_manifest["best_epoch"] != new_manifest["best_epoch"] or
        not np.array_equal(old_pred.window_id, new_pred.window_id) or
        max_difference > 1e-7):
        raise AssertionError(f"Lambda-zero V2 path differs from V1: {max_difference}")
    result = {"status": "lambda_zero_matches_v1", "fold": "validation_1", "seed": 11,
              "best_epoch": new_manifest["best_epoch"], "max_raw_prediction_difference": max_difference,
              "diagnostic_only": True}
    (ROOT / "output/runs/tmp/v2_zero_equivalence_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engineering-zero-check", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract["h3"]["local_loss_weight"] != 0.1 or not H1_BENCHMARK.exists():
        raise ValueError("Frozen H3 weight or H1 benchmark is missing")
    data = Dataset(ROOT)
    if args.engineering_zero_check:
        print(json.dumps(check_zero_equivalence(data), indent=2))
        return
    if DEFAULT_OUT.exists() and any(DEFAULT_OUT.iterdir()):
        raise FileExistsError(f"H3 output exists; use a new experiment ID: {DEFAULT_OUT}")
    DEFAULT_OUT.mkdir(parents=True, exist_ok=False)
    (DEFAULT_OUT / "run_plan.json").write_text(json.dumps({
        "experiment_id": EXPERIMENT_ID, "folds": list(VALIDATION_FOLDS),
        "seeds": list(SEEDS), "local_weight": 0.1,
        "h1_benchmark_sha256_before_runs": digest(H1_BENCHMARK),
        "contract_sha256": digest(CONTRACT)}, indent=2), encoding="utf-8")
    results = [run_one(data, fold, seed, DEFAULT_OUT, device=args.device)
               for fold in VALIDATION_FOLDS for seed in SEEDS]
    print(json.dumps({"status": "H3_runs_complete", "runs": len(results), "details": results}, indent=2))


if __name__ == "__main__":
    main()
