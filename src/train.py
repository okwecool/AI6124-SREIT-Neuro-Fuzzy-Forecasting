"""Chronological inner stopping and same-seed outer refit for development folds.

The final holdout is intentionally unavailable here. Phase 4 will use the frozen
best-epoch rule and a separate entry point after development choices are fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.export_contract import fuzzy_batch_tables
from src.models import build_model, parameter_count
from src.project_eval import Dataset, FEATURE_GROUPS, PREDICTION_COLUMNS, VALIDATION_FOLDS


SEEDS = (11, 23, 37)
BATCH_SIZE = 64
MAX_EPOCHS = 150
PATIENCE = 20
MIN_IMPROVEMENT = 1e-4


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model: nn.Module, optimizer: torch.optim.Optimizer,
                x: torch.Tensor, y: torch.Tensor, *, seed: int, epoch: int) -> float:
    model.train()
    generator = torch.Generator().manual_seed(seed + epoch)
    total = 0.0
    for ids in torch.randperm(len(x), generator=generator).split(BATCH_SIZE):
        ids = ids.to(x.device)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(x[ids]), y[ids])
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss; rerun all affected learned models at lr=0.0003")
        loss.backward()
        if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters() if parameter.requires_grad):
            raise FloatingPointError("Missing/nonfinite gradient; rerun all affected learned models at lr=0.0003")
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.detach()) * len(ids)
    return total / len(x)


@torch.no_grad()
def evaluation_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    return float(nn.functional.mse_loss(model(x), y))


def _versions() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "torch": torch.__version__}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_development(
    data: Dataset, model_id: str, source_group: str, fold: str, seed: int,
    output_root: Path, *, max_epochs: int = MAX_EPOCHS, patience: int = PATIENCE,
    device: str = "cpu", learning_rate: float = 0.001,
) -> dict:
    if fold not in VALIDATION_FOLDS:
        raise ValueError("Development runner accepts validation folds only; final holdout is sealed")
    if source_group not in FEATURE_GROUPS or seed not in SEEDS:
        raise ValueError("Use a frozen source group and seed 11, 23, or 37")
    if max_epochs < 1 or patience < 1 or learning_rate <= 0:
        raise ValueError("Training settings must be positive")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    device_obj = torch.device(device)
    input_size = len(FEATURE_GROUPS[source_group])
    inner = data.prepare(fold, source_group, inner_stop=True)
    seed_everything(seed)
    model = build_model(model_id, input_size).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    x_fit = torch.from_numpy(inner.x_train).to(device_obj)
    y_fit = torch.from_numpy(inner.y_train).to(device_obj)
    x_stop = torch.from_numpy(inner.x_eval).to(device_obj)
    y_stop = torch.from_numpy(inner.scaler.y_transform(inner.y_eval_raw)).to(device_obj)
    started = time.perf_counter()
    trace = []
    best_loss, best_epoch, stale = float("inf"), 0, 0
    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch(model, optimizer, x_fit, y_fit, seed=seed, epoch=epoch)
        stop_loss = evaluation_loss(model, x_stop, y_stop)
        if not np.isfinite(stop_loss):
            raise FloatingPointError("Nonfinite stopping loss; rerun all affected learned models at lr=0.0003")
        trace.append({"epoch": epoch, "train_mse_scaled": train_loss,
                      "stop_mse_scaled": stop_loss})
        if stop_loss <= best_loss - MIN_IMPROVEMENT:
            best_loss, best_epoch, stale = stop_loss, epoch, 0
        else:
            stale += 1
        if stale >= patience:
            break
    inner_seconds = time.perf_counter() - started

    outer = data.prepare(fold, source_group, inner_stop=False)
    seed_everything(seed)  # Same initial weights and within-epoch shuffle as inner fit.
    model = build_model(model_id, input_size).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    x_outer = torch.from_numpy(outer.x_train).to(device_obj)
    y_outer = torch.from_numpy(outer.y_train).to(device_obj)
    started = time.perf_counter()
    refit_losses = []
    for epoch in range(1, best_epoch + 1):
        refit_losses.append(train_epoch(model, optimizer, x_outer, y_outer, seed=seed, epoch=epoch))
    refit_seconds = time.perf_counter() - started

    model.eval()
    predictions, cases, rules = [], [], []
    x_eval = torch.from_numpy(outer.x_eval)
    with torch.no_grad():
        for start in range(0, len(x_eval), BATCH_SIZE):
            end = start + BATCH_SIZE
            batch = x_eval[start:end].to(device_obj)
            ids = outer.eval_ids[start:end]
            if model_id.startswith("Fuzzy"):
                values, diagnostics = model(batch, return_diagnostics=True)
                pred, case, rule = fuzzy_batch_tables(
                    model_id=model_id, source_group=source_group, fold=fold, seed=seed,
                    window_ids=ids, prediction=values, diagnostics=diagnostics,
                    scaler=outer.scaler)
                predictions.append(pred)
                cases.append(case)
                rules.append(rule)
            else:
                raw = outer.scaler.y_inverse(model(batch).cpu().numpy())
                predictions.append(pd.DataFrame([
                    (model_id, source_group, 0, fold, seed, int(window_id), float(value))
                    for window_id, value in zip(ids, raw)
                ], columns=PREDICTION_COLUMNS))
    prediction_table = pd.concat(predictions, ignore_index=True)
    data.validate_predictions(prediction_table)
    if cases:
        case_table = pd.concat(cases, ignore_index=True)
        rule_table = pd.concat(rules, ignore_index=True)
        data.validate_fuzzy_diagnostics(prediction_table, case_table, rule_table)
    run_dir = Path(output_root) / fold / source_group / model_id / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    prediction_table.to_csv(run_dir / "predictions.csv", index=False)
    if cases:
        case_table.to_csv(run_dir / "cases.csv", index=False)
        rule_table.to_csv(run_dir / "rules.csv", index=False)
    checkpoint = run_dir / "checkpoint.pt"
    torch.save({"model_id": model_id, "source_group": source_group,
                "state_dict": model.cpu().state_dict(), "best_epoch": best_epoch,
                "scaler": {"x_mean": outer.scaler.x_mean.tolist(),
                           "x_std": outer.scaler.x_std.tolist(),
                           "y_mean": outer.scaler.y_mean, "y_std": outer.scaler.y_std}}, checkpoint)
    git_revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                  text=True, check=True).stdout.strip()
    git_dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                    text=True, check=True).stdout.strip())
    manifest = {
        "status": "development" if (max_epochs, patience) == (MAX_EPOCHS, PATIENCE) else "engineering_smoke",
        "model_id": model_id, "source_group": source_group,
        "rule_count": int(model_id[5:]) if model_id.startswith("Fuzzy") else 0,
        "fold": fold, "seed": seed, "device": device, "versions": _versions(),
        "git_revision": git_revision, "git_dirty": git_dirty,
        "train_source_sha256": _sha256(Path(__file__)),
        "model_source_sha256": _sha256(Path(__file__).with_name("models.py")),
        "dataset_sha256": _sha256(data.root / "data/processed/srt_windows_unscaled.npz"),
        "fold_assignments_sha256": _sha256(data.root / "data/processed/fold_assignments.csv"),
        "input_features": list(data.features[i] for i in FEATURE_GROUPS[source_group]),
        "parameters": parameter_count(model), "inner_fit_rows": len(inner.train_ids),
        "inner_stop_rows": len(inner.eval_ids), "outer_fit_rows": len(outer.train_ids),
        "outer_validation_rows": len(outer.eval_ids), "best_epoch": best_epoch,
        "best_inner_stop_mse_scaled": best_loss, "inner_trace": trace,
        "outer_refit_train_mse_scaled": refit_losses,
        "inner_seconds": inner_seconds, "refit_seconds": refit_seconds,
        "optimizer": {"name": "AdamW", "learning_rate": learning_rate,
                      "weight_decay": 0.0001, "batch_size": BATCH_SIZE,
                      "gradient_clip_norm": 1.0},
        "stopping": {"max_epochs": max_epochs, "patience": patience,
                     "min_improvement": MIN_IMPROVEMENT},
        "checkpoint": str(checkpoint), "prediction_file": str(run_dir / "predictions.csv"),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=VALIDATION_FOLDS, required=True)
    parser.add_argument("--source-group", choices=tuple(FEATURE_GROUPS), default="all")
    parser.add_argument("--models", choices=("LSTM32", "LSTM3", "Fuzzy1", "Fuzzy2", "Fuzzy3", "Gate2"),
                        nargs="+", default=["LSTM32", "LSTM3", "Fuzzy2"])
    parser.add_argument("--seed", type=int, choices=SEEDS, default=11)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--out", type=Path, default=Path("output/runs"))
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    args = parser.parse_args()
    data = Dataset()
    for model_id in args.models:
        manifest = run_development(data, model_id, args.source_group, args.fold,
                                   args.seed, args.out, max_epochs=args.max_epochs,
                                   patience=args.patience, device=args.device)
        print(json.dumps({key: manifest[key] for key in
              ("model_id", "fold", "seed", "best_epoch", "best_inner_stop_mse_scaled",
               "inner_seconds", "refit_seconds", "prediction_file")}, indent=2))


if __name__ == "__main__":
    main()
