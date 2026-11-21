"""Train unchanged Fuzzy2 on distribution-adjusted residuals; score raw returns."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.export_contract import fuzzy_batch_tables
from src.models import build_model
from src.project_eval import Dataset, VALIDATION_FOLDS
from src.train import MAX_EPOCHS, MIN_IMPROVEMENT, PATIENCE, SEEDS, evaluation_loss, seed_everything, train_epoch
from src.v2.screen_distributions import EVENTS, announced_adjustments


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs/v2_residual_fuzzy_contract_2026-11-04.json"
OUT = ROOT / "output/v2/residual_fuzzy"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ResidualDataset(Dataset):
    """Subtract realized corporate actions from completed historical labels only."""

    def __init__(self, root: Path):
        super().__init__(root)
        forecast_closes = self.daily.set_index("date").srt_close
        label_dividends = self.daily.set_index("date").srt_dividend_sgd_on_date
        close_t = self.index.forecast_date.map(forecast_closes).to_numpy(dtype=float)
        distribution = self.index.target_date.map(label_dividends).to_numpy(dtype=float)
        eligible = self.index.target_date.le(pd.Timestamp("2025-09-30")).to_numpy()
        distribution = np.where(eligible, distribution, 0.0)
        if not np.isfinite(close_t).all() or not np.isfinite(distribution).all() or (close_t <= distribution).any():
            raise ValueError("Invalid completed training-label distribution")
        self.realized_adjustment = np.log1p(-distribution / close_t)
        self.y = (self.y.astype(np.float64) - self.realized_adjustment).astype(np.float32)


def run_one(residual_data: ResidualDataset, original_data: Dataset,
            adjustments: dict[int, float], fold: str, seed: int) -> tuple[pd.DataFrame, dict]:
    if fold not in VALIDATION_FOLDS or seed not in SEEDS:
        raise ValueError("Only the fixed development fold/seed grid is allowed")
    inner = residual_data.prepare(fold, "all", inner_stop=True)
    seed_everything(seed)
    model = build_model("Fuzzy2", 6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    x_train = torch.from_numpy(inner.x_train)
    y_train = torch.from_numpy(inner.y_train)
    x_stop = torch.from_numpy(inner.x_eval)
    y_stop = torch.from_numpy(inner.scaler.y_transform(inner.y_eval_raw))
    best_loss, best_epoch, stale = float("inf"), 0, 0
    trace = []
    began = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_epoch(model, optimizer, x_train, y_train, seed=seed, epoch=epoch)
        stop_loss = evaluation_loss(model, x_stop, y_stop)
        if not np.isfinite(stop_loss):
            raise FloatingPointError("Nonfinite inner stopping loss")
        trace.append({"epoch": epoch, "train_mse_scaled": train_loss, "stop_mse_scaled": stop_loss})
        if stop_loss <= best_loss - MIN_IMPROVEMENT:
            best_loss, best_epoch, stale = stop_loss, epoch, 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    inner_seconds = time.perf_counter() - began
    outer = residual_data.prepare(fold, "all", inner_stop=False)
    seed_everything(seed)
    model = build_model("Fuzzy2", 6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    began = time.perf_counter()
    refit_losses = []
    x_outer = torch.from_numpy(outer.x_train)
    y_outer = torch.from_numpy(outer.y_train)
    for epoch in range(1, best_epoch + 1):
        refit_losses.append(train_epoch(model, optimizer, x_outer, y_outer, seed=seed, epoch=epoch))
    refit_seconds = time.perf_counter() - began
    model.eval()
    with torch.no_grad():
        y_scaled, diagnostic = model(torch.from_numpy(outer.x_eval), return_diagnostics=True)
    pred, cases, rules = fuzzy_batch_tables(model_id="Fuzzy2", source_group="all", fold=fold, seed=seed,
        window_ids=outer.eval_ids, prediction=y_scaled, diagnostics=diagnostic, scaler=outer.scaler)
    offset = pd.Series({int(i): float(adjustments[int(i)]) for i in outer.eval_ids})
    pred["y_pred"] += pred.window_id.map(offset)
    cases["y_mix_raw"] += cases.window_id.map(offset)
    rules["expert_raw"] += rules.window_id.map(offset)
    original_data.validate_fuzzy_diagnostics(pred, cases, rules)
    run_dir = OUT / "runs" / fold / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    pred.to_csv(run_dir / "predictions.csv", index=False)
    cases.to_csv(run_dir / "cases.csv", index=False)
    rules.to_csv(run_dir / "rules.csv", index=False)
    torch.save({"state_dict": model.state_dict(), "best_epoch": best_epoch,
        "scaler": {"x_mean": outer.scaler.x_mean.tolist(), "x_std": outer.scaler.x_std.tolist(),
                   "y_mean": outer.scaler.y_mean, "y_std": outer.scaler.y_std}}, run_dir / "checkpoint.pt")
    manifest = {"experiment_id": "V2-H9-distribution-residual-fuzzy-01", "fold": fold, "seed": seed,
        "best_epoch": best_epoch, "best_inner_stop_mse_scaled": best_loss,
        "inner_fit_rows": len(inner.train_ids), "inner_stop_rows": len(inner.eval_ids),
        "outer_fit_rows": len(outer.train_ids), "eval_rows": len(outer.eval_ids),
        "inner_seconds": inner_seconds, "refit_seconds": refit_seconds,
        "inner_trace": trace, "outer_refit_train_mse_scaled": refit_losses,
        "model_source_sha256": sha256(ROOT / "src/models.py"),
        "v1_training_source_sha256": sha256(ROOT / "src/train.py"),
        "source_sha256": sha256(Path(__file__)), "contract_sha256": sha256(CONTRACT),
        "data_sha256": sha256(ROOT / "data/processed/srt_windows_unscaled.npz"),
        "event_file_sha256": sha256(EVENTS)}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return pred, manifest


def run() -> dict:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refuse to overwrite results: {OUT}")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if tuple(contract["folds"]) != VALIDATION_FOLDS or tuple(contract["seeds"]) != SEEDS:
        raise ValueError("Unexpected residual-fuzzy contract")
    original = Dataset(ROOT)
    residual = ResidualDataset(ROOT)
    events = pd.read_csv(EVENTS)
    schedule = announced_adjustments(original, events)
    adjustment = dict(zip(schedule.window_id, schedule.adjustment))
    all_pred, manifests = [], []
    for fold in VALIDATION_FOLDS:
        for seed in SEEDS:
            pred, manifest = run_one(residual, original, adjustment, fold, seed)
            all_pred.append(pred)
            manifests.append(manifest)
    predictions = pd.concat(all_pred, ignore_index=True)
    original.validate_predictions(predictions)
    scored = predictions.copy()
    scored["model_id"] = "ResidualFuzzy2"
    scored["rule_count"] = 0
    h8 = pd.read_csv(ROOT / "output/v2/announced_distribution_screen/predictions.csv")
    h8 = h8.loc[h8.model_id.isin(("Z0", "DivSchedule", "SchedulePlusFuzzy2"))]
    comparison = pd.concat([h8, scored], ignore_index=True)
    pooled = original.pooled_validation_scores(comparison).sort_values("return_rmse")
    by_fold = original.forecast_scores(comparison).sort_values(["model_id", "fold", "seed"])
    means = pooled.groupby("model_id", as_index=False).agg(
        mean_pooled_rmse=("return_rmse", "mean"), mean_pooled_mae=("return_mae", "mean"))
    schedule_score = means.loc[means.model_id.eq("DivSchedule")].iloc[0]
    candidate = means.loc[means.model_id.eq("ResidualFuzzy2")].iloc[0]
    fold_means = by_fold.groupby(["model_id", "fold"], as_index=False).return_rmse.mean()
    fold_join = fold_means.loc[fold_means.model_id.eq("ResidualFuzzy2"), ["fold", "return_rmse"]].merge(
        fold_means.loc[fold_means.model_id.eq("DivSchedule"), ["fold", "return_rmse"]],
        on="fold", suffixes=("_candidate", "_schedule"), validate="one_to_one")
    fold_wins = int((fold_join.return_rmse_candidate < fold_join.return_rmse_schedule).sum())
    pair = by_fold.loc[by_fold.model_id.eq("ResidualFuzzy2"), ["fold", "seed", "return_rmse"]].merge(
        by_fold.loc[by_fold.model_id.eq("SchedulePlusFuzzy2"), ["fold", "seed", "return_rmse"]],
        on=["fold", "seed"], suffixes=("_candidate", "_naive"), validate="one_to_one")
    paired_wins = int((pair.return_rmse_candidate < pair.return_rmse_naive).sum())
    decision = {"experiment_id": contract["experiment_id"],
        "schedule_rmse": float(schedule_score.mean_pooled_rmse),
        "residual_fuzzy_mean_rmse": float(candidate.mean_pooled_rmse),
        "relative_rmse_gain_pct": float(100 * (1 - candidate.mean_pooled_rmse / schedule_score.mean_pooled_rmse)),
        "schedule_mae": float(schedule_score.mean_pooled_mae),
        "residual_fuzzy_mean_mae": float(candidate.mean_pooled_mae),
        "fold_wins_vs_schedule": fold_wins, "paired_wins_vs_naive_hybrid": paired_wins,
        "passes_development_gate": bool(candidate.mean_pooled_rmse <= 0.99 * schedule_score.mean_pooled_rmse and
                                       fold_wins >= 2 and paired_wins >= 6 and
                                       candidate.mean_pooled_mae <= 1.01 * schedule_score.mean_pooled_mae)}
    predictions.to_csv(OUT / "predictions.csv", index=False)
    pooled.to_csv(OUT / "comparison_pooled.csv", index=False)
    by_fold.to_csv(OUT / "comparison_by_fold.csv", index=False)
    (OUT / "run_manifest_index.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return decision


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
