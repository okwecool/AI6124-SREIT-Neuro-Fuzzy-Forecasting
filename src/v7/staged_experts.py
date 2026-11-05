"""Development-only staged Gaussian expert experiment on the original task."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.project_eval import Dataset, PREDICTION_COLUMNS, Standardizer, VALIDATION_FOLDS
from src.train import seed_everything
from src.v3.hybrid_training import HybridHead, partition
from src.v5.shared_local import original_development_arrays, solve, step


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs/v7_staged_experts_contract_2026-11-05.json"
OUT = ROOT / "output/v7/staged_experts"
SEEDS = (11, 23, 37)
COMMON_CONFIG = {"head": "linear", "basis": "shared", "lambda_common": 0.1}
STAGE1_EPOCHS = 10
STAGE2_MAX_EPOCHS = 40
STAGE2_PATIENCE = 10
MIN_IMPROVEMENT = 0.0001
LOCAL_PENALTY = [1.0] * 4
ARMS = ("V7Common", "V7StagedGaussian", "V7StagedGate")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chronological_parts(data: Dataset, fold: str, *, inner: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fit, stop, train, evaluation = partition(data, fold)
    available = fit if inner else train
    cut = math.floor(0.75 * len(available))
    common, calibration = available[:cut], available[cut:]
    target = stop if inner else evaluation
    if min(map(len, (common, calibration, target))) < 1:
        raise ValueError("Empty chronological stage")
    index = data.index
    for left, right in ((common, calibration), (calibration, target)):
        if index.target_date.iloc[left].max() > index.forecast_date.iloc[right].min():
            raise ValueError("A training label is unavailable at the next stage")
    return common, calibration, target


def stage_arrays(x: np.ndarray, y: np.ndarray, common: np.ndarray,
                 calibration: np.ndarray, target: np.ndarray, *, target_labels: bool):
    standard = Standardizer.fit(x[common], y[common])
    scaler = Standardizer(standard.x_mean, standard.x_std, 0.0, standard.y_std)
    return (torch.from_numpy(scaler.x_transform(x[common])),
            torch.from_numpy(scaler.y_transform(y[common])),
            torch.from_numpy(scaler.x_transform(x[calibration])),
            torch.from_numpy(scaler.y_transform(y[calibration])),
            torch.from_numpy(scaler.x_transform(x[target])),
            torch.from_numpy(scaler.y_transform(y[target])) if target_labels else None, scaler)


def train_common(x: torch.Tensor, y: torch.Tensor, seed: int) -> tuple[HybridHead, torch.Tensor, list[float]]:
    seed_everything(seed)
    model = HybridHead("linear")
    model.initialize_from_training(x)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    trace = [step(model, optimizer, x, y, COMMON_CONFIG) for _ in range(STAGE1_EPOCHS)]
    model.initialize_from_training(x)
    with torch.no_grad():
        q, _ = model.design(x)
        b = solve(q, y, [0.1] * 4)
    return model, b, trace


def new_local(common: HybridHead, x_common: torch.Tensor, head: str, seed: int) -> HybridHead:
    if head not in ("gaussian", "gate"):
        raise ValueError("Unknown local gate")
    seed_everything(seed)
    model = HybridHead(head)
    model.encoder.load_state_dict(common.encoder.state_dict())
    model.projection.load_state_dict(common.projection.state_dict())
    model.initialize_from_training(x_common)
    for part in (model.encoder, model.projection):
        for parameter in part.parameters():
            parameter.requires_grad_(False)
    return model


def local_features(model: HybridHead, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
    _, diag = model.design(x)
    alpha = diag["alpha"]
    phi = (alpha[:, :1] - alpha[:, 1:]) * diag["affine"]
    return phi, diag


def fitted_local(model: HybridHead, x_cal: torch.Tensor, y_cal: torch.Tensor,
                 b: torch.Tensor, x_eval: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    phi_cal, cal_diag = local_features(model, x_cal)
    residual = y_cal.double() - cal_diag["affine"].double() @ b
    d = solve(phi_cal, residual, LOCAL_PENALTY)
    phi_eval, diag = local_features(model, x_eval)
    common = diag["affine"].double() @ b
    local = phi_eval.double() @ d
    return common + local, common, d, diag


def local_step(model: HybridHead, optimizer, x_cal: torch.Tensor,
               y_cal: torch.Tensor, b: torch.Tensor) -> float:
    optimizer.zero_grad(set_to_none=True)
    phi, diag = local_features(model, x_cal)
    residual = y_cal.double() - diag["affine"].double() @ b
    d = solve(phi, residual, LOCAL_PENALTY)
    loss = (phi.double() @ d - residual).square().mean() + d.square().sum()
    loss.backward()
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
        raise FloatingPointError("Invalid stage-2 gate gradient")
    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    optimizer.step()
    return float(loss.detach())


def select_local(common: HybridHead, b: torch.Tensor, x_common: torch.Tensor,
                 x_cal: torch.Tensor, y_cal: torch.Tensor, x_stop: torch.Tensor,
                 y_stop: torch.Tensor, head: str, seed: int) -> dict:
    model = new_local(common, x_common, head, seed)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=0.001, weight_decay=0.0001)
    with torch.no_grad():
        q_stop, _ = common.design(x_stop)
        common_loss = float(((q_stop.double() @ b - y_stop.double()) ** 2).mean())
    best, chosen, stale = common_loss, -1, 0
    trace = []
    for epoch in range(STAGE2_MAX_EPOCHS + 1):
        train_loss = local_step(model, optimizer, x_cal, y_cal, b) if epoch else None
        with torch.no_grad():
            pred, _, _, _ = fitted_local(model, x_cal, y_cal, b, x_stop)
            loss = float(((pred - y_stop.double()) ** 2).mean())
        if not np.isfinite(loss):
            raise FloatingPointError("Nonfinite local stop loss")
        trace.append({"epoch": epoch, "calibration_penalized_loss": train_loss, "stop_mse": loss})
        if loss <= best - MIN_IMPROVEMENT:
            best, chosen, stale = loss, epoch, 0
        else:
            stale += 1
        if stale >= STAGE2_PATIENCE:
            break
    return {"head": head, "common_stop_mse": common_loss, "best_stop_mse": best,
            "chosen_epoch": chosen, "trace": trace}


def refit_local(common: HybridHead, b: torch.Tensor, x_common: torch.Tensor,
                x_cal: torch.Tensor, y_cal: torch.Tensor, x_eval: torch.Tensor,
                head: str, seed: int, chosen_epoch: int):
    model = new_local(common, x_common, head, seed)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=0.001, weight_decay=0.0001)
    for _ in range(max(0, chosen_epoch)):
        local_step(model, optimizer, x_cal, y_cal, b)
    with torch.no_grad():
        pred, common_pred, d, diag = fitted_local(model, x_cal, y_cal, b, x_eval)
        if chosen_epoch == -1:
            d = torch.zeros_like(d)
            pred = common_pred
        q = diag["affine"].double()
        experts = torch.stack((q @ (b + d), q @ (b - d)), dim=1)
        torch.testing.assert_close((diag["alpha"].double() * experts).sum(1), pred,
                                   rtol=1e-6, atol=1e-7)
    return pred.numpy(), common_pred.numpy(), d.numpy(), diag, experts.numpy()


def rows_for_arm(arm: str, fold: str, seed: int, ids: np.ndarray,
                 scaled: np.ndarray, scaler: Standardizer) -> pd.DataFrame:
    return pd.DataFrame([(arm, "all", 0, fold, seed, int(i), float(v))
                         for i, v in zip(ids, scaler.y_inverse(scaled))],
                        columns=PREDICTION_COLUMNS)


def run_one(data: Dataset, x: np.ndarray, y: np.ndarray, fold: str, seed: int):
    common_i, cal_i, stop = chronological_parts(data, fold, inner=True)
    xc, yc, xcal, ycal, xs, ys, _ = stage_arrays(x, y, common_i, cal_i, stop, target_labels=True)
    started = time.perf_counter()
    inner_common, inner_b, inner_trace = train_common(xc, yc, seed)
    selections = {head: select_local(inner_common, inner_b, xc, xcal, ycal, xs, ys, head, seed)
                  for head in ("gaussian", "gate")}
    inner_seconds = time.perf_counter() - started
    common_o, cal_o, evaluation = chronological_parts(data, fold, inner=False)
    xc, yc, xcal, ycal, xe, _, scale = stage_arrays(x, y, common_o, cal_o, evaluation,
                                                    target_labels=False)
    started = time.perf_counter()
    outer_common, b, outer_trace = train_common(xc, yc, seed)
    with torch.no_grad():
        q_eval, _ = outer_common.design(xe)
        common_scaled = (q_eval.double() @ b).numpy()
    output = [rows_for_arm("V7Common", fold, seed, evaluation, common_scaled, scale)]
    details = []
    for head, arm in (("gaussian", "V7StagedGaussian"), ("gate", "V7StagedGate")):
        prediction, common_pred, d, diag, experts = refit_local(
            outer_common, b, xc, xcal, ycal, xe, head, seed, selections[head]["chosen_epoch"])
        np.testing.assert_allclose(common_pred, common_scaled, rtol=1e-6, atol=1e-7)
        output.append(rows_for_arm(arm, fold, seed, evaluation, prediction, scale))
        alpha = diag["alpha"].detach().numpy()
        details.extend({"arm": arm, "fold": fold, "seed": seed, "window_id": int(i),
                        "common_raw": float(scale.y_inverse(c)),
                        "local_raw": float(scale.y_inverse(p - c)),
                        "alpha1": float(a[0]), "alpha2": float(a[1]),
                        "expert1_raw": float(scale.y_inverse(e[0])),
                        "expert2_raw": float(scale.y_inverse(e[1]))}
                       for i, p, c, a, e in zip(evaluation, prediction, common_pred, alpha, experts))
        selections[head]["outer_local_coefficients"] = d.tolist()
        selections[head]["mean_max_alpha"] = float(alpha.max(1).mean())
    manifest = {"fold": fold, "seed": seed,
                "inner_common_ids": [int(common_i[0]), int(common_i[-1])],
                "inner_calibration_ids": [int(cal_i[0]), int(cal_i[-1])],
                "inner_stop_ids": [int(stop[0]), int(stop[-1])],
                "outer_common_ids": [int(common_o[0]), int(common_o[-1])],
                "outer_calibration_ids": [int(cal_o[0]), int(cal_o[-1])],
                "outer_evaluation_ids": [int(evaluation[0]), int(evaluation[-1])],
                "common_coefficients": b.tolist(), "inner_common_trace": inner_trace,
                "outer_common_trace": outer_trace, "selections": selections,
                "inner_seconds": inner_seconds, "outer_seconds": time.perf_counter() - started}
    return pd.concat(output, ignore_index=True), pd.DataFrame(details), manifest


def comparisons(data: Dataset, predictions: pd.DataFrame):
    old = pd.read_csv(ROOT / "output/development/predictions.csv")
    old = old.loc[old.source_group.eq("all") & old.model_id.isin(("Fuzzy2", "LSTM3", "Gate2")), PREDICTION_COLUMNS]
    v5 = pd.read_csv(ROOT / "output/v5/shared_local/predictions.csv")
    v5 = v5.loc[v5.arm.isin(("gaussian_shared_local", "gate_shared_local")), :].copy()
    v5["model_id"] = v5.arm.map({"gaussian_shared_local": "V5Gaussian", "gate_shared_local": "V5Gate"})
    v5["rule_count"] = 0
    comparison = pd.concat((predictions, old, v5[list(PREDICTION_COLUMNS)], data.zero_predictions()),
                           ignore_index=True)
    data.validate_predictions(comparison)
    fold_seed = data.forecast_scores(comparison)
    pooled_seed = data.pooled_validation_scores(comparison)
    pooled = pooled_seed.groupby("model_id", as_index=False).agg(
        rmse=("return_rmse", "mean"), mae=("return_mae", "mean"), seeds=("seed", "nunique"))
    folds = fold_seed.groupby(["model_id", "fold"], as_index=False).agg(
        rmse=("return_rmse", "mean"), mae=("return_mae", "mean"))
    score = pooled.set_index("model_id")
    candidate = score.loc["V7StagedGaussian"]
    common = score.loc["V7Common"]
    z0 = score.loc["Z0"]
    pair = fold_seed.loc[fold_seed.model_id.eq("V7StagedGaussian"), ["fold", "seed", "return_rmse"]].merge(
        fold_seed.loc[fold_seed.model_id.eq("V7Common"), ["fold", "seed", "return_rmse"]],
        on=["fold", "seed"], suffixes=("_candidate", "_common"), validate="one_to_one")
    fold_pair = folds.loc[folds.model_id.eq("V7StagedGaussian"), ["fold", "rmse"]]
    wins = {}
    for reference in ("V7Common", "Z0"):
        merged = fold_pair.merge(folds.loc[folds.model_id.eq(reference), ["fold", "rmse"]],
                                 on="fold", suffixes=("_candidate", "_reference"), validate="one_to_one")
        wins[reference] = int((merged.rmse_candidate < merged.rmse_reference).sum())
    return pooled.sort_values("rmse"), folds.sort_values(["model_id", "fold"]), fold_seed, {
        "experiment_id": "V7-H19-chronological-calibration-staged-experts-01",
        "gaussian_rmse": float(candidate.rmse), "common_rmse": float(common.rmse), "z0_rmse": float(z0.rmse),
        "gain_vs_common_pct": float(100 * (1 - candidate.rmse / common.rmse)),
        "gain_vs_z0_pct": float(100 * (1 - candidate.rmse / z0.rmse)),
        "fold_wins": wins, "paired_wins_vs_common": int((pair.return_rmse_candidate < pair.return_rmse_common).sum()),
        "mae_within_z0_1pct": bool(candidate.mae <= 1.01 * z0.mae),
        "scope": "development-only; no final holdout evaluation"}


def run() -> dict:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refuse to overwrite V7: {OUT}")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if (contract["experiment_id"] != "V7-H19-chronological-calibration-staged-experts-01"
            or tuple(contract["fixed_primary_task"]["folds"]) != VALIDATION_FOLDS
            or tuple(contract["fixed_primary_task"]["seeds"]) != SEEDS
            or tuple(contract["arms"]) != ARMS):
        raise ValueError("Unexpected V7 contract")
    torch.set_num_threads(1)
    data = Dataset(ROOT)
    x, y = original_development_arrays(data)
    dependencies = [CONTRACT, Path(__file__), ROOT / "src/v3/hybrid_training.py",
                    ROOT / "src/v5/shared_local.py", ROOT / "src/project_eval.py",
                    ROOT / "data/processed/srt_windows_unscaled.npz",
                    ROOT / "data/processed/fold_assignments.csv"]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in dependencies}
    predictions, details, manifests = [], [], []
    for fold in VALIDATION_FOLDS:
        for seed in SEEDS:
            pred, detail, manifest = run_one(data, x, y, fold, seed)
            predictions.append(pred)
            details.append(detail)
            manifests.append(manifest)
            print(json.dumps({"fold": fold, "seed": seed,
                              "gaussian_epoch": manifest["selections"]["gaussian"]["chosen_epoch"],
                              "gate_epoch": manifest["selections"]["gate"]["chosen_epoch"]}), flush=True)
    prediction_table = pd.concat(predictions, ignore_index=True)
    data.validate_predictions(prediction_table)
    pooled, folds, fold_seed, decision = comparisons(data, prediction_table)
    accepted = sum(m["selections"]["gaussian"]["chosen_epoch"] >= 0 for m in manifests)
    decision["gaussian_nonzero_local_selections"] = accepted
    decision["candidate_gate"] = bool(
        decision["gaussian_rmse"] <= 0.99 * min(decision["common_rmse"], decision["z0_rmse"])
        and all(n >= 2 for n in decision["fold_wins"].values())
        and decision["paired_wins_vs_common"] >= 6
        and decision["mae_within_z0_1pct"] and accepted >= 6)
    strongest = min(float(pooled.set_index("model_id").loc[name].rmse)
                    for name in ("Z0", "Fuzzy2", "LSTM3", "Gate2", "V5Gaussian", "V5Gate"))
    decision["substantial_goal"] = bool(decision["gaussian_rmse"] <= 0.97 * strongest)
    if hashes != {str(p.relative_to(ROOT)): sha(p) for p in dependencies}:
        raise RuntimeError("Dependency changed during V7 fitting")
    OUT.mkdir(parents=True, exist_ok=False)
    prediction_table.to_csv(OUT / "predictions.csv", index=False)
    pd.concat(details, ignore_index=True).to_csv(OUT / "components.csv", index=False)
    pooled.to_csv(OUT / "comparison_pooled.csv", index=False)
    folds.to_csv(OUT / "comparison_by_fold.csv", index=False)
    fold_seed.to_csv(OUT / "comparison_by_fold_seed.csv", index=False)
    (OUT / "manifests.json").write_text(json.dumps({"contract_sha256": hashes[str(CONTRACT.relative_to(ROOT))],
        "dependency_sha256": hashes, "runs": manifests}, indent=2), encoding="utf-8")
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return decision


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
