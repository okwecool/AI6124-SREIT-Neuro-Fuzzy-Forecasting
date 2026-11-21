"""Controlled shared/local expert training; original inputs and raw target only."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.export_contract import fuzzy_batch_tables
from src.project_eval import Dataset, PREDICTION_COLUMNS, VALIDATION_FOLDS
from src.train import seed_everything
from src.v3.hybrid_training import HybridHead, arrays, partition


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs/v5_model_only_contract_2026-11-05.json"
OUT = ROOT / "output/v5/shared_local"
HEAD_IDS = {"linear": "LSTM3", "gaussian": "Fuzzy2", "gate": "Gate2"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def original_development_arrays(data):
    """Copy exactly the frozen values, and withhold every final-period row."""
    allowed = data.index.target_date.le(pd.Timestamp("2025-09-30")).to_numpy()
    x = np.full(data.x.shape, np.nan, dtype=data.x.dtype)
    y = np.full(data.y.shape, np.nan, dtype=np.float64)
    x[allowed] = data.x[allowed]
    y[allowed] = data.y[allowed]
    return x, y


def solve(phi, y, penalty):
    """Mean-SSE ridge with positive, coefficient-specific penalties."""
    a, b = phi.double(), y.double()
    p = torch.as_tensor(penalty, dtype=a.dtype, device=a.device)
    if p.shape != (a.shape[1],) or not torch.isfinite(p).all() or (p <= 0).any():
        raise ValueError("One finite positive penalty per coefficient is required")
    return torch.linalg.solve(a.T @ a / len(a) + torch.diag(p), a.T @ b / len(a))


def design(model, x, config):
    independent, diag = model.design(x)
    if config["head"] == "linear":
        return independent, diag, [config["lambda_common"]] * 4
    if config["basis"] == "independent":
        return independent, diag, [config["lambda_expert"]] * 8
    q, alpha = diag["affine"], diag["alpha"]
    difference = alpha[:, :1] - alpha[:, 1:]
    phi = torch.cat((q, difference * q), 1)
    return phi, diag, [config["lambda_common"]] * 4 + [config["lambda_local"]] * 4


def step(model, optimizer, x, y, config):
    optimizer.zero_grad(set_to_none=True)
    phi, _, penalties = design(model, x, config)
    beta = solve(phi, y, penalties)
    loss = (phi.double() @ beta - y).square().mean() + (beta.square() * beta.new_tensor(penalties)).sum()
    loss.backward()
    if not torch.isfinite(loss) or not all(p.grad is not None and torch.isfinite(p.grad).all()
                                         for p in model.parameters()):
        raise FloatingPointError("Missing or nonfinite shared/local gradient")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def fitted(model, xt, yt, xe, config):
    phi, _, penalty = design(model, xt, config)
    beta = solve(phi, yt, penalty)
    evaluation, diag, _ = design(model, xe, config)
    return evaluation.double() @ beta, beta, diag


def components(beta, diag, config):
    q = diag.get("affine")
    if config["head"] == "linear":
        q = torch.cat((diag["z"], torch.ones_like(diag["z"][:, :1])), 1)
        common = q.double() @ beta
        return common, torch.zeros_like(common), None
    if config["basis"] == "independent":
        b = (beta[:4] + beta[4:]) / 2
        d = (beta[:4] - beta[4:]) / 2
    else:
        b, d = beta[:4], beta[4:]
    common, delta = q.double() @ b, q.double() @ d
    experts = torch.stack((common + delta, common - delta), 1)
    local = (diag["alpha"][:, 0] - diag["alpha"][:, 1]).double() * delta
    return common, local, experts


def new_model(config, seed, xt, settings):
    seed_everything(seed)
    model = HybridHead(config["head"])
    model.initialize_from_training(xt)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    return model, optimizer


def run_one(data, x, y, arm, config, fold, seed, contract):
    fit, stop, train, evaluation = partition(data, fold)
    settings = contract["training"]
    xt, yt, xs, scale = arrays(x, y, fit, stop)
    ys = torch.from_numpy(scale.y_transform(y[stop]))
    model, optimizer = new_model(config, seed, xt, settings)
    best, chosen, stale = float("inf"), 0, 0
    zero_loss = float(ys.double().square().mean())
    trace = []
    began = time.perf_counter()
    for epoch in range(settings["max_epochs"] + 1):
        objective = step(model, optimizer, xt, yt, config) if epoch else None
        pred, _, _ = fitted(model, xt, yt, xs, config)
        loss = float((pred - ys).square().mean())
        if not np.isfinite(loss):
            raise FloatingPointError("Nonfinite stopping loss")
        trace.append({"epoch": epoch, "train_penalized_objective": objective, "stop_mse": loss})
        if loss <= best - settings["min_improvement"]:
            best, chosen, stale = loss, epoch, 0
        else:
            stale += 1
        if stale >= settings["patience"]:
            break
    inner_seconds = time.perf_counter() - began
    fallback = best > zero_loss - settings["min_improvement"]
    xt, yt, xe, scale = arrays(x, y, train, evaluation)
    model, optimizer = new_model(config, seed, xt, settings)
    began = time.perf_counter()
    for _ in range(chosen):
        step(model, optimizer, xt, yt, config)
    fitted_prediction, beta_fit, diag = fitted(model, xt, yt, xe, config)
    beta = torch.zeros_like(beta_fit) if fallback else beta_fit
    prediction_scaled = torch.zeros_like(fitted_prediction) if fallback else fitted_prediction
    common, local, experts = components(beta, diag, config)
    torch.testing.assert_close(common + local, prediction_scaled, rtol=1e-5, atol=1e-7)
    model_id = HEAD_IDS[config["head"]]
    pred = pd.DataFrame([(model_id, "all", 2 if config["head"] == "gaussian" else 0,
                          fold, seed, int(i), float(p))
                         for i, p in zip(evaluation, scale.y_inverse(prediction_scaled.numpy()))],
                        columns=PREDICTION_COLUMNS)
    data.validate_predictions(pred)
    run_dir = OUT / arm / fold / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    pred.to_csv(run_dir / "predictions.csv", index=False)
    detail = pd.DataFrame({"window_id": evaluation, "common_raw": scale.y_inverse(common.numpy()),
                           "local_raw": scale.y_inverse(local.numpy()),
                           "fitted_without_fallback_raw": scale.y_inverse(fitted_prediction.numpy())})
    stats = {}
    local_rows = []
    if config["head"] != "linear":
        alpha = diag["alpha"].numpy()
        _, train_diag, _ = design(model, xt, config)
        train_alpha = train_diag["alpha"].detach().numpy()
        raw_experts = scale.y_inverse(experts.numpy())
        detail["alpha1"], detail["alpha2"] = alpha[:, 0], alpha[:, 1]
        detail["expert1_raw"], detail["expert2_raw"] = raw_experts[:, 0], raw_experts[:, 1]
        stats = {"mean_max_weight": float(alpha.max(1).mean()),
                 "fraction_max_weight_above_08": float((alpha.max(1) > .8).mean()),
                 "mean_abs_expert_difference": float(np.abs(raw_experts[:, 0] - raw_experts[:, 1]).mean()),
                 "common_rms": float(np.sqrt(np.mean(detail.common_raw ** 2))),
                 "local_rms": float(np.sqrt(np.mean(detail.local_raw ** 2)))}
        for k in (0, 1):
            cut = float(np.quantile(train_alpha[:, k], .75))
            for region, mask in (("train_upper_quartile", alpha[:, k] >= cut), ("fixed_08", alpha[:, k] > .8)):
                n = int(mask.sum())
                local_rows.append({"rule": k + 1, "region": region, "cut": cut if region == "train_upper_quartile" else .8,
                    "n": n, "small_region": n < 20, "zero_fallback": bool(fallback),
                    "own_mae": float(np.abs(raw_experts[mask, k] - y[evaluation][mask]).mean()) if n else None,
                    "other_mae": float(np.abs(raw_experts[mask, 1-k] - y[evaluation][mask]).mean()) if n else None})
        if config["head"] == "gaussian":
            diag["expert_predictions"] = experts
            _, cases, rules = fuzzy_batch_tables(model_id=model_id, source_group="all", fold=fold,
                seed=seed, window_ids=evaluation, prediction=prediction_scaled,
                diagnostics={k: diag[k] for k in ("z", "alpha", "distance2", "log_membership", "expert_predictions")},
                scaler=scale)
            data.validate_fuzzy_diagnostics(pred, cases, rules)
            cases.to_csv(run_dir / "cases.csv", index=False)
            rules.to_csv(run_dir / "rules.csv", index=False)
            support_cut = float(torch.quantile(train_diag["log_membership"].detach().max(1).values, .05))
            stats["fraction_raw_support_below_train_p05"] = float((diag["log_membership"].max(1).values < support_cut).double().mean())
            stats["centers"] = model.centers.detach().tolist()
            stats["widths"] = (torch.nn.functional.softplus(model.width_logits) + .05).detach().tolist()
    detail.to_csv(run_dir / "components.csv", index=False)
    if local_rows:
        pd.DataFrame(local_rows).to_csv(run_dir / "local_errors.csv", index=False)
    torch.save({"model_state": model.state_dict(), "config": config, "beta": beta,
                "fitted_beta": beta_fit, "scaler": {"x_mean": scale.x_mean.tolist(),
                "x_std": scale.x_std.tolist(), "y_mean": 0.0, "y_std": scale.y_std}}, run_dir / "checkpoint.pt")
    manifest = {"experiment_id": contract["experiment_id"], "arm": arm, "fold": fold, "seed": seed,
                "head": config["head"], "best_fitted_epoch": chosen, "zero_fallback": bool(fallback),
                "inner_zero_mse": zero_loss, "inner_fitted_mse": best, "trace": trace,
                "train_rows": len(train), "eval_rows": len(evaluation),
                "inner_seconds": inner_seconds, "outer_seconds": time.perf_counter() - began,
                "parameters": sum(p.numel() for p in model.parameters()) + len(beta), **stats,
                "output_sha256": {p.name: sha(p) for p in run_dir.glob("*.csv")}}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return pred, manifest


def run():
    if OUT.exists():
        raise FileExistsError("Refuse to overwrite the fixed V5 experiment")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert list(contract["arms"]) == ["linear", "gaussian_independent", "gaussian_common_matched",
                                      "gaussian_shared_local", "gate_shared_local"]
    assert tuple(contract["folds"]) == VALIDATION_FOLDS and contract["seeds"] == [11, 23, 37]
    torch.set_num_threads(1)
    data = Dataset(ROOT)
    x, y = original_development_arrays(data)
    dependencies = [CONTRACT, Path(__file__), ROOT / "src/v3/hybrid_training.py", ROOT / "src/models.py",
                    ROOT / "src/project_eval.py", ROOT / "src/train.py", ROOT / "src/export_contract.py",
                    *[ROOT / "data/processed" / name for name in ("srt_windows_unscaled.npz", "window_index.csv", "fold_assignments.csv", "srt_daily_aligned.csv")]]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in dependencies}
    OUT.mkdir(parents=True, exist_ok=False)
    allowed = np.isfinite(y)
    np.testing.assert_array_equal(x[allowed], data.x[allowed])
    np.testing.assert_array_equal(y[allowed], data.y[allowed])
    (OUT / "pre_execution.json").write_text(json.dumps({"contract": contract, "dependency_sha256": hashes,
        "original_inputs_and_targets_equal": True, "development_rows": int(allowed.sum()),
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}}, indent=2), encoding="utf-8")
    predictions, manifests = [], []
    for arm, config in contract["arms"].items():
        for fold in VALIDATION_FOLDS:
            for seed in contract["seeds"]:
                pred, manifest = run_one(data, x, y, arm, config, fold, seed, contract)
                pred["arm"] = arm
                predictions.append(pred)
                manifests.append(manifest)
                print(json.dumps({k: manifest[k] for k in ("arm", "fold", "seed", "best_fitted_epoch", "zero_fallback")}), flush=True)
    pd.concat(predictions, ignore_index=True).to_csv(OUT / "predictions.csv", index=False)
    (OUT / "manifests.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    if hashes != {str(p.relative_to(ROOT)): sha(p) for p in dependencies}:
        raise RuntimeError("A fixed dependency changed during fitting")
    (OUT / "complete.json").write_text(json.dumps({"status": "complete", "runs": len(manifests),
        "prediction_sha256": sha(OUT / "predictions.csv"), "manifests_sha256": sha(OUT / "manifests.json")}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
