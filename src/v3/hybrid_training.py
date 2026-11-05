"""Joint neural representation and ridge-solved affine experts, development only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.models import LSTMEncoder
from src.project_eval import Dataset, PREDICTION_COLUMNS, Standardizer, VALIDATION_FOLDS
from src.train import seed_everything
from src.export_contract import fuzzy_batch_tables
from src.v2.screen_distributions import EVENTS, announced_adjustments


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs/v3_hybrid_training_contract_2026-11-05.json"
DEFAULT_OUT = ROOT / "output/v3/hybrid_training"
HEAD_IDS = {"linear": "LSTM3", "gaussian": "Fuzzy2", "gate": "Gate2"}


def corrected_daily(daily: pd.DataFrame) -> np.ndarray:
    q = np.log((daily.srt_close + daily.srt_dividend_sgd_on_date) / daily.srt_close.shift(1))
    return np.column_stack((q, q.rolling(5, min_periods=5).sum(),
                            q.rolling(20, min_periods=20).std(ddof=0)))


def inputs_and_residual(data: Dataset, arm: str) -> tuple[np.ndarray, np.ndarray]:
    if arm not in ("raw", "gross"):
        raise ValueError("Unknown fixed input arm")
    x = data.x.copy()
    if arm == "gross":
        daily = corrected_daily(data.daily)
        x[:, :, :3] = np.stack([daily[int(row.daily_start_row):int(row.daily_end_row) + 1]
                                for row in data.index.itertuples()])
    # Only completed development labels are exposed to the runner.
    y = np.full(len(data.y), np.nan, dtype=np.float64)
    development = data.index.target_date.le(pd.Timestamp("2025-09-30"))
    rows = data.index.loc[development]
    close = rows.forecast_date.map(data.daily.set_index("date").srt_close).to_numpy()
    distribution = rows.target_date.map(data.daily.set_index("date").srt_dividend_sgd_on_date).to_numpy()
    y[rows.window_id] = data.y[rows.window_id].astype(float) - np.log1p(-distribution / close)
    if not np.isfinite(x[development]).all() or not np.isfinite(y[development]).all():
        raise ValueError("Nonfinite development inputs or residual labels")
    return x, y


def ridge_coefficients(phi: torch.Tensor, target: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty <= 0 or len(phi) != len(target) or not len(phi):
        raise ValueError("Positive ridge penalty and aligned nonempty training data required")
    # Float64 solve is cheap (4 or 8 columns) and remains differentiable.
    a, b = phi.double(), target.double()
    matrix = a.T @ a / len(a) + penalty * torch.eye(a.shape[1], dtype=a.dtype, device=a.device)
    return torch.linalg.solve(matrix, a.T @ b / len(a)).to(phi.dtype)


class HybridHead(nn.Module):
    def __init__(self, head: str):
        super().__init__()
        if head not in HEAD_IDS:
            raise ValueError("Unknown head")
        self.head = head
        self.encoder = LSTMEncoder(6)
        self.projection = nn.Linear(32, 3)
        self.register_buffer("latent_mean", torch.zeros(3))
        self.register_buffer("latent_std", torch.ones(3))
        self.centers = nn.Parameter(torch.zeros(2, 3)) if head == "gaussian" else None
        self.width_logits = nn.Parameter(torch.zeros(2, 3)) if head == "gaussian" else None
        self.gate = nn.Sequential(nn.Linear(3, 2), nn.Tanh(), nn.Linear(2, 2)) if head == "gate" else None

    def latent(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(self.projection(self.encoder(x)))
        return (z - self.latent_mean) / self.latent_std

    @torch.no_grad()
    def initialize_from_training(self, x: torch.Tensor) -> None:
        z = torch.tanh(self.projection(self.encoder(x)))
        self.latent_mean.copy_(z.mean(0))
        self.latent_std.copy_(z.std(0, unbiased=False).clamp_min(0.001))
        if self.head != "gaussian":
            return
        u = self.latent(x)
        _, _, vectors = torch.linalg.svd(u, full_matrices=False)
        score = u @ vectors[0]
        centers = torch.stack((u[score.argmin()], u[score.argmax()]))
        for _ in range(10):
            assignment = torch.cdist(u, centers).argmin(1)
            centers = torch.stack([u[assignment == k].mean(0) if (assignment == k).any()
                                   else centers[k] for k in range(2)])
        assignment = torch.cdist(u, centers).argmin(1)
        widths = torch.stack([u[assignment == k].std(0, unbiased=False).clamp_min(0.25)
                              if (assignment == k).sum() >= 2 else torch.ones(3)
                              for k in range(2)])
        self.centers.copy_(centers)
        self.width_logits.copy_(torch.log(torch.expm1(widths - 0.05)))

    def design(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        z = self.latent(x)
        affine = torch.cat((z, torch.ones_like(z[:, :1])), 1)
        if self.head == "linear":
            return affine, {"z": z}
        if self.head == "gaussian":
            widths = nn.functional.softplus(self.width_logits) + 0.05
            distance = (((z[:, None, :] - self.centers) / widths) ** 2).sum(-1)
            logits = -distance / 2
        else:
            logits = self.gate(z)
            distance = None
        alpha = logits.softmax(-1)
        phi = (alpha[:, :, None] * affine[:, None, :]).reshape(len(z), -1)
        return phi, {"z": z, "alpha": alpha, "distance2": distance,
                     "log_membership": logits, "affine": affine}


def partition(data: Dataset, fold: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if fold not in VALIDATION_FOLDS:
        raise ValueError("Only original development folds are allowed")
    train, evaluation = data.ids(fold, "train"), data.ids(fold, "validation")
    n = math.ceil(0.15 * len(train))
    fit, stop = train[:-n], train[-n:]
    for left, right in ((fit, stop), (train, evaluation)):
        # A completed close label at the first forecast close is available then.
        if data.index.target_date.iloc[left].max() > data.index.forecast_date.iloc[right].min():
            raise ValueError("Training label not available at evaluation forecast")
    return fit, stop, train, evaluation


def arrays(x: np.ndarray, y: np.ndarray, fit: np.ndarray, evaluation: np.ndarray):
    s = Standardizer.fit(x[fit], y[fit])
    scale = Standardizer(s.x_mean, s.x_std, 0.0, s.y_std)
    return (torch.from_numpy(scale.x_transform(x[fit])),
            torch.from_numpy(scale.y_transform(y[fit])),
            torch.from_numpy(scale.x_transform(x[evaluation])), scale)


def step(model: HybridHead, optimizer, x, y, penalty: float) -> float:
    optimizer.zero_grad(set_to_none=True)
    phi, _ = model.design(x)
    beta = ridge_coefficients(phi, y, penalty)
    loss = ((phi @ beta - y) ** 2).mean() + penalty * (beta ** 2).sum()
    loss.backward()
    if not torch.isfinite(loss) or not all(p.grad is not None and torch.isfinite(p.grad).all()
                                         for p in model.parameters()):
        raise FloatingPointError("Nonfinite or missing hybrid-training gradient")
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def predict_from_training(model, x, y, eval_x, penalty):
    phi, _ = model.design(x)
    beta = ridge_coefficients(phi, y, penalty)
    eval_phi, diagnostics = model.design(eval_x)
    return eval_phi @ beta, beta, diagnostics


def run_one(data, x, y, adjustment, arm, head, fold, seed, out, contract):
    fit, stop, train, evaluation = partition(data, fold)
    settings, penalty = contract["training"], contract["lambda"]
    xt, yt, xs, scale = arrays(x, y, fit, stop)
    ys = torch.from_numpy(scale.y_transform(y[stop]))
    seed_everything(seed)
    model = HybridHead(head)
    model.initialize_from_training(xt)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    best, chosen, stale = float((ys ** 2).mean()), -1, 0
    trace = []
    began = time.perf_counter()
    for epoch in range(settings["max_epochs"] + 1):
        objective = step(model, optimizer, xt, yt, penalty) if epoch else None
        prediction, _, _ = predict_from_training(model, xt, yt, xs, penalty)
        loss = float(((prediction - ys) ** 2).mean())
        if not np.isfinite(loss):
            raise FloatingPointError("Nonfinite stopping MSE")
        trace.append({"epoch": epoch, "train_penalized_objective": objective, "stop_mse": loss})
        if loss <= best - settings["min_improvement"]:
            best, chosen, stale = loss, epoch, 0
        else:
            stale += 1
        if stale >= settings["patience"]:
            break
    inner_seconds = time.perf_counter() - began
    xt, yt, xe, scale = arrays(x, y, train, evaluation)
    seed_everything(seed)
    model = HybridHead(head)
    model.initialize_from_training(xt)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    began = time.perf_counter()
    for _ in range(max(0, chosen)):
        step(model, optimizer, xt, yt, penalty)
    residual_prediction, beta, diagnostic = predict_from_training(model, xt, yt, xe, penalty)
    if chosen == -1:
        beta = torch.zeros_like(beta)
        residual_prediction = torch.zeros_like(residual_prediction)
    offset = adjustment.loc[evaluation].to_numpy(dtype=float)
    raw_prediction = scale.y_inverse(residual_prediction.numpy()) + offset
    model_id = HEAD_IDS[head]
    run_dir = out / arm / head / fold / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    prediction = pd.DataFrame([(model_id, "all", 2 if head == "gaussian" else 0, fold,
                                seed, int(i), float(p)) for i, p in zip(evaluation, raw_prediction)],
                              columns=PREDICTION_COLUMNS)
    data.validate_predictions(prediction)
    stats = {}
    if head != "linear":
        experts = diagnostic["affine"] @ beta.reshape(2, 4).T
        alpha = diagnostic["alpha"].numpy()
        stats = {"mean_max_weight": float(alpha.max(1).mean()),
                 "fraction_max_weight_above_08": float((alpha.max(1) > 0.8).mean()),
                 "residual_expert_disagreement": float(torch.abs(experts[:, 0] - experts[:, 1]).mean()) * scale.y_std}
        if head == "gaussian":
            diagnostic["expert_predictions"] = experts
            export = {k: diagnostic[k] for k in ("z", "alpha", "distance2", "log_membership", "expert_predictions")}
            _, cases, rules = fuzzy_batch_tables(model_id=model_id, source_group="all", fold=fold,
                seed=seed, window_ids=evaluation, prediction=residual_prediction, diagnostics=export, scaler=scale)
            mapping = dict(zip(evaluation, offset))
            cases["y_mix_raw"] += cases.window_id.map(mapping)
            rules["expert_raw"] += rules.window_id.map(mapping)
            data.validate_fuzzy_diagnostics(prediction, cases, rules)
            cases.to_csv(run_dir / "cases.csv", index=False)
            rules.to_csv(run_dir / "rules.csv", index=False)
    prediction.to_csv(run_dir / "predictions.csv", index=False)
    torch.save({"model_state": model.state_dict(), "head": head, "beta": beta,
                "scale": {"x_mean": scale.x_mean.tolist(), "x_std": scale.x_std.tolist(),
                          "y_mean": 0.0, "y_std": scale.y_std}}, run_dir / "checkpoint.pt")
    manifest = {"experiment_id": contract["experiment_id"], "input_arm": arm, "head": head,
                "fold": fold, "seed": seed, "chosen_epoch": chosen, "zero_fallback": chosen == -1,
                "inner_seconds": inner_seconds, "outer_seconds": time.perf_counter() - began,
                "train_rows": len(train), "eval_rows": len(evaluation), "trace": trace, **stats,
                "output_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in run_dir.glob("*.csv")}}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return prediction, manifest


def run(out: Path):
    if out.exists():
        raise FileExistsError("Refuse to overwrite a V3 run")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if (contract["lambda"] != 0.1 or contract["input_arms"] != ["raw", "gross"]
            or contract["head_arms"] != list(HEAD_IDS) or tuple(contract["folds"]) != VALIDATION_FOLDS
            or contract["seeds"] != [11, 23, 37]):
        raise ValueError("Unexpected fixed contract")
    torch.set_num_threads(1)
    data = Dataset(ROOT)
    offset = announced_adjustments(data, pd.read_csv(EVENTS)).set_index("window_id").adjustment
    dependencies = [CONTRACT, Path(__file__), ROOT / "src/models.py", ROOT / "src/project_eval.py",
                    ROOT / "src/train.py", ROOT / "src/export_contract.py", ROOT / "src/v2/screen_distributions.py",
                    EVENTS, *[ROOT / "data/processed" / n for n in
                    ("srt_daily_aligned.csv", "srt_windows_unscaled.npz", "window_index.csv", "fold_assignments.csv")]]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
    out.mkdir(parents=True, exist_ok=False)
    (out / "pre_execution.json").write_text(json.dumps({"contract": contract, "dependency_sha256": hashes,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__}}, indent=2), encoding="utf-8")
    all_predictions, manifests = [], []
    for arm in contract["input_arms"]:
        x, y = inputs_and_residual(data, arm)
        for head in contract["head_arms"]:
            for fold in VALIDATION_FOLDS:
                for seed in contract["seeds"]:
                    p, m = run_one(data, x, y, offset, arm, head, fold, seed, out, contract)
                    p["experiment_arm"] = f"{arm}/{head}"
                    all_predictions.append(p)
                    manifests.append(m)
                    print(json.dumps({k: m[k] for k in ("input_arm", "head", "fold", "seed", "chosen_epoch", "zero_fallback")}), flush=True)
    pd.concat(all_predictions, ignore_index=True).to_csv(out / "predictions.csv", index=False)
    (out / "manifests.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    current = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
    if current != hashes:
        raise RuntimeError("A source or contract changed during execution")
    (out / "complete.json").write_text(json.dumps({"runs": len(manifests), "status": "complete",
        "prediction_sha256": hashlib.sha256((out / "predictions.csv").read_bytes()).hexdigest()}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    run(parser.parse_args().out)
