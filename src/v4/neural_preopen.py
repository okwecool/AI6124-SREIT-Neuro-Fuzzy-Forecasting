"""Fixed 8-input neural/fuzzy comparison, with an explicitly declared ARX head."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.project_eval import Dataset, PREDICTION_COLUMNS
from src.export_contract import fuzzy_batch_tables
from src.train import seed_everything
from src.v2.screen_distributions import EVENTS, announced_adjustments
from src.v3.hybrid_training import HybridHead, arrays, partition, ridge_coefficients
from src.v4.preopen_data import build


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/v4/neural_preopen"
CONTRACT = ROOT / "docs/v4_neural_contract_2026-11-05.json"
MODEL_IDS = {"linear": "LSTM3", "gaussian": "Fuzzy2", "gate": "Gate2"}


class Encoder8(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(8, 32, batch_first=True)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (20, 8):
            raise ValueError("Expected20x8 historical input")
        _, (hidden, _) = self.lstm(x)
        return hidden[-1]


class TimedHead(HybridHead):
    def __init__(self, head, topology):
        # Initialize directly, avoiding an unused six-input encoder/RNG stream.
        nn.Module.__init__(self)
        if head not in MODEL_IDS or topology not in ("latent", "direct_us"):
            raise ValueError("Unregistered head/topology")
        self.head, self.topology = head, topology
        self.encoder, self.projection = Encoder8(), nn.Linear(32, 3)
        self.register_buffer("latent_mean", torch.zeros(3))
        self.register_buffer("latent_std", torch.ones(3))
        self.centers = nn.Parameter(torch.zeros(2, 3)) if head == "gaussian" else None
        self.width_logits = nn.Parameter(torch.zeros(2, 3)) if head == "gaussian" else None
        self.gate = nn.Sequential(nn.Linear(3, 2), nn.Tanh(), nn.Linear(2, 2)) if head == "gate" else None

    def design(self, x):
        phi, diagnostic = super().design(x)
        if self.topology == "latent":
            return phi, diagnostic
        affine = torch.cat((diagnostic["z"], x[:, -1, 6:], torch.ones_like(x[:, -1, :1])), 1)
        diagnostic["affine"] = affine
        if self.head == "linear":
            return affine, diagnostic
        return (diagnostic["alpha"][:, :, None] * affine[:, None, :]).reshape(len(x), -1), diagnostic


def construct(head, topology, seed, x):
    seed_everything(seed)
    model = TimedHead(head, topology)
    model.initialize_from_training(x)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    return model, optimizer


def train_step(model, optimizer, x, y):
    optimizer.zero_grad(set_to_none=True)
    phi, _ = model.design(x)
    beta = ridge_coefficients(phi, y, 0.1)
    loss = ((phi @ beta - y) ** 2).mean() + 0.1 * beta.square().sum()
    loss.backward()
    if not torch.isfinite(loss) or not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError("Invalid neural gradient/loss")
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def predict(model, x, y, evaluate):
    phi, _ = model.design(x)
    beta = ridge_coefficients(phi, y, 0.1)
    design, diag = model.design(evaluate)
    return design @ beta, beta, diag


def one(data, x, y, offset, clock, topology, head, fold, seed, contract):
    inner, stop, train, evaluate = partition(data, fold)
    xt, yt, xs, scale = arrays(x, y, inner, stop)
    ys = torch.from_numpy(scale.y_transform(y[stop]))
    model, optimizer = construct(head, topology, seed, xt)
    best, selected, stale = float(ys.square().mean()), -1, 0
    trace, start = [], time.perf_counter()
    for epoch in range(contract["training"]["max_epochs"] + 1):
        objective = train_step(model, optimizer, xt, yt) if epoch else None
        estimate, _, _ = predict(model, xt, yt, xs)
        loss = float((estimate - ys).square().mean())
        if not np.isfinite(loss):
            raise FloatingPointError("Nonfinite stopping loss")
        trace.append({"epoch": epoch, "training_objective": objective, "stop_mse": loss})
        if loss <= best - 1e-4:
            best, selected, stale = loss, epoch, 0
        else:
            stale += 1
        if stale >= 20:
            break
    inner_seconds = time.perf_counter() - start
    xt, yt, xe, scale = arrays(x, y, train, evaluate)
    model, optimizer = construct(head, topology, seed, xt)
    start = time.perf_counter()
    for _ in range(max(selected, 0)):
        train_step(model, optimizer, xt, yt)
    prediction, beta, diag = predict(model, xt, yt, xe)
    if selected < 0:
        prediction, beta = torch.zeros_like(prediction), torch.zeros_like(beta)
    extra = offset.loc[evaluate].to_numpy()
    raw = scale.y_inverse(prediction.numpy()) + extra
    model_id = MODEL_IDS[head]
    frame = pd.DataFrame([(model_id, "all", 2 if head == "gaussian" else 0, fold, seed, int(i), float(p))
                          for i, p in zip(evaluate, raw)], columns=PREDICTION_COLUMNS)
    data.validate_predictions(frame)
    path = OUT / clock / topology / head / fold / f"seed_{seed}"
    path.mkdir(parents=True, exist_ok=False)
    frame.to_csv(path / "predictions.csv", index=False)
    if head == "gaussian":
        diag["expert_predictions"] = diag["affine"] @ beta.reshape(2, -1).T
        exported = {k: diag[k] for k in ("z", "alpha", "expert_predictions", "log_membership", "distance2")}
        _, cases, rules = fuzzy_batch_tables(model_id=model_id, source_group="all", fold=fold, seed=seed,
            window_ids=evaluate, prediction=prediction, diagnostics=exported, scaler=scale)
        mapping = dict(zip(evaluate, extra))
        cases["y_mix_raw"] += cases.window_id.map(mapping)
        rules["expert_raw"] += rules.window_id.map(mapping)
        data.validate_fuzzy_diagnostics(frame, cases, rules)
        cases.to_csv(path / "cases.csv", index=False)
        rules.to_csv(path / "rules.csv", index=False)
    torch.save({"state": model.state_dict(), "beta": beta,
        "scaler": {"x_mean": scale.x_mean.tolist(), "x_std": scale.x_std.tolist(), "y_mean": 0.0, "y_std": scale.y_std}}, path / "checkpoint.pt")
    manifest = {"arm": f"{clock}/{topology}/{head}", "fold": fold, "seed": seed,
        "selected_epoch": selected, "zero_fallback": selected < 0, "trace": trace,
        "inner_seconds": inner_seconds, "outer_seconds": time.perf_counter() - start,
        "output_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.glob("*.csv")}}
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    frame["experiment_arm"] = manifest["arm"]
    return frame, manifest


def run():
    if OUT.exists():
        raise FileExistsError("Refuse to overwrite neural experiments")
    torch.set_num_threads(1)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    screen = json.loads((ROOT / "output/v4/preopen_screen/decision.json").read_text(encoding="utf-8"))
    if not any(row["advance"] for row in screen):
        raise ValueError("Information screen did not advance")
    if (contract["training"]["lambda"] != 0.1 or contract["heads"] != list(MODEL_IDS)
            or contract["clocks"] != ["close", "preopen"] or contract["topologies"] != ["latent", "direct_us"]
            or contract["seeds"] != [11, 23, 37]):
        raise ValueError("Unexpected neural matrix")
    data = Dataset(ROOT)
    inputs, y, alignment = build(data)
    offset = announced_adjustments(data, pd.read_csv(EVENTS)).set_index("window_id").adjustment
    screen_pre = json.loads((ROOT / "output/v4/preopen_screen/pre_execution.json").read_text(encoding="utf-8"))
    dependencies = [ROOT / p for p in screen_pre["dependency_sha256"]] + [CONTRACT, Path(__file__),
        ROOT / "src/models.py", ROOT / "src/train.py", ROOT / "src/export_contract.py", ROOT / "output/v4/preopen_screen/decision.json"]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
    for path, digest in screen_pre["dependency_sha256"].items():
        if hashes[path] != digest:
            raise ValueError("Information-screen dependency changed")
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "pre_execution.json").write_text(json.dumps({"contract": contract, "dependency_sha256": hashes}, indent=2), encoding="utf-8")
    all_rows, manifests = [], []
    for clock in contract["clocks"]:
        for topology in contract["topologies"]:
            for head in contract["heads"]:
                for fold in contract["folds"]:
                    for seed in contract["seeds"]:
                        frame, manifest = one(data, inputs[clock], y, offset, clock, topology, head, fold, seed, contract)
                        all_rows.append(frame)
                        manifests.append(manifest)
                print(json.dumps({"completed": f"{clock}/{topology}/{head}", "runs_so_far": len(manifests)}), flush=True)
    pd.concat(all_rows, ignore_index=True).to_csv(OUT / "predictions.csv", index=False)
    (OUT / "manifests.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    if hashes != {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}:
        raise RuntimeError("Execution dependencies changed")
    (OUT / "complete.json").write_text(json.dumps({"runs": len(manifests),
        "prediction_sha256": hashlib.sha256((OUT / "predictions.csv").read_bytes()).hexdigest()}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
