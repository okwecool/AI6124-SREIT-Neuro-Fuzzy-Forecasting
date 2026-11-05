"""Earlier-CLR pretraining and a strictly matched frozen transfer probe."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from data.prepare_dataset import yahoo_frame
from src.models import LSTM3
from src.project_eval import Dataset, Standardizer, PREDICTION_COLUMNS, VALIDATION_FOLDS
from src.train import seed_everything, train_epoch, evaluation_loss
from src.export_contract import fuzzy_batch_tables
from src.v2.screen_distributions import EVENTS, announced_adjustments
from src.v3.hybrid_training import HybridHead, arrays, inputs_and_residual, partition, predict_from_training, HEAD_IDS


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/v3/transfer_probe"
CONTRACT = ROOT / "docs/v3_transfer_contract_2026-11-05.json"
SOURCE = ROOT / "data/raw/yahoo_clr_si_20171001_20260930.json"


def source_windows():
    daily, events, _ = yahoo_frame(SOURCE, "CLR.SI")
    daily = daily.loc[daily.date.le("2021-09-30")].copy().reset_index(drop=True)
    if daily.date.duplicated().any() or not daily.date.is_monotonic_increasing or daily.close.isna().any():
        raise ValueError("Invalid CLR source calendar")
    dividend = daily.date.map(events.set_index("date").cash_dividend_sgd).fillna(0)
    q = np.log((daily.close + dividend) / daily.close.shift(1))
    features = np.column_stack((q, q.rolling(5).sum(), q.rolling(20).std(ddof=0)))
    x, y, dates = [], [], []
    for end in range(19, len(daily) - 1):
        sample = features[end - 19:end + 1]
        if np.isfinite(sample).all() and np.isfinite(q.iloc[end + 1]):
            x.append(sample)
            y.append(q.iloc[end + 1])
            dates.append(str(daily.date.iloc[end + 1].date()))
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float64), dates


def source_model(seed):
    seed_everything(seed)
    common = HybridHead("linear")
    model = LSTM3(3)
    with torch.no_grad():
        model.encoder.lstm.weight_ih_l0.copy_(common.encoder.lstm.weight_ih_l0[:, :3])
        for name in ("weight_hh_l0", "bias_ih_l0", "bias_hh_l0"):
            getattr(model.encoder.lstm, name).copy_(getattr(common.encoder.lstm, name))
        model.projection.load_state_dict(common.projection.state_dict())
    return model


def pretrain(x, y, seed):
    count = math.ceil(0.15 * len(y))
    split = len(y) - count
    scale = Standardizer.fit(x[:split], y[:split])
    xt, yt = torch.from_numpy(scale.x_transform(x[:split])), torch.from_numpy(scale.y_transform(y[:split]))
    xs, ys = torch.from_numpy(scale.x_transform(x[split:])), torch.from_numpy(scale.y_transform(y[split:]))
    model = source_model(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    best, epoch_best, stale = float("inf"), 0, 0
    trace = []
    for epoch in range(1, 151):
        loss = train_epoch(model, optimizer, xt, yt, seed=seed, epoch=epoch)
        stop = evaluation_loss(model, xs, ys)
        trace.append({"epoch": epoch, "train_loss": loss, "stop_loss": stop})
        if stop <= best - 1e-4:
            best, epoch_best, stale = stop, epoch, 0
        else:
            stale += 1
        if stale >= 20:
            break
    scale = Standardizer.fit(x, y)
    xt, yt = torch.from_numpy(scale.x_transform(x)), torch.from_numpy(scale.y_transform(y))
    model = source_model(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    for epoch in range(1, epoch_best + 1):
        train_epoch(model, optimizer, xt, yt, seed=seed, epoch=epoch)
    path = OUT / "source" / f"seed_{seed}"
    path.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), path / "checkpoint.pt")
    (path / "manifest.json").write_text(json.dumps({"seed": seed, "selected_epoch": epoch_best,
        "training_rows": len(y), "inner_rows": split, "stop_rows": count, "trace": trace,
        "source_scaler": {"x_mean": scale.x_mean.tolist(), "x_std": scale.x_std.tolist(),
                          "y_mean": scale.y_mean, "y_std": scale.y_std}}, indent=2), encoding="utf-8")
    return model


def initialized(head, seed, source, training_x):
    seed_everything(seed)
    model = HybridHead(head)
    if source is not None:
        with torch.no_grad():
            model.encoder.lstm.weight_ih_l0[:, :3].copy_(source.encoder.lstm.weight_ih_l0)
            for name in ("weight_hh_l0", "bias_ih_l0", "bias_hh_l0"):
                getattr(model.encoder.lstm, name).copy_(getattr(source.encoder.lstm, name))
            model.projection.load_state_dict(source.projection.state_dict())
    model.initialize_from_training(training_x)
    model.requires_grad_(False)
    return model


def run():
    if OUT.exists():
        raise FileExistsError("Do not overwrite transfer results")
    torch.set_num_threads(1)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract["seeds"] != [11, 23, 37] or tuple(contract["folds"]) != VALIDATION_FOLDS:
        raise ValueError("Unexpected transfer contract")
    data = Dataset(ROOT)
    sx, sy, source_dates = source_windows()
    if pd.Timestamp(max(source_dates)) >= data.daily.date.min():
        raise ValueError("Source history overlaps the target experiment")
    dependencies = [CONTRACT, Path(__file__), SOURCE, EVENTS, ROOT / "data/prepare_dataset.py",
        ROOT / "src/v3/hybrid_training.py", ROOT / "src/models.py", ROOT / "src/project_eval.py",
        ROOT / "src/train.py", ROOT / "src/export_contract.py", ROOT / "src/v2/screen_distributions.py",
        *[ROOT / "data/processed" / p for p in ("srt_daily_aligned.csv", "srt_windows_unscaled.npz", "window_index.csv", "fold_assignments.csv")]]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "pre_execution.json").write_text(json.dumps({"contract": contract, "dependency_sha256": hashes,
        "source_windows": len(sy), "first_source_target": min(source_dates), "last_source_target": max(source_dates)}, indent=2), encoding="utf-8")
    pretrained = {seed: pretrain(sx, sy, seed) for seed in contract["seeds"]}
    x, y = inputs_and_residual(data, "gross")
    offset = announced_adjustments(data, pd.read_csv(EVENTS)).set_index("window_id").adjustment
    rows, manifests = [], []
    for transfer in ("random", "transfer"):
        for head in HEAD_IDS:
            for fold in VALIDATION_FOLDS:
                inner, stop, train, evaluate = partition(data, fold)
                for seed in contract["seeds"]:
                    source = pretrained[seed] if transfer == "transfer" else None
                    xt, yt, xs, scale = arrays(x, y, inner, stop)
                    model = initialized(head, seed, source, xt)
                    prediction, _, _ = predict_from_training(model, xt, yt, xs, 0.1)
                    ys = torch.from_numpy(scale.y_transform(y[stop]))
                    fallback = float(((prediction - ys) ** 2).mean()) > float((ys ** 2).mean()) - 1e-4
                    xt, yt, xe, scale = arrays(x, y, train, evaluate)
                    model = initialized(head, seed, source, xt)
                    prediction, beta, diag = predict_from_training(model, xt, yt, xe, 0.1)
                    if fallback:
                        beta, prediction = torch.zeros_like(beta), torch.zeros_like(prediction)
                    added = offset.loc[evaluate].to_numpy()
                    raw = scale.y_inverse(prediction.numpy()) + added
                    model_id = HEAD_IDS[head]
                    frame = pd.DataFrame([(model_id, "all", 2 if head == "gaussian" else 0, fold, seed, int(i), float(v))
                                          for i, v in zip(evaluate, raw)], columns=PREDICTION_COLUMNS)
                    data.validate_predictions(frame)
                    path = OUT / transfer / head / fold / f"seed_{seed}"
                    path.mkdir(parents=True, exist_ok=False)
                    frame.to_csv(path / "predictions.csv", index=False)
                    if head == "gaussian":
                        diag["expert_predictions"] = diag["affine"] @ beta.reshape(2, 4).T
                        export = {k: diag[k] for k in ("z", "alpha", "distance2", "log_membership", "expert_predictions")}
                        _, cases, rules = fuzzy_batch_tables(model_id=model_id, source_group="all", fold=fold,
                            seed=seed, window_ids=evaluate, prediction=prediction, diagnostics=export, scaler=scale)
                        mapping = dict(zip(evaluate, added))
                        cases["y_mix_raw"] += cases.window_id.map(mapping)
                        rules["expert_raw"] += rules.window_id.map(mapping)
                        data.validate_fuzzy_diagnostics(frame, cases, rules)
                        cases.to_csv(path / "cases.csv", index=False)
                        rules.to_csv(path / "rules.csv", index=False)
                    torch.save({"state": model.state_dict(), "beta": beta,
                        "scale": {"x_mean": scale.x_mean.tolist(), "x_std": scale.x_std.tolist(), "y_std": scale.y_std}}, path / "checkpoint.pt")
                    manifest = {"arm": f"{transfer}/{head}", "fold": fold, "seed": seed, "zero_fallback": fallback,
                        "output_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.glob("*.csv")}}
                    (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                    frame["experiment_arm"] = manifest["arm"]
                    rows.append(frame)
                    manifests.append(manifest)
    pd.concat(rows, ignore_index=True).to_csv(OUT / "predictions.csv", index=False)
    (OUT / "manifests.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    if hashes != {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}:
        raise RuntimeError("Execution dependency changed")
    (OUT / "complete.json").write_text(json.dumps({"status": "complete", "runs": len(manifests),
        "prediction_sha256": hashlib.sha256((OUT / "predictions.csv").read_bytes()).hexdigest()}, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "source_windows": len(sy), "runs": len(manifests)}))


if __name__ == "__main__":
    run()
