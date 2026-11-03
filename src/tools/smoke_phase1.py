"""Short development-fold engineering check; not an experiment result."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.export_contract import fuzzy_batch_tables
from src.models import build_model, parameter_count
from src.project_eval import Dataset


def main() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(11)
    np.random.seed(11)
    data = Dataset()
    inner = data.prepare("validation_1", "all", inner_stop=True)
    x_train = torch.from_numpy(inner.x_train)
    y_train = torch.from_numpy(inner.y_train)
    x_stop = torch.from_numpy(inner.x_eval)
    y_stop = torch.from_numpy(inner.scaler.y_transform(inner.y_eval_raw))
    records = []
    fuzzy_model = None
    for model_id in ("LSTM32", "LSTM3", "Fuzzy2"):
        torch.manual_seed(11)
        model = build_model(model_id, input_size=6)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        start = time.perf_counter()
        losses = []
        for epoch in range(2):
            model.train()
            generator = torch.Generator().manual_seed(11 + epoch)
            for indices in torch.randperm(len(x_train), generator=generator).split(64):
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(model(x_train[indices]), y_train[indices])
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite {model_id} loss")
                loss.backward()
                if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise RuntimeError(f"Missing/nonfinite {model_id} gradient")
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                losses.append(float(nn.functional.mse_loss(model(x_stop), y_stop)))
        record = {"model_id": model_id, "parameters": parameter_count(model),
                  "inner_fit_rows": len(x_train), "inner_stop_rows": len(x_stop),
                  "epochs": 2, "stop_mse_scaled": losses,
                  "elapsed_seconds": round(time.perf_counter() - start, 3)}
        records.append(record)
        if model_id == "Fuzzy2":
            fuzzy_model = model
    # Four outer-validation IDs only demonstrate the export schema and algebra.
    outer = data.prepare("validation_1", "all")
    with torch.no_grad():
        prediction, diagnostics = fuzzy_model(torch.from_numpy(outer.x_eval[:4]), return_diagnostics=True)
    predictions, cases, rules = fuzzy_batch_tables(
        model_id="Fuzzy2", source_group="all", fold="validation_1", seed=11,
        window_ids=outer.eval_ids[:4], prediction=prediction, diagnostics=diagnostics,
        scaler=outer.scaler,
    )
    data.validate_fuzzy_diagnostics(predictions, cases, rules, complete=False)
    destination = Path("output/runs/tmp/phase1_smoke")
    destination.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(destination / "sample_predictions.csv", index=False)
    cases.to_csv(destination / "sample_cases.csv", index=False)
    rules.to_csv(destination / "sample_rules.csv", index=False)
    (destination / "smoke.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))
    print(f"Sample export validation passed: {destination}")


if __name__ == "__main__":
    main()
