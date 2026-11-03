"""Export refit-training fuzzy activations for train-defined diagnostic cuts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.export_contract import fuzzy_batch_tables
from src.models import build_model
from src.project_eval import Dataset, FORECAST_KEY


def export_all(run_root: Path) -> dict:
    data = Dataset()
    completed = []
    for manifest_path in sorted(Path(run_root).rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["status"] != "development" or not manifest["model_id"].startswith("Fuzzy"):
            continue
        run_dir = manifest_path.parent
        outer = data.prepare(manifest["fold"], manifest["source_group"])
        checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
        if not np.allclose(outer.scaler.x_mean, checkpoint["scaler"]["x_mean"]) or not np.isclose(
            outer.scaler.y_std, checkpoint["scaler"]["y_std"]
        ):
            raise ValueError(f"Saved scaler differs from fold scaler: {run_dir}")
        model = build_model(manifest["model_id"], outer.x_train.shape[-1])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        predictions, cases, rules = [], [], []
        with torch.no_grad():
            for start in range(0, len(outer.train_ids), 64):
                stop = start + 64
                values, diagnostics = model(torch.from_numpy(outer.x_train[start:stop]),
                                            return_diagnostics=True)
                pred, case, rule = fuzzy_batch_tables(
                    model_id=manifest["model_id"], source_group=manifest["source_group"],
                    fold=manifest["fold"], seed=manifest["seed"],
                    window_ids=outer.train_ids[start:stop], prediction=values,
                    diagnostics=diagnostics, scaler=outer.scaler)
                predictions.append(pred)
                cases.append(case)
                rules.append(rule)
        pred = pd.concat(predictions, ignore_index=True)
        case = pd.concat(cases, ignore_index=True)
        rule = pd.concat(rules, ignore_index=True)
        if len(pred) != len(outer.train_ids) or len(rule) != len(pred) * manifest["rule_count"]:
            raise ValueError(f"Incomplete training diagnostics: {run_dir}")
        weighted = rule.assign(contribution=rule.alpha * rule.expert_raw).groupby(
            FORECAST_KEY, as_index=False).contribution.sum()
        check = pred.merge(case, on=FORECAST_KEY, validate="one_to_one").merge(
            weighted, on=FORECAST_KEY, validate="one_to_one")
        if not (np.allclose(check.y_pred, check.y_mix_raw, atol=1e-6, rtol=0) and
                np.allclose(check.y_pred, check.contribution, atol=1e-6, rtol=0)):
            raise ValueError(f"Training diagnostic mixture mismatch: {run_dir}")
        pred.to_csv(run_dir / "training_predictions.csv", index=False)
        case.to_csv(run_dir / "training_cases.csv", index=False)
        rule.to_csv(run_dir / "training_rules.csv", index=False)
        completed.append({"run": str(run_dir), "training_cases": len(case),
                          "training_rule_rows": len(rule)})
    return {"fuzzy_runs": len(completed), "training_cases": sum(x["training_cases"] for x in completed),
            "training_rule_rows": sum(x["training_rule_rows"] for x in completed)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_all(args.runs), indent=2))


if __name__ == "__main__":
    main()
