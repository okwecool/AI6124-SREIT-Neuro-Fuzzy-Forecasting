"""Recompute saved H1 predictions after the Z0-only summary correction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.project_eval import Dataset, VALIDATION_FOLDS
from src.v2.baselines import CONTRACT, OUT, ROOT, digest, fit_ridge, predict_ridge


def main() -> None:
    target = OUT / "reproduction_check.json"
    if target.exists():
        raise FileExistsError(f"Reproduction check already exists: {target}")
    data = Dataset(ROOT)
    saved = pd.read_csv(OUT / "predictions.csv")
    data.validate_predictions(saved)
    manifests = json.loads((OUT / "manifests.json").read_text(encoding="utf-8"))
    max_gap = 0.0
    for fold in VALIDATION_FOLDS:
        manifest = next(item for item in manifests if item["fold"] == fold)
        outer = data.prepare(fold, "all", inner_stop=False)
        fitted = fit_ridge(outer.x_train, outer.y_train, manifest["selected_alpha"])
        recreated = {
            "Ridge": outer.scaler.y_inverse(predict_ridge(outer.x_eval, fitted)),
            "MeanTrain": np.full(len(outer.eval_ids), outer.scaler.y_mean),
        }
        for model, values in recreated.items():
            old = saved.loc[saved.fold.eq(fold) & saved.model_id.eq(model)].sort_values("window_id")
            if not np.array_equal(old.window_id.to_numpy(), outer.eval_ids):
                raise ValueError(f"H1 IDs changed for {fold}/{model}")
            max_gap = max(max_gap, float(np.max(np.abs(old.y_pred.to_numpy() - values))))
    if max_gap > 1e-12:
        raise AssertionError(f"H1 predictions changed after analysis correction: {max_gap}")
    result = {"status": "H1_saved_predictions_reproduced", "folds": 3,
              "verification_only_ridge_refits": 3,
              "maximum_raw_prediction_difference": max_gap,
              "saved_prediction_sha256": digest(OUT / "predictions.csv"),
              "contract_sha256": digest(CONTRACT),
              "note": "The fit logic is unchanged; the source-file hash differs because the Z0 summary was repaired before H3."}
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
