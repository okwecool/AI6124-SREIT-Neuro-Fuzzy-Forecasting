"""Supplement V2 H3 analysis with matched prediction-shape and V1 mechanism checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.project_eval import Dataset


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/v2/analysis"


def main() -> None:
    target_shape = OUT / "prediction_shape_comparison.csv"
    target_mechanism = OUT / "mechanism_paired_vs_v1.csv"
    if target_shape.exists() or target_mechanism.exists():
        raise FileExistsError("V2 supplement already exists; do not overwrite")
    data = Dataset(ROOT)
    original = pd.read_csv(ROOT / "output/development/predictions.csv")
    original = original.loc[original.model_id.eq("Fuzzy2") & original.source_group.eq("all")]
    local = pd.read_csv(OUT / "h3_predictions.csv")
    frames = []
    for label, frame in (("V1 Fuzzy2", original), ("H3 Fuzzy2+local", local)):
        checked = data.validate_predictions(frame)
        shape = checked.groupby(["fold", "seed"], as_index=False).agg(
            n=("window_id", "size"), true_mean=("y_true", "mean"),
            true_sd=("y_true", "std"), pred_mean=("y_pred", "mean"),
            pred_sd=("y_pred", "std"),
            positive_predictions=("y_pred", lambda s: int((s > 0).sum())))
        shape.insert(0, "variant", label)
        shape["pred_to_true_sd_ratio"] = shape.pred_sd / shape.true_sd
        frames.append(shape)
    pd.concat(frames, ignore_index=True).to_csv(target_shape, index=False)

    old_mechanism = pd.read_csv(ROOT / "output/diagnostics_2026-11-04/expert_diagnostics.csv")
    old_mechanism = old_mechanism.loc[old_mechanism.fold.ne("final_holdout"),
                                      ["fold", "seed", "mean_max_alpha",
                                       "opposite_expert_sign_fraction", "mean_cancellation_fraction"]]
    new_mechanism = pd.read_csv(OUT / "mechanism_by_run.csv")
    joined = new_mechanism.merge(old_mechanism, on=["fold", "seed"],
                                  suffixes=("_local", "_v1"), validate="one_to_one")
    joined["delta_cancellation_local_minus_v1"] = (joined.mean_cancellation -
                                                     joined.mean_cancellation_fraction)
    joined["delta_max_alpha_local_minus_v1"] = (joined.mean_max_alpha_local -
                                                 joined.mean_max_alpha_v1)
    if len(joined) != 9 or not np.isfinite(joined.delta_cancellation_local_minus_v1).all():
        raise ValueError("Mechanism comparison is incomplete")
    joined.to_csv(target_mechanism, index=False)
    print({"status": "supplement_complete", "shape_rows": 18,
           "mechanism_pairs": len(joined)})


if __name__ == "__main__":
    main()
