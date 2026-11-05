"""Labelled post-hoc check of V9 forecast bias and distribution-day influence."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.project_eval import Dataset, VALIDATION_FOLDS


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/v9/signal_audit"


def run() -> pd.DataFrame:
    target = OUT / "posthoc_event_sensitivity.csv"
    if target.exists():
        raise FileExistsError(target)
    data = Dataset(ROOT)
    splits = pd.read_csv(OUT / "feature_median_splits.csv")
    predictions = pd.read_csv(ROOT / "output/development/predictions.csv")
    predictions = predictions.loc[predictions.model_id.eq("Fuzzy2") & predictions.source_group.eq("all")]
    distributions = data.daily.set_index("date").srt_dividend_sgd_on_date
    rows = []
    for fold in VALIDATION_FOLDS:
        cut = float(splits.loc[splits.fold.eq(fold) & splits.feature.eq("srt_volatility_20d"),
                               "train_median_threshold"].iloc[0])
        for seed, group in predictions.loc[predictions.fold.eq(fold)].groupby("seed"):
            ids = group.window_id.to_numpy(dtype=int)
            high = data.x[ids, -1, 2] >= cut
            dates = data.index.loc[ids, "target_date"]
            event = dates.map(distributions).fillna(0).to_numpy() > 0
            y = data.y[ids]
            pred = group.y_pred.to_numpy(dtype=float)
            for state, mask in (("low", ~high), ("high", high)):
                remaining = mask & ~event
                rows.append({"fold": fold, "seed": int(seed), "state": state,
                    "n": int(mask.sum()), "event_days": int((mask & event).sum()),
                    "actual_mean": float(y[mask].mean()),
                    "prediction_mean": float(pred[mask].mean()),
                    "prediction_sd": float(pred[mask].std(ddof=0)),
                    "mse_difference_vs_z0_all": float(np.mean((pred[mask] - y[mask]) ** 2 - y[mask] ** 2)),
                    "mse_difference_vs_z0_no_events": float(np.mean(
                        (pred[remaining] - y[remaining]) ** 2 - y[remaining] ** 2))})
    result = pd.DataFrame(rows)
    result.to_csv(target, index=False)
    return result


if __name__ == "__main__":
    print(run().groupby(["fold", "state"]).mean(numeric_only=True).to_string())
