"""Independent loss recomputation and exploratory within-fold block intervals."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.project_eval import Dataset, VALIDATION_FOLDS
from src.analysis_utils import circular_block_interval, sha256_file
from src.v4.preopen_data import asof_values, us_frame

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/v4/neural_preopen"


def run():
    data = Dataset(ROOT)
    error = pd.read_csv(OUT / "paired_errors.csv")
    expected = pd.read_csv(OUT / "comparison_pooled.csv").set_index("arm")
    max_difference = 0.0
    for arm, part in error.groupby("arm"):
        rows = []
        for _, seed in part.groupby("seed"):
            if len(seed) != 378 or seed.window_id.nunique() != 378:
                raise ValueError("Expected 378 unique development dates per seed")
            np.testing.assert_allclose(
                seed.y_true.to_numpy(), data.y[seed.window_id.to_numpy()], atol=1e-12, rtol=0
            )
            e = seed.y_pred.to_numpy() - seed.y_true.to_numpy()
            rows.append((np.sqrt(np.mean(e**2)), np.mean(np.abs(e))))
        actual = np.mean(rows, axis=0)
        saved = expected.loc[arm, ["return_rmse", "return_mae"]].to_numpy(dtype=float)
        max_difference = max(max_difference, float(np.max(np.abs(actual - saved))))
    if max_difference > 1e-12:
        raise ValueError("Independent score recomputation disagrees")
    # Same dates are resampled for each comparison. Average losses, never
    # concatenate three seeds as 1,134 independent market dates.
    error["loss"] = (error.y_pred - error.y_true) ** 2
    losses = error.groupby(["arm", "fold", "window_id"]).loss.mean()
    pairs = [
        ("preopen_us", "DivSchedule"),
        ("preopen/latent/gaussian", "DivSchedule"),
        ("preopen/direct_us/gaussian", "DivSchedule"),
        ("preopen/direct_us/gaussian", "preopen_us"),
    ]
    records = []
    for candidate, reference in pairs:
        chunks = [
            (losses.loc[(candidate, fold)] - losses.loc[(reference, fold)]).sort_index().to_numpy()
            for fold in VALIDATION_FOLDS
        ]
        for length in (5, 10, 20):
            mean, lower, upper = circular_block_interval(chunks, length)
            records.append(
                {
                    "candidate": candidate,
                    "reference": reference,
                    "block_length": length,
                    "mean_mse_difference": mean,
                    "lower95": float(lower),
                    "upper95": float(upper),
                    "crosses_zero": bool(lower <= 0 <= upper),
                }
            )
    intervals = pd.DataFrame(records)
    intervals.to_csv(OUT / "exploratory_block_intervals.csv", index=False)

    alignment = pd.read_csv(ROOT / "output/v4/preopen_screen/time_alignment.csv")
    timing = []
    for (clock, symbol), g in alignment.groupby(["clock", "symbol"]):
        source = us_frame(symbol)
        extra_hour = source.copy()
        extra_hour["assumed_available_at"] += pd.Timedelta(hours=1)
        checked = asof_values(pd.to_datetime(g.cutoff, utc=True), extra_hour)
        same = np.array_equal(
            checked.session_date.to_numpy(), pd.to_datetime(g.session_date).to_numpy()
        )
        if not same:
            raise ValueError("Extra one-hour source latency changes the fixed inputs")
        timing.append(
            {
                "clock": clock,
                "symbol": symbol,
                "same_sessions_with_three_hour_latency": same,
                "rows": len(g),
                "max_stale_hours": float(g.stale_hours.max()),
            }
        )
    record = {
        "status": "pass",
        "prediction_records_checked": len(error),
        "max_recomputed_metric_difference": max_difference,
        "latency_sensitivity": timing,
        "bootstrap": "post-hoc circular blocks within each development fold, lengths5/10/20,5000 resamples,seed20261105; descriptive and not corrected for prior model/hypothesis search",
        "trading": "post-hoc application diagnostic of existing fixed theta0 long/cash policy, price costs0/10/20bps and gross10bps; no policy or threshold selected using these results; actual timestamps are Singapore local time",
        "analysis_source_sha256": sha256_file(__file__),
        "analysis_utils_sha256": sha256_file(ROOT / "src/analysis_utils.py"),
    }
    (OUT / "independent_audit.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    print(intervals.loc[intervals.block_length.eq(10)].to_string(index=False))


if __name__ == "__main__":
    run()
