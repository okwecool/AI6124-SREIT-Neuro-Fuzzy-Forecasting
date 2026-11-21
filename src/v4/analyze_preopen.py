"""Audit H16 outputs, attribute forecast gains, and report fixed-policy diagnostics."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.project_eval import Dataset, PREDICTION_COLUMNS
from src.analysis_utils import sha256_file, verify_hashes

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/v4/neural_preopen"


def run():
    data = Dataset(ROOT)
    pre = json.loads((OUT / "pre_execution.json").read_text(encoding="utf-8"))
    verify_hashes(ROOT, pre["dependency_sha256"])
    complete = json.loads((OUT / "complete.json").read_text(encoding="utf-8"))
    if (
        complete["runs"] != 108
        or sha256_file(OUT / "predictions.csv") != complete["prediction_sha256"]
    ):
        raise ValueError("Incomplete or changed experiment")
    manifests = pd.DataFrame(json.loads((OUT / "manifests.json").read_text(encoding="utf-8")))
    for m in manifests.itertuples():
        path = OUT / m.arm / m.fold / f"seed_{m.seed}"
        verify_hashes(path, m.output_sha256)
        if m.arm.endswith("/gaussian"):
            data.validate_fuzzy_diagnostics(
                *[
                    pd.read_csv(path / name)
                    for name in ("predictions.csv", "cases.csv", "rules.csv")
                ]
            )
    neural = pd.read_csv(OUT / "predictions.csv")
    screen = pd.read_csv(ROOT / "output/v4/preopen_screen/predictions.csv")
    screen["experiment_arm"] = screen.model_id
    prediction = pd.concat((neural, screen), ignore_index=True)
    events = set(pd.read_csv(ROOT / "data/reference/srt_announced_distributions_dev.csv").ex_date)
    seeds, folds, event_scores, errors = [], [], [], []
    for arm, g in prediction.groupby("experiment_arm"):
        s, f = data.pooled_validation_scores(g), data.forecast_scores(g)
        s["arm"], f["arm"] = arm, arm
        seeds.append(s[["arm", "seed", "return_rmse", "return_mae"]])
        folds.append(f[["arm", "fold", "seed", "return_rmse", "return_mae"]])
        frame = data.validate_predictions(g)
        frame["arm"] = arm
        frame["event"] = frame.target_date.dt.strftime("%Y-%m-%d").isin(events)
        errors.append(
            frame[["arm", "fold", "seed", "window_id", "target_date", "event", "y_true", "y_pred"]]
        )
        for (seed, event), part in frame.groupby(["seed", "event"]):
            e = part.y_pred - part.y_true
            event_scores.append(
                {
                    "arm": arm,
                    "seed": int(seed),
                    "event": bool(event),
                    "n": len(part),
                    "rmse": float(np.sqrt(np.mean(e**2))),
                    "mae": float(np.mean(np.abs(e))),
                }
            )
    seed_scores, fold_scores, event_scores = (
        pd.concat(seeds),
        pd.concat(folds),
        pd.DataFrame(event_scores),
    )
    all_errors = pd.concat(errors, ignore_index=True)
    pooled = seed_scores.groupby("arm")[["return_rmse", "return_mae"]].mean()
    pooled["gain_vs_schedule_pct"] = 100 * (
        1 - pooled.return_rmse / pooled.loc["DivSchedule", "return_rmse"]
    )
    fold_means = fold_scores.groupby(["arm", "fold"])[["return_rmse", "return_mae"]].mean()
    ne = event_scores.loc[~event_scores.event].groupby("arm").rmse.mean()
    decisions = []
    for candidate in ("preopen/latent/gaussian", "preopen/direct_us/gaussian"):
        reference_names = [
            "DivSchedule",
            candidate.replace("preopen", "close"),
            candidate.replace("gaussian", "linear"),
            candidate.replace("gaussian", "gate"),
            "preopen_us",
            "preopen_all",
        ]
        comparisons = []
        for name in reference_names:
            wins = int(
                (
                    fold_means.loc[candidate].return_rmse < fold_means.loc[name].return_rmse - 1e-12
                ).sum()
            )
            paired = None
            if "/" in name:
                a = fold_scores.loc[fold_scores.arm.eq(candidate)].set_index(["fold", "seed"])
                b = fold_scores.loc[fold_scores.arm.eq(name)].set_index(["fold", "seed"])
                paired = int((a.return_rmse < b.return_rmse - 1e-12).sum())
            comparisons.append(
                {
                    "reference": name,
                    "rmse_gain_pct": float(
                        100
                        * (
                            1
                            - pooled.loc[candidate, "return_rmse"] / pooled.loc[name, "return_rmse"]
                        )
                    ),
                    "mae_change_pct": float(
                        100
                        * (pooled.loc[candidate, "return_mae"] / pooled.loc[name, "return_mae"] - 1)
                    ),
                    "fold_wins": wins,
                    "paired_wins": paired,
                }
            )
        ne_gain = float(100 * (1 - ne[candidate] / ne["DivSchedule"]))
        pipeline = (
            comparisons[0]["rmse_gain_pct"] >= 3
            and comparisons[1]["rmse_gain_pct"] >= 3
            and comparisons[0]["fold_wins"] >= 2
            and comparisons[0]["mae_change_pct"] <= 1
            and ne_gain >= 0.5
        )
        fuzzy = all(
            c["rmse_gain_pct"] >= 1
            and c["fold_wins"] >= 2
            and c["mae_change_pct"] <= 1
            and (c["paired_wins"] is None or c["paired_wins"] >= 6)
            for c in comparisons[2:]
        )
        decisions.append(
            {
                "candidate": candidate,
                "comparisons": comparisons,
                "non_event_gain_pct": ne_gain,
                "pipeline_goal_passed": bool(pipeline),
                "fuzzy_incremental_goal_passed": bool(fuzzy),
            }
        )
    for filename, frame in (
        ("comparison_pooled.csv", pooled.reset_index()),
        ("comparison_by_seed.csv", seed_scores),
        ("comparison_by_fold.csv", fold_scores),
        ("event_scores.csv", event_scores),
        ("paired_errors.csv", all_errors),
    ):
        frame.to_csv(OUT / filename, index=False)
    (OUT / "decision.json").write_text(json.dumps(decisions, indent=2), encoding="utf-8")
    print(pooled.sort_values("return_rmse").to_string())
    print(json.dumps(decisions, indent=2))

    # Descriptive application check only: original untuned theta=0 and costs.
    # Keep original ledger frozen; add the true signal timestamps and verify
    # that the preopen08:00 forecast precedes the nominal09:00 opening.
    wanted = [
        "preopen/direct_us/gaussian",
        "preopen/direct_us/linear",
        "preopen/direct_us/gate",
        "preopen/latent/gaussian",
        "preopen/latent/linear",
        "preopen/latent/gate",
        "close/direct_us/gaussian",
        "preopen_us",
        "DivSchedule",
    ]
    summaries = []
    for arm in wanted:
        g = prediction.loc[prediction.experiment_arm.eq(arm), PREDICTION_COLUMNS]
        timing = (
            data.validate_predictions(g)[["target_date", "forecast_date"]]
            .drop_duplicates()
            .set_index("target_date")
        )
        timing["actual_signal_time"] = (
            timing.index + pd.Timedelta(hours=8)
            if arm.startswith("preopen")
            else timing.forecast_date + pd.Timedelta(hours=18)
        )
        if not (timing.actual_signal_time < timing.index + pd.Timedelta(hours=9)).all():
            raise ValueError("Forecast arrives too late to trade the open")
        for mode, cost in (
            ("gross_distribution", 10),
            ("price_only", 0),
            ("price_only", 10),
            ("price_only", 20),
        ):
            ledger, score = data.portfolio_ledger(g, wealth_mode=mode, cost_bps=cost, threshold=0)
            score["arm"] = arm
            summaries.append(score)
            if mode == "gross_distribution" and arm == "preopen/direct_us/gaussian":
                ledger["feature_end_date"] = ledger.signal_time
                ledger["signal_time"] = ledger.date.map(timing.actual_signal_time)
                ledger.to_csv(OUT / "primary_gross_10bps_ledger.csv", index=False)
    scores = pd.concat(summaries, ignore_index=True)
    scores.to_csv(OUT / "portfolio_by_fold.csv", index=False)
    product = (
        scores.groupby(["arm", "strategy", "wealth_mode", "cost_bps_per_side", "seed"])
        .terminal_wealth_multiple.prod()
        .reset_index(name="three_fold_compounded_wealth")
    )
    product.to_csv(OUT / "portfolio_by_seed.csv", index=False)
    mean = (
        product.groupby(["arm", "strategy", "wealth_mode", "cost_bps_per_side"])
        .three_fold_compounded_wealth.mean()
        .reset_index()
    )
    mean.to_csv(OUT / "portfolio_summary.csv", index=False)
    print(
        mean.loc[
            mean.wealth_mode.eq("gross_distribution") & mean.strategy.eq("signal_long_cash")
        ].to_string(index=False)
    )


if __name__ == "__main__":
    run()
