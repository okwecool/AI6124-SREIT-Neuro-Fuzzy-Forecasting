"""Verify and compare all fixed H12 arms, without fitting or viewing final scores."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.project_eval import Dataset, PREDICTION_COLUMNS
from src.analysis_utils import sha256_file, verify_hashes

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/v3/hybrid_training"


def analyze():
    pre = json.loads((OUT / "pre_execution.json").read_text(encoding="utf-8"))
    verify_hashes(ROOT, pre["dependency_sha256"])
    complete = json.loads((OUT / "complete.json").read_text(encoding="utf-8"))
    if (
        complete["runs"] != 54
        or sha256_file(OUT / "predictions.csv") != complete["prediction_sha256"]
    ):
        raise ValueError("Incomplete or altered experiment")
    data = Dataset(ROOT)
    predictions = pd.read_csv(OUT / "predictions.csv")
    manifests = json.loads((OUT / "manifests.json").read_text(encoding="utf-8"))
    if len(manifests) != 54:
        raise ValueError("Expected 54 manifests")
    fuzzy_runs = 0
    for manifest in manifests:
        path = (
            OUT
            / manifest["input_arm"]
            / manifest["head"]
            / manifest["fold"]
            / f"seed_{manifest['seed']}"
        )
        verify_hashes(path, manifest["output_sha256"])
        if manifest["head"] == "gaussian":
            data.validate_fuzzy_diagnostics(
                *[
                    pd.read_csv(path / name)
                    for name in ("predictions.csv", "cases.csv", "rules.csv")
                ]
            )
            fuzzy_runs += 1
    tables = []
    for arm, g in predictions.groupby("experiment_arm"):
        checked = data.validate_predictions(g.loc[:, PREDICTION_COLUMNS])
        checked["arm"] = arm
        tables.append(checked)
    reference = pd.read_csv(ROOT / "output/v2/announced_distribution_screen/predictions.csv")
    for name in ("Z0", "DivSchedule"):
        checked = data.validate_predictions(
            reference.loc[reference.model_id.eq(name), PREDICTION_COLUMNS]
        )
        checked["arm"] = name
        tables.append(checked)
    old = pd.read_csv(ROOT / "output/v2/residual_fuzzy/predictions.csv")
    checked = data.validate_predictions(old)
    checked["arm"] = "V2ResidualFuzzy"
    tables.append(checked)
    table = pd.concat(tables, ignore_index=True)
    event_dates = set(
        pd.read_csv(ROOT / "data/reference/srt_announced_distributions_dev.csv").ex_date
    )
    table["event"] = table.target_date.dt.strftime("%Y-%m-%d").isin(event_dates)

    def metric(g):
        error = g.y_pred.to_numpy() - g.y_true.to_numpy()
        return {
            "n": len(g),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "mae": float(np.mean(np.abs(error))),
        }

    per_seed = pd.DataFrame(
        [
            {"arm": arm, "seed": int(seed), **metric(g)}
            for (arm, seed), g in table.groupby(["arm", "seed"])
        ]
    )
    by_fold = pd.DataFrame(
        [
            {"arm": arm, "fold": fold, "seed": int(seed), **metric(g)}
            for (arm, fold, seed), g in table.groupby(["arm", "fold", "seed"])
        ]
    )
    event_scores = pd.DataFrame(
        [
            {"arm": arm, "seed": int(seed), "event": bool(event), **metric(g)}
            for (arm, seed, event), g in table.groupby(["arm", "seed", "event"])
        ]
    )
    pooled = per_seed.groupby("arm", as_index=False).agg(
        rmse=("rmse", "mean"), mae=("mae", "mean"), seeds=("seed", "nunique")
    )
    reference_score = pooled.set_index("arm").loc["DivSchedule"]
    pooled["rmse_gain_vs_schedule_pct"] = 100 * (1 - pooled.rmse / reference_score.rmse)
    candidate = pooled.set_index("arm").loc["gross/gaussian"]
    fold_means = by_fold.groupby(["arm", "fold"])[["rmse", "mae"]].mean()
    candidate_folds = fold_means.loc["gross/gaussian"]
    comparisons = []
    for arm in ("DivSchedule", "gross/linear", "gross/gate"):
        other = pooled.set_index("arm").loc[arm]
        wins = int((candidate_folds.rmse < fold_means.loc[arm].rmse - 1e-12).sum())
        paired_wins = None
        if arm != "DivSchedule":
            a = by_fold.loc[by_fold.arm.eq("gross/gaussian")].set_index(["fold", "seed"])
            b = by_fold.loc[by_fold.arm.eq(arm)].set_index(["fold", "seed"])
            paired_wins = int((a.rmse < b.rmse - 1e-12).sum())
        comparisons.append(
            {
                "reference": arm,
                "rmse_gain_pct": float(100 * (1 - candidate.rmse / other.rmse)),
                "mae_change_pct": float(100 * (candidate.mae / other.mae - 1)),
                "fold_wins": wins,
                "paired_wins": paired_wins,
            }
        )
    non_event = event_scores.loc[~event_scores.event].groupby("arm").rmse.mean()
    nonevent_gain = float(100 * (1 - non_event["gross/gaussian"] / non_event["DivSchedule"]))

    def passes(threshold):
        return bool(
            nonevent_gain >= 0.5
            and all(
                c["rmse_gain_pct"] >= threshold
                and c["mae_change_pct"] <= 1
                and c["fold_wins"] >= 2
                and (c["paired_wins"] is None or c["paired_wins"] >= 6)
                for c in comparisons
            )
        )

    m = pd.DataFrame(manifests)
    mechanism = m.groupby(["input_arm", "head"], as_index=False).agg(
        zero_fallbacks=("zero_fallback", "sum"),
        mean_max_weight=("mean_max_weight", "mean"),
        fraction_above_08=("fraction_max_weight_above_08", "mean"),
    )
    decision = {
        "experiment_id": pre["contract"]["experiment_id"],
        "primary_candidate": "gross/gaussian",
        "comparisons": comparisons,
        "non_event_rmse_gain_vs_schedule_pct": nonevent_gain,
        "material_goal_passed": passes(3),
        "candidate_floor_passed": passes(1),
        "total_runs": len(m),
        "zero_fallback_runs": int(m.zero_fallback.sum()),
        "selected_random_representation_runs": int((m.chosen_epoch == 0).sum()),
        "selected_updated_representation_runs": int((m.chosen_epoch > 0).sum()),
        "recorded_fit_and_export_seconds": float(m.inner_seconds.sum() + m.outer_seconds.sum()),
        "gaussian_runs_validated": fuzzy_runs,
        "status": (
            "development candidate"
            if passes(1)
            else "stop this training package; no demonstrated optimization"
        ),
    }
    for name, frame in (
        ("comparison_pooled.csv", pooled),
        ("comparison_by_seed.csv", per_seed),
        ("comparison_by_fold.csv", by_fold),
        ("event_scores.csv", event_scores),
        ("mechanism.csv", mechanism),
    ):
        frame.to_csv(OUT / name, index=False)
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(pooled.sort_values("rmse").to_string(index=False))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    analyze()
