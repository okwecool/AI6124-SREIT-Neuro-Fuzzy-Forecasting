"""Verify the saved earlier-CLR transfer probe and its fixed promotion rule."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.project_eval import Dataset
from src.analysis_utils import sha256_file, verify_hashes

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/v3/transfer_probe"


def run():
    pre = json.loads((OUT / "pre_execution.json").read_text(encoding="utf-8"))
    verify_hashes(ROOT, pre["dependency_sha256"])
    complete = json.loads((OUT / "complete.json").read_text(encoding="utf-8"))
    if (
        complete["runs"] != 54
        or sha256_file(OUT / "predictions.csv") != complete["prediction_sha256"]
    ):
        raise ValueError("Incomplete or changed transfer predictions")
    data = Dataset(ROOT)
    manifests = pd.DataFrame(json.loads((OUT / "manifests.json").read_text(encoding="utf-8")))
    for row in manifests.itertuples():
        path = OUT / row.arm / row.fold / f"seed_{row.seed}"
        verify_hashes(path, row.output_sha256)
        if row.arm.endswith("/gaussian"):
            data.validate_fuzzy_diagnostics(
                *[
                    pd.read_csv(path / name)
                    for name in ("predictions.csv", "cases.csv", "rules.csv")
                ]
            )
    prediction = pd.read_csv(OUT / "predictions.csv")
    references = pd.read_csv(ROOT / "output/v2/announced_distribution_screen/predictions.csv")
    reference = references.loc[references.model_id.eq("DivSchedule")].copy()
    reference["experiment_arm"] = "DivSchedule"
    prediction = pd.concat((prediction, reference), ignore_index=True)
    pooled_rows, fold_rows, event_rows = [], [], []
    events = set(pd.read_csv(ROOT / "data/reference/srt_announced_distributions_dev.csv").ex_date)
    for arm, g in prediction.groupby("experiment_arm"):
        s = data.pooled_validation_scores(g)
        s["arm"] = arm
        pooled_rows.append(s[["arm", "seed", "return_rmse", "return_mae"]])
        f = data.forecast_scores(g)
        f["arm"] = arm
        fold_rows.append(f[["arm", "fold", "seed", "return_rmse", "return_mae"]])
        v = data.validate_predictions(g)
        v["event"] = v.target_date.dt.strftime("%Y-%m-%d").isin(events)
        for (seed, event), part in v.groupby(["seed", "event"]):
            error = part.y_pred - part.y_true
            event_rows.append(
                {
                    "arm": arm,
                    "seed": int(seed),
                    "event": bool(event),
                    "n": len(part),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "mae": float(np.mean(np.abs(error))),
                }
            )
    seed_scores = pd.concat(pooled_rows, ignore_index=True)
    fold_scores = pd.concat(fold_rows, ignore_index=True)
    event_scores = pd.DataFrame(event_rows)
    pooled = seed_scores.groupby("arm")[["return_rmse", "return_mae"]].mean()
    candidate = pooled.loc["transfer/gaussian"]
    fold_means = fold_scores.groupby(["arm", "fold"])[["return_rmse", "return_mae"]].mean()
    comparisons = []
    for arm in pre["contract"]["references"]:
        other = pooled.loc[arm]
        wins = int(
            (
                fold_means.loc["transfer/gaussian"].return_rmse
                < fold_means.loc[arm].return_rmse - 1e-12
            ).sum()
        )
        paired = None
        if arm != "DivSchedule":
            a = fold_scores.loc[fold_scores.arm.eq("transfer/gaussian")].set_index(["fold", "seed"])
            b = fold_scores.loc[fold_scores.arm.eq(arm)].set_index(["fold", "seed"])
            paired = int((a.return_rmse < b.return_rmse - 1e-12).sum())
        comparisons.append(
            {
                "reference": arm,
                "rmse_gain_pct": float(100 * (1 - candidate.return_rmse / other.return_rmse)),
                "mae_change_pct": float(100 * (candidate.return_mae / other.return_mae - 1)),
                "fold_wins": wins,
                "paired_wins": paired,
            }
        )
    ne = event_scores.loc[~event_scores.event].groupby("arm").rmse.mean()
    ne_gain = float(100 * (1 - ne["transfer/gaussian"] / ne["DivSchedule"]))

    def passes(threshold):
        return bool(
            ne_gain >= 0.5
            and all(
                c["rmse_gain_pct"] >= threshold
                and c["fold_wins"] >= 2
                and c["mae_change_pct"] <= 1
                and (c["paired_wins"] is None or c["paired_wins"] >= 6)
                for c in comparisons
            )
        )

    decision = {
        "experiment_id": pre["contract"]["experiment_id"],
        "comparisons": comparisons,
        "non_event_rmse_gain_pct": ne_gain,
        "candidate_floor_passed": passes(1),
        "material_goal_passed": passes(3),
        "zero_fallbacks": manifests.groupby("arm").zero_fallback.sum().to_dict(),
        "source_windows": pre["source_windows"],
        "source_date_end": pre["last_source_target"],
        "status": (
            "candidate" if passes(1) else "stop frozen-transfer probe; no material optimization"
        ),
    }
    for name, frame in (
        ("comparison_pooled.csv", pooled.reset_index()),
        ("comparison_by_seed.csv", seed_scores),
        ("comparison_by_fold.csv", fold_scores),
        ("event_scores.csv", event_scores),
    ):
        frame.to_csv(OUT / name, index=False)
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    run()
