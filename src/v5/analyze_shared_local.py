"""Audit all V5 runs and isolate model/training gains on the frozen task."""

import json

import numpy as np
import pandas as pd

from src.project_eval import Dataset, PREDICTION_COLUMNS, VALIDATION_FOLDS
from src.analysis_utils import circular_block_interval, sha256_file as sha, verify_hashes
from src.v5.shared_local import OUT, ROOT

PRIMARY = "gaussian_shared_local"


def metrics(group):
    error = group.y_pred.to_numpy() - group.y_true.to_numpy()
    return {
        "n": len(group),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
    }


def analyze():
    pre = json.loads((OUT / "pre_execution.json").read_text(encoding="utf-8"))
    verify_hashes(ROOT, pre["dependency_sha256"])
    complete = json.loads((OUT / "complete.json").read_text(encoding="utf-8"))
    if complete["runs"] != 45:
        raise ValueError("Expected 45 completed V5 runs")
    verify_hashes(
        OUT,
        {
            "predictions.csv": complete["prediction_sha256"],
            "manifests.json": complete["manifests_sha256"],
        },
    )
    data = Dataset(ROOT)
    predictions = pd.read_csv(OUT / "predictions.csv")
    manifests = json.loads((OUT / "manifests.json").read_text(encoding="utf-8"))
    if len(manifests) != 45:
        raise ValueError("Expected 45 V5 run manifests")
    gaussian_count, component_max_diff = 0, 0.0
    diagnostic_rows, local_regions = [], []
    aggregate = predictions.set_index(["arm", "fold", "seed", "window_id"]).sort_index()
    for m in manifests:
        directory = OUT / m["arm"] / m["fold"] / f"seed_{m['seed']}"
        verify_hashes(directory, m["output_sha256"])
        run = pd.read_csv(directory / "predictions.csv").sort_values("window_id")
        np.testing.assert_allclose(
            run.y_pred,
            aggregate.loc[(m["arm"], m["fold"], m["seed"])].sort_index().y_pred,
            rtol=0,
            atol=1e-15,
        )
        if m["head"] == "gaussian":
            data.validate_fuzzy_diagnostics(
                run, pd.read_csv(directory / "cases.csv"), pd.read_csv(directory / "rules.csv")
            )
            gaussian_count += 1
        comp = pd.read_csv(directory / "components.csv").sort_values("window_id")
        np.testing.assert_array_equal(comp.window_id, run.window_id)
        discrepancy = float(np.max(np.abs(comp.common_raw + comp.local_raw - run.y_pred)))
        component_max_diff = max(component_max_diff, discrepancy)
        if discrepancy >= 1e-8:
            raise ValueError("Common/local components do not reproduce predictions")
        if m["zero_fallback"]:
            if not (
                (run.y_pred == 0).all()
                and (comp.common_raw == 0).all()
                and (comp.local_raw == 0).all()
            ):
                raise ValueError("Zero fallback has nonzero predictions or components")
        for view, column in (
            ("drop_local_without_refit", "common_raw"),
            ("fitted_without_fallback", "fitted_without_fallback_raw"),
        ):
            frame = pd.DataFrame(
                {"y_pred": comp[column].to_numpy(), "y_true": data.y[comp.window_id.to_numpy()]}
            )
            diagnostic_rows.append(
                {
                    "arm": m["arm"],
                    "fold": m["fold"],
                    "seed": m["seed"],
                    "view": view,
                    "zero_fallback": m["zero_fallback"],
                    **metrics(frame),
                }
            )
        if (directory / "local_errors.csv").exists():
            regions = pd.read_csv(directory / "local_errors.csv")
            for key in ("arm", "fold", "seed"):
                regions[key] = m[key]
            local_regions.append(regions)
    tables = []

    def add(frame, arm):
        checked = data.validate_predictions(frame.loc[:, PREDICTION_COLUMNS])
        if set(checked.fold) != set(VALIDATION_FOLDS) or checked.target_date.max() > pd.Timestamp(
            "2025-09-30"
        ):
            raise ValueError("Only the three development folds are permitted")
        for _, g in checked.groupby("seed"):
            if len(g) != 378 or g.window_id.nunique() != 378:
                raise ValueError("Expected 378 unique development dates per seed")
        checked["arm"] = arm
        tables.append(checked)

    for arm, part in predictions.groupby("arm"):
        add(part, arm)
    add(data.zero_predictions(), "Z0")
    v1 = pd.read_csv(ROOT / "output/development/predictions.csv")
    for model_id in ("Fuzzy2", "LSTM3", "LSTM32", "Gate2"):
        add(v1.loc[v1.model_id.eq(model_id) & v1.source_group.eq("all")], "V1_" + model_id)
    ridge = pd.read_csv(ROOT / "output/v2/h1/predictions.csv")
    for model_id in ("Ridge", "MeanTrain"):
        part = ridge.loc[ridge.model_id.eq(model_id)]
        if part.empty:
            raise ValueError(f"Missing fixed-input reference {model_id}")
        add(part, model_id)
    table = pd.concat(tables, ignore_index=True)
    per_seed = pd.DataFrame(
        [
            {"arm": arm, "seed": int(seed), **metrics(part)}
            for (arm, seed), part in table.groupby(["arm", "seed"])
        ]
    )
    by_fold = pd.DataFrame(
        [
            {"arm": arm, "fold": fold, "seed": int(seed), **metrics(part)}
            for (arm, fold, seed), part in table.groupby(["arm", "fold", "seed"])
        ]
    )
    pooled = per_seed.groupby("arm", as_index=False).agg(
        rmse=("rmse", "mean"), mae=("mae", "mean"), seeds=("seed", "nunique")
    )
    score = pooled.set_index("arm")
    pooled["rmse_gain_vs_zero_pct"] = 100 * (1 - pooled.rmse / score.loc["Z0", "rmse"])
    pooled["rmse_gain_vs_v1_fuzzy_pct"] = 100 * (1 - pooled.rmse / score.loc["V1_Fuzzy2", "rmse"])
    fold_means = by_fold.groupby(["arm", "fold"])[["rmse", "mae"]].mean()
    candidate_pairs = by_fold.loc[by_fold.arm.eq(PRIMARY)].set_index(["fold", "seed"])
    comparisons = []
    for arm in score.index:
        if arm == PRIMARY:
            continue
        other = by_fold.loc[by_fold.arm.eq(arm)].set_index(["fold", "seed"])
        paired_wins = (
            None
            if score.loc[arm, "seeds"] == 1
            else int((candidate_pairs.rmse < other.rmse - 1e-12).sum())
        )
        comparisons.append(
            {
                "reference": arm,
                "rmse_gain_pct": float(
                    100 * (1 - score.loc[PRIMARY, "rmse"] / score.loc[arm, "rmse"])
                ),
                "mae_change_pct": float(
                    100 * (score.loc[PRIMARY, "mae"] / score.loc[arm, "mae"] - 1)
                ),
                "fold_wins": int(
                    (fold_means.loc[PRIMARY].rmse < fold_means.loc[arm].rmse - 1e-12).sum()
                ),
                "paired_wins": paired_wins,
            }
        )

    def passes(threshold, refs):
        selected = [c for c in comparisons if c["reference"] in refs]
        if len(selected) != len(refs):
            raise ValueError("A required comparison is missing")
        return bool(
            all(
                c["rmse_gain_pct"] >= threshold
                and c["mae_change_pct"] <= 1
                and c["fold_wins"] >= 2
                and (c["paired_wins"] is None or c["paired_wins"] >= 6)
                for c in selected
            )
        )

    # No fit or selection uses these paired-loss uncertainty summaries.
    table["squared_error"] = (table.y_pred - table.y_true) ** 2
    loss = table.groupby(["arm", "fold", "window_id"]).squared_error.mean()
    intervals = []
    for reference in (
        "Z0",
        "V1_Fuzzy2",
        "linear",
        "gate_shared_local",
        "gaussian_independent",
        "gaussian_common_matched",
    ):
        chunks = [
            (loss.loc[(PRIMARY, fold)] - loss.loc[(reference, fold)]).sort_index().to_numpy()
            for fold in VALIDATION_FOLDS
        ]
        for length in (5, 10, 20):
            mean, lower, upper = circular_block_interval(chunks, length)
            intervals.append(
                {
                    "candidate": PRIMARY,
                    "reference": reference,
                    "block_length": length,
                    "mean_mse_difference": mean,
                    "lower95": float(lower),
                    "upper95": float(upper),
                    "crosses_zero": bool(lower <= 0 <= upper),
                }
            )
    m = pd.DataFrame(manifests)
    mechanism = m.groupby("arm", as_index=False).agg(
        zero_fallbacks=("zero_fallback", "sum"),
        mean_best_epoch=("best_fitted_epoch", "mean"),
        mean_max_weight=("mean_max_weight", "mean"),
        mean_common_rms=("common_rms", "mean"),
        mean_local_rms=("local_rms", "mean"),
        mean_expert_difference=("mean_abs_expert_difference", "mean"),
        low_support_fraction=("fraction_raw_support_below_train_p05", "mean"),
    )
    material = passes(3, ["Z0", "V1_Fuzzy2", "V1_LSTM3"])
    candidate = passes(1, ["Z0", "V1_Fuzzy2", "V1_LSTM3"])
    decision = {
        "experiment_id": pre["contract"]["experiment_id"],
        "primary_candidate": PRIMARY,
        "comparisons": comparisons,
        "material_model_goal_passed": material,
        "model_candidate_floor_passed": candidate,
        "local_head_increment_candidate": passes(1, ["linear"]),
        "gaussian_specific_increment_candidate": passes(1, ["gate_shared_local"]),
        "total_runs": len(m),
        "zero_fallback_runs": int(m.zero_fallback.sum()),
        "selected_updated_model_runs": int(((~m.zero_fallback) & (m.best_fitted_epoch > 0)).sum()),
        "gaussian_runs_validated": gaussian_count,
        "max_component_identity_error": component_max_diff,
        "recorded_fit_and_export_seconds": float(m.inner_seconds.sum() + m.outer_seconds.sum()),
        "status": (
            "material development model candidate"
            if material
            else (
                "development candidate"
                if candidate
                else "stop this package; material model improvement not established"
            )
        ),
        "boundary": "model-only development exploration; no new information or final-period scoring; multi-round search prevents independent confirmation",
    }
    outputs = {
        "comparison_pooled.csv": pooled,
        "comparison_by_seed.csv": per_seed,
        "comparison_by_fold.csv": by_fold,
        "mechanism.csv": mechanism,
        "component_diagnostics.csv": pd.DataFrame(diagnostic_rows),
        "local_regions.csv": pd.concat(local_regions, ignore_index=True),
        "exploratory_block_intervals.csv": pd.DataFrame(intervals),
        "paired_errors.csv": table[
            ["arm", "fold", "seed", "window_id", "y_true", "y_pred", "squared_error"]
        ],
    }
    for filename, frame in outputs.items():
        frame.to_csv(OUT / filename, index=False)
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    (OUT / "analysis_record.json").write_text(
        json.dumps(
            {
                "analysis_source_sha256": sha(__file__),
                "analysis_utils_sha256": sha(ROOT / "src/analysis_utils.py"),
                "reference_sha256": {
                    str(p.relative_to(ROOT)): sha(p)
                    for p in (
                        ROOT / "output/development/predictions.csv",
                        ROOT / "output/v2/h1/predictions.csv",
                    )
                },
                "output_sha256": {name: sha(OUT / name) for name in outputs},
                "prediction_records_checked": len(table),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(pooled.sort_values("rmse").to_string(index=False))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    analyze()
