"""Compare V2 H1/H3 on the original development dates only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.project_eval import Dataset, VALIDATION_FOLDS
from src.train import SEEDS


ROOT = Path(__file__).resolve().parents[2]
H1 = ROOT / "output/v2/h1"
H3 = ROOT / "output/v2/h3"
OUT = ROOT / "output/v2/analysis"
EXPERIMENT_ID = "V2-H3-local-loss-01"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_h3(data: Dataset) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    plan = json.loads((H3 / "run_plan.json").read_text(encoding="utf-8"))
    if plan["h1_benchmark_sha256_before_runs"] != digest(H1 / "selected_benchmark_pre_h3.json"):
        raise ValueError("H1 benchmark changed after H3 began")
    predictions, cases, rules, cuts = [], [], [], {}
    for fold in VALIDATION_FOLDS:
        for seed in SEEDS:
            run = H3 / fold / "all/Fuzzy2" / f"seed_{seed}"
            manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
            if (manifest["experiment_id"] != EXPERIMENT_ID or manifest["fold"] != fold or
                manifest["seed"] != seed or manifest["local_weight"] != 0.1):
                raise ValueError(f"Wrong H3 manifest: {run}")
            for name, key in (("predictions.csv", "prediction_file_sha256"),
                              ("cases.csv", "cases_file_sha256"),
                              ("rules.csv", "rules_file_sha256")):
                if digest(run / name) != manifest[key]:
                    raise ValueError(f"H3 export changed: {run / name}")
            predictions.append(pd.read_csv(run / "predictions.csv"))
            cases.append(pd.read_csv(run / "cases.csv"))
            rules.append(pd.read_csv(run / "rules.csv"))
            cuts[(fold, seed)] = json.loads((run / "training_rule_cuts.json").read_text(encoding="utf-8"))
    pred = pd.concat(predictions, ignore_index=True)
    case = pd.concat(cases, ignore_index=True)
    rule = pd.concat(rules, ignore_index=True)
    data.validate_predictions(pred)
    data.validate_fuzzy_diagnostics(pred, case, rule)
    if len(pred) != 378 * 3 or len(case) != 378 * 3 or len(rule) != 378 * 3 * 2:
        raise ValueError("H3 run matrix is incomplete")
    return pred, case, rule, cuts


def forecast_tables(data: Dataset, h3_pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    old = pd.read_csv(ROOT / "output/development/predictions.csv")
    old = old.loc[old.source_group.eq("all") & old.model_id.isin(
        ("LSTM32", "LSTM3", "Fuzzy2", "Gate2"))]
    h1 = pd.read_csv(H1 / "predictions.csv")
    references = pd.concat([old, data.zero_predictions(), h1], ignore_index=True)
    ref_scores = data.forecast_scores(references)
    h3_scores = data.forecast_scores(h3_pred)
    h3_scores["model_id"] = "Fuzzy2+local"
    by_run = pd.concat([ref_scores, h3_scores], ignore_index=True)
    by_fold = by_run.groupby(["model_id", "fold"], as_index=False).agg(
        seeds=("seed", "nunique"), mean_rmse=("return_rmse", "mean"),
        mean_mae=("return_mae", "mean"))
    pooled_rows = []
    for model in by_run.model_id.unique():
        subset = by_run.loc[by_run.model_id.eq(model)]
        # A model/seed's pooled score must be computed from all 378 dates, not from fold RMSE averages.
        raw = h3_pred if model == "Fuzzy2+local" else references.loc[references.model_id.eq(model)]
        pooled = data.pooled_validation_scores(raw)
        pooled_rows.append({"model_id": model, "seeds": int(pooled.seed.nunique()),
                            "mean_pooled_rmse": float(pooled.return_rmse.mean()),
                            "mean_pooled_mae": float(pooled.return_mae.mean())})
    pooled_table = pd.DataFrame(pooled_rows).sort_values("mean_pooled_rmse")
    old_fuzzy = ref_scores.loc[ref_scores.model_id.eq("Fuzzy2")]
    new_fuzzy = h3_scores.loc[h3_scores.model_id.eq("Fuzzy2+local")]
    paired = new_fuzzy[["fold", "seed", "return_rmse", "return_mae"]].merge(
        old_fuzzy[["fold", "seed", "return_rmse", "return_mae"]],
        on=["fold", "seed"], suffixes=("_local", "_v1"), validate="one_to_one")
    paired["delta_rmse_local_minus_v1"] = paired.return_rmse_local - paired.return_rmse_v1
    paired["delta_mae_local_minus_v1"] = paired.return_mae_local - paired.return_mae_v1
    return pooled_table, by_fold, paired


def local_tables(data: Dataset, rules: pd.DataFrame, cuts: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = rules.pivot(index=["fold", "seed", "window_id"], columns="rule_id",
                       values=["alpha", "expert_raw"])
    wide.columns = [f"{name}_{rule}" for name, rule in wide.columns]
    wide = wide.reset_index()
    wide["y_true"] = data.y[wide.window_id.to_numpy(dtype=int)]
    c1, c2 = wide.alpha_1 * wide.expert_raw_1, wide.alpha_2 * wide.expert_raw_2
    wide["abs_contributions_sum"] = c1.abs() + c2.abs()
    wide["cancellation"] = 1 - (c1 + c2).abs() / (wide.abs_contributions_sum + 1e-15)
    wide["opposite_sign"] = wide.expert_raw_1 * wide.expert_raw_2 < 0
    wide["abs_expert_gap"] = (wide.expert_raw_1 - wide.expert_raw_2).abs()
    wide["max_alpha"] = wide[["alpha_1", "alpha_2"]].max(axis=1)
    metrics, regions = [], []
    for (fold, seed), group in wide.groupby(["fold", "seed"]):
        training = cuts[(fold, int(seed))]
        metrics.append({"fold": fold, "seed": int(seed), "n": len(group),
                        "mean_max_alpha": float(group.max_alpha.mean()),
                        "fraction_max_alpha_gt_08": float((group.max_alpha > .8).mean()),
                        "opposite_sign_fraction": float(group.opposite_sign.mean()),
                        "mean_cancellation": float(group.cancellation.mean()),
                        "mean_abs_contributions_sum": float(group.abs_contributions_sum.mean()),
                        "mean_abs_expert_gap": float(group.abs_expert_gap.mean())})
        for rule in (1, 2):
            cut = training["alpha_q75_by_rule"][rule - 1]
            selected = group.loc[group[f"alpha_{rule}"].gt(cut)]
            other = 3 - rule
            own_mae = float((selected[f"expert_raw_{rule}"] - selected.y_true).abs().mean()) if len(selected) else None
            other_mae = float((selected[f"expert_raw_{other}"] - selected.y_true).abs().mean()) if len(selected) else None
            regions.append({"fold": fold, "seed": int(seed), "rule_id": rule,
                            "train_q75_alpha": cut, "validation_days": len(selected),
                            "train_effective_sample_size": training["effective_sample_size_by_rule"][rule - 1],
                            "activated_expert_mae": own_mae, "other_expert_mae": other_mae,
                            "eligible_n_ge_20": len(selected) >= 20,
                            "local_advantage": bool(len(selected) >= 20 and own_mae < other_mae)})
    return pd.DataFrame(metrics), pd.DataFrame(regions)


def decide(pooled: pd.DataFrame, by_fold: pd.DataFrame, paired: pd.DataFrame,
           regions: pd.DataFrame) -> dict:
    benchmark = json.loads((H1 / "selected_benchmark_pre_h3.json").read_text(encoding="utf-8"))
    if benchmark["model_id"] not in set(pooled.model_id):
        raise ValueError("Fixed H1 benchmark is missing from H3 comparison")
    proposed = pooled.loc[pooled.model_id.eq("Fuzzy2+local")].iloc[0]
    fixed = pooled.loc[pooled.model_id.eq(benchmark["model_id"])].iloc[0]
    if abs(float(fixed.mean_pooled_rmse) - benchmark["mean_pooled_rmse"]) > 1e-12:
        raise ValueError("H1 benchmark score changed")
    relative_gain = float((fixed.mean_pooled_rmse - proposed.mean_pooled_rmse) /
                          fixed.mean_pooled_rmse)
    mae_change = float((proposed.mean_pooled_mae - fixed.mean_pooled_mae) /
                       fixed.mean_pooled_mae)
    fold_pair = by_fold.loc[by_fold.model_id.isin(("Fuzzy2+local", benchmark["model_id"]))]
    fold_wide = fold_pair.pivot(index="fold", columns="model_id", values="mean_rmse")
    folds_better = int((fold_wide["Fuzzy2+local"] < fold_wide[benchmark["model_id"]]).sum())
    parent_pair_wins = int(paired.delta_rmse_local_minus_v1.lt(0).sum())
    local_fold_wins = 0
    for fold, group in regions.groupby("fold"):
        by_seed = group.pivot(index="seed", columns="rule_id", values="local_advantage")
        if bool((by_seed[1] & by_seed[2]).any()):
            local_fold_wins += 1
    forecast_gate = bool(relative_gain >= .01 and folds_better >= 2 and
                         parent_pair_wins >= 6 and mae_change <= .01)
    local_gate = bool(local_fold_wins >= 2)
    return {"status": "development_screen_only", "fixed_h1_benchmark": benchmark["model_id"],
            "fixed_h1_benchmark_rmse": float(fixed.mean_pooled_rmse),
            "h3_mean_pooled_rmse": float(proposed.mean_pooled_rmse),
            "relative_rmse_gain_vs_fixed_reference": relative_gain,
            "relative_mae_change_vs_fixed_reference": mae_change,
            "folds_better_than_fixed_reference": folds_better,
            "paired_fold_seed_wins_vs_v1_fuzzy2": parent_pair_wins,
            "folds_with_both_rules_local_advantage_in_same_seed": local_fold_wins,
            "forecast_promotion_gate_passed": forecast_gate,
            "local_specialization_gate_passed": local_gate,
            "both_gates_passed": forecast_gate and local_gate,
            "final_holdout_used_for_v2_selection": False}


def run() -> dict:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"V2 analysis already exists: {OUT}")
    data = Dataset(ROOT)
    predictions, cases, rules, cuts = load_h3(data)
    pooled, by_fold, paired = forecast_tables(data, predictions)
    mechanism, regions = local_tables(data, rules, cuts)
    decision = decide(pooled, by_fold, paired, regions)
    OUT.mkdir(parents=True, exist_ok=False)
    for name, frame in (("h3_predictions.csv", predictions), ("h3_cases.csv", cases),
                        ("h3_rules.csv", rules), ("comparison_pooled.csv", pooled),
                        ("comparison_by_fold.csv", by_fold), ("paired_local_vs_v1.csv", paired),
                        ("mechanism_by_run.csv", mechanism), ("local_regions.csv", regions)):
        frame.to_csv(OUT / name, index=False)
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return decision


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
