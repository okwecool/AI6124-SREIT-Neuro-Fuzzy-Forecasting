"""Development-only comparisons, fuzzy checks, horizon check and policy freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.project_eval import Dataset, FORECAST_KEY, RUN_KEY, VALIDATION_FOLDS


PRIMARY = ("LSTM32", "LSTM3", "Fuzzy2", "Gate2")
THRESHOLDS = (0.0, 0.001, 0.002, 0.003)


def _run_subset(frame: pd.DataFrame, key: tuple) -> pd.DataFrame:
    mask = np.logical_and.reduce([frame[column].eq(value).to_numpy()
                                  for column, value in zip(RUN_KEY, key)])
    return frame.loc[mask].copy()


def _save(frame: pd.DataFrame, folder: Path, name: str) -> None:
    frame.to_csv(folder / name, index=False)


def _plot_bar(frame: pd.DataFrame, x: str, y: str, title: str, target: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(frame[x].astype(str), frame[y], color="#0b7884")
    ax.set_title(title)
    ax.set_ylabel(y.replace("_", " "))
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(target, dpi=150)
    plt.close(fig)


def _plot_metric_dot(frame: pd.DataFrame, label: str, metric: str,
                     title: str, target: Path, spread: str | None = None) -> None:
    ordered = frame.sort_values(metric).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    locations = np.arange(len(ordered))
    for index, row in ordered.iterrows():
        error = row[spread] if spread and pd.notna(row[spread]) else 0.0
        ax.errorbar(row[metric], index, xerr=error, fmt="o", color="#0b7884",
                    capsize=3)
    ax.set_yticks(locations, ordered[label].astype(str))
    ax.set_xlabel("Raw next-day log-return RMSE")
    ax.set_title(title)
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(target, dpi=150)
    plt.close(fig)


def forecast_comparisons(data: Dataset, predictions: pd.DataFrame, out: Path) -> dict:
    scores = data.forecast_scores(predictions)
    pooled = data.pooled_validation_scores(predictions)
    _save(scores, out, "forecast_by_fold.csv")
    _save(pooled, out, "forecast_pooled_by_seed.csv")
    baseline = data.pooled_validation_scores(data.zero_predictions())
    primary = pd.concat([pooled.loc[pooled.source_group.eq("all")], baseline], ignore_index=True)
    primary_summary = primary.groupby(["model_id", "source_group", "rule_count"], as_index=False).agg(
        seeds=("seed", "nunique"), mean_return_rmse=("return_rmse", "mean"),
        sd_return_rmse=("return_rmse", "std"), mean_return_mae=("return_mae", "mean"),
        mean_price_rmse_sgd=("price_rmse_sgd", "mean"),
        mean_direction_accuracy=("direction_accuracy", "mean"),
    ).sort_values("mean_return_rmse")
    _save(primary_summary, out, "forecast_primary_summary.csv")
    _plot_metric_dot(primary_summary, "model_id", "mean_return_rmse",
                     "Pooled development RMSE; bars show seed SD",
                     out / "forecast_primary_rmse.png", "sd_return_rmse")
    paired = pd.concat([
        data.paired_validation_rmse_difference(predictions, challenger="Fuzzy2",
                                               benchmark=benchmark, source_group="all")
        for benchmark in ("LSTM32", "LSTM3", "Gate2")
    ], ignore_index=True)
    _save(paired, out, "forecast_paired_fuzzy2.csv")
    source = pooled.loc[pooled.model_id.eq("Fuzzy2")].groupby("source_group", as_index=False).agg(
        seeds=("seed", "nunique"), mean_return_rmse=("return_rmse", "mean"),
        sd_return_rmse=("return_rmse", "std"))
    _save(source, out, "fuzzy2_source_ablation.csv")
    _plot_metric_dot(source, "source_group", "mean_return_rmse",
                     "Fuzzy2 source ablation; bars show seed SD",
                     out / "fuzzy2_source_ablation.png", "sd_return_rmse")
    rule = pooled.loc[(pooled.source_group == "all") & (pooled.seed == 11) &
                      pooled.model_id.isin(("LSTM3", "Fuzzy2", "Fuzzy3"))].copy()
    rule["effective_k"] = rule.model_id.map({"LSTM3": 1, "Fuzzy2": 2, "Fuzzy3": 3})
    rule["k1_note"] = np.where(rule.effective_k == 1, "LSTM3 is functionally Fuzzy1", "")
    _save(rule, out, "rule_count_sensitivity_seed11.csv")
    _plot_metric_dot(rule, "effective_k", "return_rmse",
                     "Rule count sensitivity, seed 11", out / "rule_count_sensitivity.png")
    return {"primary": primary_summary, "paired": paired, "pooled": pooled}


def resource_summary(run_root: Path, out: Path) -> pd.DataFrame:
    records = []
    dataset_hashes, train_hashes, model_hashes = set(), set(), set()
    for path in sorted(run_root.rglob("manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        dataset_hashes.add(manifest["dataset_sha256"])
        train_hashes.add(manifest["train_source_sha256"])
        model_hashes.add(manifest["model_source_sha256"])
        records.append({key: manifest[key] for key in
                        ("model_id", "source_group", "rule_count", "fold", "seed", "device",
                         "parameters", "best_epoch", "best_inner_stop_mse_scaled",
                         "inner_seconds", "refit_seconds", "git_revision", "git_dirty")})
    if len(records) != 57 or any(len(hashes) != 1 for hashes in
                                 (dataset_hashes, train_hashes, model_hashes)):
        raise ValueError("Expected 57 runs from one frozen dataset and model/training code")
    resources = pd.DataFrame(records)
    _save(resources, out, "run_resources.csv")
    return resources


def _case_matrix(rule: pd.DataFrame, column: str) -> pd.DataFrame:
    return rule.pivot(index="window_id", columns="rule_id", values=column).sort_index()


def fuzzy_diagnostics(data: Dataset, joined: pd.DataFrame, cases: pd.DataFrame,
                      rules: pd.DataFrame, run_root: Path, out: Path) -> dict:
    rows, local_rows, bin_rows, cuts = [], [], [], {}
    daily = data.daily.set_index("date")
    for manifest_path in sorted(run_root.rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest["model_id"].startswith("Fuzzy"):
            continue
        key = tuple(manifest[column] for column in RUN_KEY)
        run_dir = manifest_path.parent
        train_rule = pd.read_csv(run_dir / "training_rules.csv")
        eval_rule = _run_subset(rules, key)
        eval_case = _run_subset(cases, key)
        actual = _run_subset(joined, key).set_index("window_id").sort_index()
        train_alpha = _case_matrix(train_rule, "alpha")
        alpha = _case_matrix(eval_rule, "alpha")
        log_m = _case_matrix(eval_rule, "log_membership")
        expert = _case_matrix(eval_rule, "expert_raw")
        support_train = _case_matrix(train_rule, "log_membership").max(axis=1)
        support_eval = log_m.max(axis=1)
        entropy = -(alpha * np.log(alpha.clip(lower=1e-12))).sum(axis=1)
        max_alpha = alpha.max(axis=1)
        checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
        centers = checkpoint["state_dict"]["centers"].numpy()
        scales = (0.1 + 1.9 * torch.sigmoid(checkpoint["state_dict"]["scale_logits"])).numpy()
        pairwise_sep = [float(np.linalg.norm(centers[i] - centers[j]))
                        for i in range(len(centers)) for j in range(i + 1, len(centers))]
        support_cut = float(support_train.quantile(0.05))
        info = dict(zip(RUN_KEY, key))
        rows.append({**info, "n": len(alpha), "mean_max_alpha": float(max_alpha.mean()),
                     "median_max_alpha": float(max_alpha.median()),
                     "mean_entropy": float(entropy.mean()),
                     "fraction_max_alpha_gt_08": float((max_alpha > 0.8).mean()),
                     "expert_spread_mean_raw": float((expert.max(axis=1) - expert.min(axis=1)).mean()),
                     "support_train_p05": support_cut,
                     "support_eval_p10": float(support_eval.quantile(.1)),
                     "support_eval_p50": float(support_eval.quantile(.5)),
                     "support_eval_p90": float(support_eval.quantile(.9)),
                     "fraction_below_train_support_p05": float((support_eval < support_cut).mean()),
                     "minimum_center_separation": min(pairwise_sep) if pairwise_sep else np.nan,
                     "centers_json": json.dumps(centers.tolist()),
                     "scales_json": json.dumps(scales.tolist()),
                     "center_first_coordinate_order_flip": bool(centers[0, 0] > centers[-1, 0])})
        run_cuts = {"alpha_q75": {}, "support_p05": support_cut}
        for rule_id in alpha.columns:
            cutoff = float(train_alpha[rule_id].quantile(.75))
            run_cuts["alpha_q75"][str(rule_id)] = cutoff
            high = alpha[rule_id] >= cutoff
            ids = alpha.index[high]
            target = actual.loc[ids, "y_true"].to_numpy()
            local_rows.append({**info, "rule_id": int(rule_id), "alpha_train_q75": cutoff,
                               "high_activation_n": len(ids),
                               "fixed_alpha_gt_08_n": int((alpha[rule_id] > .8).sum()),
                               "activated_expert_mae": float(np.mean(np.abs(expert.loc[ids, rule_id] - target))) if len(ids) else np.nan,
                               "other_expert_mae": float(np.mean(np.abs(expert.loc[ids, 3 - rule_id] - target)))
                               if len(ids) and len(alpha.columns) == 2 else np.nan,
                               "mixture_mae_same_rows": float(np.mean(np.abs(actual.loc[ids, "y_pred"] - target))) if len(ids) else np.nan})
        train_ids = train_alpha.index.to_numpy(dtype=int)
        eval_ids = alpha.index.to_numpy(dtype=int)
        train_dates = data.index.loc[train_ids, "forecast_date"]
        eval_dates = data.index.loc[eval_ids, "forecast_date"]
        for feature, column in (("volatility", "srt_volatility_20"),
                                ("momentum", "srt_momentum_5")):
            train_values = train_dates.map(daily[column]).to_numpy(dtype=float)
            eval_values = eval_dates.map(daily[column]).to_numpy(dtype=float)
            boundaries = np.quantile(train_values, [1 / 3, 2 / 3]).tolist()
            run_cuts[f"{feature}_terciles"] = boundaries
            bins = np.digitize(eval_values, boundaries, right=True)
            for number, label in enumerate(("low", "middle", "high")):
                selected = bins == number
                ids = eval_ids[selected]
                for rule_id in alpha.columns:
                    bin_rows.append({**info, "feature": feature, "bin": label,
                                     "training_cut_1": boundaries[0],
                                     "training_cut_2": boundaries[1],
                                     "rule_id": int(rule_id), "n": len(ids),
                                     "mean_alpha": float(alpha.loc[ids, rule_id].mean()) if len(ids) else np.nan,
                                     "mixture_mae_raw": float(np.mean(np.abs(actual.loc[ids, "y_pred"] - actual.loc[ids, "y_true"]))) if len(ids) else np.nan})
        cuts["/".join(map(str, key))] = run_cuts
    _save(pd.DataFrame(rows), out, "fuzzy_coverage.csv")
    _save(pd.DataFrame(local_rows), out, "fuzzy_local_errors.csv")
    _save(pd.DataFrame(bin_rows), out, "fuzzy_market_bins.csv")
    (out / "fuzzy_training_defined_cuts.json").write_text(json.dumps(cuts, indent=2), encoding="utf-8")
    return {"coverage_rows": len(rows), "local_error_rows": len(local_rows),
            "market_bin_rows": len(bin_rows)}


def horizon_diagnostic(data: Dataset, joined: pd.DataFrame, out: Path) -> pd.DataFrame:
    daily = data.daily.sort_values("date").set_index("date")
    next_date = pd.Series(daily.index.to_series().shift(-1).to_numpy(), index=daily.index)
    next_open = daily.srt_open.shift(-1)
    horizon = joined[RUN_KEY + ["window_id", "forecast_date", "target_date", "y_true", "y_pred"]].copy()
    horizon["entry_open"] = horizon.target_date.map(daily.srt_open)
    horizon["following_open_date"] = horizon.target_date.map(next_date)
    horizon["following_open"] = horizon.target_date.map(next_open)
    last_dates = {fold: data.index.loc[data.ids(fold, "validation")[-1], "target_date"]
                  for fold in VALIDATION_FOLDS}
    valid = ((horizon.following_open_date <= horizon.fold.map(last_dates)) &
             horizon.entry_open.gt(0) & horizon.following_open.gt(0))
    horizon = horizon.loc[valid].copy()
    horizon["trade_log_return"] = np.log(horizon.following_open / horizon.entry_open)
    _save(horizon, out, "three_return_horizon_rows.csv")
    correlations = []
    for key, group in horizon.groupby(RUN_KEY):
        correlations.append(dict(zip(RUN_KEY, key)) | {
            "n": len(group),
            "corr_pred_vs_true_close": group.y_pred.corr(group.y_true),
            "corr_pred_vs_trade_open": group.y_pred.corr(group.trade_log_return),
            "corr_true_close_vs_trade_open": group.y_true.corr(group.trade_log_return),
        })
    result = pd.DataFrame(correlations)
    _save(result, out, "three_return_horizon_correlations.csv")
    sample = horizon.loc[(horizon.model_id == "Fuzzy2") & (horizon.source_group == "all") &
                         (horizon.seed == 11) & (horizon.fold == "validation_3")].sort_values("target_date")
    fig, ax = plt.subplots(figsize=(11, 5))
    for column, label in (("y_true", "True close return"), ("y_pred", "Predicted close return"),
                          ("trade_log_return", "Realized next-open return")):
        ax.plot(sample.target_date, sample[column], label=label, linewidth=1)
    ax.set_title("Forecast/trade horizon, development fold 3")
    ax.set_ylabel("Log return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "three_return_horizon_sample.png", dpi=150)
    plt.close(fig)
    return result


def wealth_comparisons(predictions: pd.DataFrame, out: Path) -> pd.DataFrame:
    records = []
    for wealth_mode, cost in (("price_only", 0), ("price_only", 10),
                              ("price_only", 20), ("gross_distribution", 10)):
        summary = pd.read_csv(out / f"wealth_summary_{wealth_mode}_{cost}bps.csv")
        primary = summary.loc[summary.source_group.eq("all") &
                              summary.model_id.isin(PRIMARY) &
                              summary.strategy.eq("signal_long_cash")]
        for (model_id, seed), group in primary.groupby(["model_id", "seed"]):
            if set(group.fold) != set(VALIDATION_FOLDS):
                raise ValueError("Incomplete wealth folds")
            records.append({"wealth_mode": wealth_mode, "cost_bps": cost,
                            "policy": model_id, "seed": seed,
                            "compounded_three_fold_wealth": float(group.terminal_wealth_multiple.prod())})
        controls = summary.loc[(summary.model_id == "LSTM32") &
                               (summary.source_group == "all") & (summary.seed == 11) &
                               summary.strategy.isin(("momentum_long_cash", "always_long", "always_cash"))]
        for strategy, group in controls.groupby("strategy"):
            records.append({"wealth_mode": wealth_mode, "cost_bps": cost,
                            "policy": strategy, "seed": -1,
                            "compounded_three_fold_wealth": float(group.terminal_wealth_multiple.prod())})
    result = pd.DataFrame(records)
    _save(result, out, "wealth_compounded_by_seed.csv")
    picture = result.loc[(result.wealth_mode == "price_only") & (result.cost_bps == 10)].groupby(
        "policy", as_index=False).compounded_three_fold_wealth.mean().sort_values("compounded_three_fold_wealth")
    _plot_bar(picture, "policy", "compounded_three_fold_wealth",
              "Untuned development price wealth, 10 bps; three folds compounded",
              out / "wealth_compounded_primary.png")
    return result


def select_thresholds(data: Dataset, predictions: pd.DataFrame, out: Path) -> dict:
    primary = predictions.loc[predictions.source_group.eq("all") &
                              predictions.model_id.isin(PRIMARY)].copy()
    objectives = []
    for theta in THRESHOLDS:
        _, wealth = data.portfolio_ledger(primary, cost_bps=10, threshold=theta,
                                          wealth_mode="price_only")
        signal = wealth.loc[wealth.strategy.eq("signal_long_cash")]
        for (model_id, seed), group in signal.groupby(["model_id", "seed"]):
            if set(group.fold) != set(VALIDATION_FOLDS):
                raise ValueError("Threshold selection requires all three folds")
            objectives.append({"policy": model_id, "seed": int(seed), "theta": theta,
                               "sum_fold_log_wealth": float(np.log(group.terminal_wealth_multiple).sum())})
        momentum = wealth.loc[(wealth.model_id == "LSTM32") & (wealth.seed == 11) &
                              wealth.strategy.eq("momentum_long_cash")]
        objectives.append({"policy": "momentum_long_cash", "seed": -1, "theta": theta,
                           "sum_fold_log_wealth": float(np.log(momentum.terminal_wealth_multiple).sum())})
    frame = pd.DataFrame(objectives)
    _save(frame, out, "threshold_objective_by_seed.csv")
    grouped = frame.groupby(["policy", "theta"], as_index=False).sum_fold_log_wealth.mean().rename(
        columns={"sum_fold_log_wealth": "mean_sum_fold_log_wealth"})
    _save(grouped, out, "threshold_objective_mean.csv")
    winners = grouped.sort_values(["policy", "mean_sum_fold_log_wealth", "theta"],
                                  ascending=[True, False, True]).drop_duplicates("policy")
    manifest = {"status": "frozen_from_development_only", "source_group": "all",
                "wealth_mode": "price_only", "cost_bps": 10,
                "theta_grid": list(THRESHOLDS), "tie_break": "smaller theta",
                "objective": "mean across seeds of sum across folds of log terminal wealth",
                "selected": {row.policy: float(row.theta) for row in winners.itertuples()},
                "objectives": grouped.to_dict(orient="records"),
                "holdout_viewed": False}
    (out / "selected_policy_thresholds.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def freeze_pretest_configuration(resources: pd.DataFrame, run_root: Path,
                                 policy: dict, out: Path) -> dict:
    selected_models = ("LSTM32", "LSTM3", "Fuzzy2", "Gate2")
    core = resources.loc[resources.source_group.eq("all") &
                         resources.model_id.isin(selected_models)]
    counts = core.groupby("model_id").size()
    if any(int(counts.get(model_id, 0)) != 9 for model_id in selected_models):
        raise ValueError("Final epoch rule requires nine development fits per selected model")
    epochs = {model_id: int(core.loc[core.model_id.eq(model_id), "best_epoch"].median())
              for model_id in selected_models}
    first_manifest = json.loads(next(iter(sorted(run_root.rglob("manifest.json")))).read_text(encoding="utf-8"))
    frozen = {
        "status": "frozen_before_first_final_holdout_evaluation",
        "holdout_predictions_or_metrics_viewed": False,
        "primary_source_group": "all", "primary_target": "next raw SRT close log return",
        "final_train_windows": 931, "final_eval_windows": 251,
        "seeds": [11, 23, 37], "final_models": list(selected_models),
        "final_refit_epochs_by_model": epochs,
        "final_model_fits": 12,
        "zero_return_forecast_reference": True,
        "primary_forecast_metric": "raw-return RMSE",
        "primary_trading_threshold": 0.0,
        "secondary_thresholds_development_selected": policy["selected"],
        "threshold_selection_file": "selected_policy_thresholds.json",
        "price_only_cost_bps_per_side": [0, 10, 20],
        "primary_trading_cost_bps_per_side": 10,
        "gross_distribution_secondary": True,
        "distribution_source": "data/reference/srt_distributions_verified.csv",
        "missing_2025_10_24_open": "no_trade_and_carry; label constrained-data simulation",
        "trade_target_extension": "not adopted",
        "rule_interpretation_claim": "traceable latent-feature mixture; no market-regime-specialist claim",
        "dataset_sha256": first_manifest["dataset_sha256"],
        "fold_assignments_sha256": first_manifest["fold_assignments_sha256"],
        "train_source_sha256": first_manifest["train_source_sha256"],
        "model_source_sha256": first_manifest["model_source_sha256"],
        "development_run_count": len(resources),
    }
    (out / "final_configuration_pretest.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--aggregate", type=Path, default=Path("output/development"))
    parser.add_argument("--saved-only", action="store_true",
                        help="Audit saved exports without checkpoints or writing analysis files")
    args = parser.parse_args()
    out = args.aggregate
    out.mkdir(parents=True, exist_ok=True)
    data = Dataset()
    predictions = pd.read_csv(out / "predictions.csv")
    cases = pd.read_csv(out / "cases.csv")
    rules = pd.read_csv(out / "rules.csv")
    joined = data.validate_predictions(predictions)
    data.validate_fuzzy_diagnostics(predictions, cases, rules)
    if args.saved_only:
        summary = pd.read_csv(out / "forecast_primary_summary.csv")
        recomputed = data.pooled_validation_scores(predictions)
        for row in summary.itertuples():
            if row.model_id == "Z0":
                continue
            matches = recomputed.loc[recomputed.model_id.eq(row.model_id) &
                                     recomputed.source_group.eq(row.source_group) &
                                     recomputed.rule_count.eq(row.rule_count)]
            if len(matches) != row.seeds or not np.isclose(
                matches.return_rmse.mean(), row.mean_return_rmse, atol=1e-12, rtol=0
            ):
                raise ValueError(f"Saved development score differs: {row.model_id}/{row.source_group}")
        for name in ("fuzzy_coverage.csv", "fuzzy_local_errors.csv",
                     "fuzzy_training_defined_cuts.json", "wealth_compounded_by_seed.csv",
                     "selected_policy_thresholds.json", "final_configuration_pretest.json"):
            if not (out / name).exists():
                raise FileNotFoundError(out / name)
        print(json.dumps({"status": "saved_development_audit_pass",
                          "prediction_rows": len(predictions), "fuzzy_cases": len(cases),
                          "fuzzy_rule_rows": len(rules),
                          "score_groups": len(summary), "checkpoint_required": False}, indent=2))
        return
    if args.runs is None:
        parser.error("--runs is required unless --saved-only is used")
    resources = resource_summary(args.runs, out)
    forecast = forecast_comparisons(data, predictions, out)
    fuzzy = fuzzy_diagnostics(data, joined, cases, rules, args.runs, out)
    horizon = horizon_diagnostic(data, joined, out)
    wealth = wealth_comparisons(predictions, out)
    policy = select_thresholds(data, predictions, out)
    frozen = freeze_pretest_configuration(resources, args.runs, policy, out)
    print(json.dumps({"runs": len(resources), "fuzzy_diagnostics": fuzzy,
                      "horizon_rows": len(horizon), "wealth_compounded_rows": len(wealth),
                      "selected_thresholds": policy["selected"],
                      "final_refit_epochs": frozen["final_refit_epochs_by_model"],
                      "primary_rmse": forecast["primary"][["model_id", "mean_return_rmse"]].to_dict(orient="records")},
                     indent=2))


if __name__ == "__main__":
    main()
