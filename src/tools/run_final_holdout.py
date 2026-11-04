"""Run the once-only, development-frozen final holdout comparison.

Use --preflight to check the freeze without scoring or writing final predictions.
The full run refuses an existing final output directory, including partial runs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.export_contract import fuzzy_batch_tables
from src.models import build_model, parameter_count
from src.project_eval import Dataset, FORECAST_KEY, PREDICTION_COLUMNS, RUN_KEY
from src.train import BATCH_SIZE, _sha256, seed_everything, train_epoch


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "output/development/final_configuration_pretest.json"
FINAL_MODELS = ("LSTM32", "LSTM3", "Fuzzy2", "Gate2")
SEEDS = (11, 23, 37)


def preflight(config_path: Path = CONFIG) -> tuple[dict, Dataset]:
    frozen = json.loads(config_path.read_text(encoding="utf-8"))
    if frozen["status"] != "frozen_before_first_final_holdout_evaluation" or frozen["holdout_predictions_or_metrics_viewed"]:
        raise ValueError("Final configuration is not an unopened pretest freeze")
    if (tuple(frozen["final_models"]) != FINAL_MODELS or tuple(frozen["seeds"]) != SEEDS or
        frozen["primary_source_group"] != "all" or frozen["final_model_fits"] != 12 or
        frozen["primary_trading_threshold"] != 0 or
        frozen["primary_trading_cost_bps_per_side"] != 10 or
        frozen["price_only_cost_bps_per_side"] != [0, 10, 20] or
        not frozen["gross_distribution_secondary"]):
        raise ValueError("Frozen model, seed or policy settings differ from the approved plan")
    if set(frozen["final_refit_epochs_by_model"]) != set(FINAL_MODELS) or any(
        not isinstance(frozen["final_refit_epochs_by_model"][model], int) or
        frozen["final_refit_epochs_by_model"][model] < 1 for model in FINAL_MODELS
    ):
        raise ValueError("Invalid frozen refit epochs")
    for name, relative in (
        ("dataset_sha256", "data/processed/srt_windows_unscaled.npz"),
        ("fold_assignments_sha256", "data/processed/fold_assignments.csv"),
        ("train_source_sha256", "src/train.py"),
        ("model_source_sha256", "src/models.py"),
    ):
        if frozen[name] != _sha256(ROOT / relative):
            raise ValueError(f"Frozen {name} differs from the current file")
    data = Dataset(ROOT)
    train_ids, test_ids = data.ids("final_holdout", "train"), data.ids("final_holdout", "test")
    if (len(train_ids), len(test_ids)) != (frozen["final_train_windows"], frozen["final_eval_windows"]):
        raise ValueError("Final split differs from the frozen 931/251 windows")
    if data.index.loc[train_ids, "target_date"].max() >= data.index.loc[test_ids, "target_date"].min():
        raise ValueError("Final train/test chronology is invalid")
    policy = json.loads((config_path.parent / frozen["threshold_selection_file"]).read_text(encoding="utf-8"))
    if policy["selected"] != frozen["secondary_thresholds_development_selected"] or policy["holdout_viewed"]:
        raise ValueError("Selected thresholds differ from the development freeze")
    return frozen, data


def _fit_and_predict(data: Dataset, frozen: dict, model_id: str, seed: int,
                     run_root: Path, device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame | None,
                                                                     pd.DataFrame | None, dict]:
    prepared = data.prepare("final_holdout", "all")
    if len(prepared.train_ids) != 931 or len(prepared.eval_ids) != 251:
        raise ValueError("Unexpected final fit dimensions")
    seed_everything(seed)
    model = build_model(model_id, prepared.x_train.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    x_train = torch.from_numpy(prepared.x_train).to(device)
    y_train = torch.from_numpy(prepared.y_train).to(device)
    epochs = frozen["final_refit_epochs_by_model"][model_id]
    start_time = time.perf_counter()
    losses = [train_epoch(model, optimizer, x_train, y_train, seed=seed, epoch=epoch)
              for epoch in range(1, epochs + 1)]
    elapsed = time.perf_counter() - start_time
    model.eval()
    predictions, cases, rules = [], [], []
    x_test = torch.from_numpy(prepared.x_eval)
    with torch.no_grad():
        for start in range(0, len(x_test), BATCH_SIZE):
            stop = start + BATCH_SIZE
            batch = x_test[start:stop].to(device)
            ids = prepared.eval_ids[start:stop]
            if model_id == "Fuzzy2":
                values, diagnostics = model(batch, return_diagnostics=True)
                pred, case, rule = fuzzy_batch_tables(
                    model_id=model_id, source_group="all", fold="final_holdout",
                    seed=seed, window_ids=ids, prediction=values,
                    diagnostics=diagnostics, scaler=prepared.scaler)
                predictions.append(pred)
                cases.append(case)
                rules.append(rule)
            else:
                raw = prepared.scaler.y_inverse(model(batch).cpu().numpy())
                predictions.append(pd.DataFrame([
                    (model_id, "all", 0, "final_holdout", seed, int(window_id), float(value))
                    for window_id, value in zip(ids, raw)
                ], columns=PREDICTION_COLUMNS))
    pred_table = pd.concat(predictions, ignore_index=True)
    case_table = pd.concat(cases, ignore_index=True) if cases else None
    rule_table = pd.concat(rules, ignore_index=True) if rules else None
    run_dir = run_root / "final_holdout" / "all" / model_id / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    pred_table.to_csv(run_dir / "predictions.csv", index=False)
    if case_table is not None:
        case_table.to_csv(run_dir / "cases.csv", index=False)
        rule_table.to_csv(run_dir / "rules.csv", index=False)
    checkpoint = run_dir / "checkpoint.pt"
    torch.save({"model_id": model_id, "source_group": "all", "seed": seed,
                "fixed_epochs": epochs, "state_dict": model.cpu().state_dict(),
                "scaler": {"x_mean": prepared.scaler.x_mean.tolist(),
                           "x_std": prepared.scaler.x_std.tolist(),
                           "y_mean": prepared.scaler.y_mean, "y_std": prepared.scaler.y_std}}, checkpoint)
    manifest = {"status": "final_frozen_refit", "model_id": model_id,
                "source_group": "all", "fold": "final_holdout", "seed": seed,
                "rule_count": 2 if model_id == "Fuzzy2" else 0,
                "fixed_epochs": epochs, "train_windows": len(prepared.train_ids),
                "test_windows": len(prepared.eval_ids), "parameters": parameter_count(model),
                "train_mse_scaled_by_epoch": losses, "fit_seconds": elapsed,
                "checkpoint": str(checkpoint), "device": str(device),
                "dataset_sha256": frozen["dataset_sha256"],
                "model_source_sha256": frozen["model_source_sha256"],
                "train_source_sha256": frozen["train_source_sha256"]}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return pred_table, case_table, rule_table, manifest


def _forecast_summary(data: Dataset, predictions: pd.DataFrame, out: Path) -> pd.DataFrame:
    scores = data.forecast_scores(predictions)
    scores.to_csv(out / "forecast_by_seed.csv", index=False)
    summary = scores.groupby("model_id", as_index=False).agg(
        seeds=("seed", "nunique"), return_rmse_mean=("return_rmse", "mean"),
        return_rmse_sd=("return_rmse", "std"), return_mae_mean=("return_mae", "mean"),
        price_rmse_sgd_mean=("price_rmse_sgd", "mean"),
        direction_accuracy_mean=("direction_accuracy", "mean")
    ).sort_values("return_rmse_mean")
    summary.to_csv(out / "forecast_summary.csv", index=False)
    baseline = float(summary.loc[summary.model_id.eq("Z0"), "return_rmse_mean"].iloc[0])
    summary.assign(rmse_minus_zero=summary.return_rmse_mean - baseline).to_csv(
        out / "forecast_vs_zero.csv", index=False)
    paired = []
    for seed in SEEDS:
        ref = scores.loc[scores.model_id.eq("Fuzzy2") & scores.seed.eq(seed)].iloc[0]
        for benchmark in ("LSTM32", "LSTM3", "Gate2"):
            other = scores.loc[scores.model_id.eq(benchmark) & scores.seed.eq(seed)].iloc[0]
            paired.append({"seed": seed, "challenger": "Fuzzy2", "benchmark": benchmark,
                           "delta_rmse": ref.return_rmse - other.return_rmse,
                           "delta_mae": ref.return_mae - other.return_mae})
    pd.DataFrame(paired).to_csv(out / "forecast_paired_fuzzy2.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ordered = summary.sort_values("return_rmse_mean").reset_index(drop=True)
    for i, row in ordered.iterrows():
        error = 0 if pd.isna(row.return_rmse_sd) else row.return_rmse_sd
        ax.errorbar(row.return_rmse_mean, i, xerr=error, fmt="o", color="#0b7884", capsize=3)
    ax.set_yticks(np.arange(len(ordered)), ordered.model_id)
    ax.set_xlabel("Raw next-close log-return RMSE")
    ax.set_title("Final holdout; bars show seed SD")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "forecast_rmse.png", dpi=150)
    plt.close(fig)
    return summary


def _wealth_outputs(data: Dataset, predictions: pd.DataFrame, frozen: dict,
                    out: Path) -> pd.DataFrame:
    summaries = []
    for mode, cost in (("price_only", 0), ("price_only", 10),
                       ("price_only", 20), ("gross_distribution", 10)):
        ledger, wealth = data.portfolio_ledger(predictions, cost_bps=cost,
                                                threshold=0, wealth_mode=mode)
        ledger.to_csv(out / f"wealth_ledger_{mode}_{cost}bps.csv", index=False)
        wealth.to_csv(out / f"wealth_summary_{mode}_{cost}bps.csv", index=False)
        summaries.append(wealth)
        if cost == 10:
            _render_wealth_plot(ledger, mode, out)
    combined = pd.concat(summaries, ignore_index=True)
    signal = combined.loc[combined.strategy.eq("signal_long_cash")].groupby(
        ["wealth_mode", "cost_bps_per_side", "model_id"], as_index=False).agg(
            seeds=("seed", "nunique"), terminal_wealth_mean=("terminal_wealth_multiple", "mean"),
            terminal_wealth_sd=("terminal_wealth_multiple", "std"),
            drawdown_mean=("max_drawdown", "mean"),
            trades_mean=("one_way_trade_count", "mean"))
    controls = combined.loc[combined.model_id.eq("Z0") &
                            combined.strategy.ne("signal_long_cash")].copy()
    controls["model_id"] = controls["strategy"]
    controls = controls[["wealth_mode", "cost_bps_per_side", "model_id",
                         "terminal_wealth_multiple", "max_drawdown", "one_way_trade_count"]].copy()
    controls = controls.rename(columns={"terminal_wealth_multiple": "terminal_wealth_mean",
                                       "max_drawdown": "drawdown_mean",
                                       "one_way_trade_count": "trades_mean"})
    controls["seeds"] = 0
    controls["terminal_wealth_sd"] = np.nan
    display = pd.concat([signal, controls[signal.columns]], ignore_index=True).sort_values(
        ["wealth_mode", "cost_bps_per_side", "model_id"])
    display.to_csv(out / "wealth_comparison.csv", index=False)
    selected = frozen["secondary_thresholds_development_selected"]
    selected_rows, selected_ledgers = [], []
    for model in FINAL_MODELS:
        for seed in SEEDS:
            one = predictions.loc[predictions.model_id.eq(model) & predictions.seed.eq(seed)]
            ledger, wealth = data.portfolio_ledger(one, cost_bps=10,
                                                    threshold=float(selected[model]),
                                                    wealth_mode="price_only")
            selected_ledgers.append(ledger.loc[ledger.strategy.eq("signal_long_cash")])
            selected_rows.append(wealth.loc[wealth.strategy.eq("signal_long_cash")])
    pd.concat(selected_ledgers, ignore_index=True).to_csv(
        out / "wealth_ledger_selected_threshold_10bps.csv", index=False)
    pd.concat(selected_rows, ignore_index=True).to_csv(
        out / "wealth_selected_threshold_10bps.csv", index=False)
    return display


def _render_wealth_plot(ledger: pd.DataFrame, mode: str, out: Path) -> None:
    sample = ledger.loc[(ledger.seed.eq(11) | ledger.model_id.eq("Z0")) &
                        ledger.source_group.eq("all") & ledger.observed_open].copy()
    sample["date"] = pd.to_datetime(sample.date)
    fig, ax = plt.subplots(figsize=(10, 4))
    for model in ("Fuzzy2", "LSTM32", "LSTM3", "Gate2"):
        group = sample.loc[sample.model_id.eq(model) &
                           sample.strategy.eq("signal_long_cash")].sort_values("date")
        ax.plot(group.date, group.wealth, label=model)
    for strategy in ("momentum_long_cash", "always_long", "always_cash"):
        group = sample.loc[sample.model_id.eq("Z0") &
                           sample.strategy.eq(strategy)].sort_values("date")
        if len(group) == 0:
            raise ValueError(f"Missing final control path: {strategy}")
        ax.plot(group.date, group.wealth, label=strategy, linestyle="--")
    ax.set_title(f"Final holdout wealth, {mode}, 10 bps; seed 11")
    ax.set_ylabel("Wealth / initial SGD")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / f"wealth_paths_{mode}_10bps.png", dpi=150)
    plt.close(fig)


def refresh_plots(output_dir: Path) -> dict:
    """Correct presentation from saved ledgers, without touching predictions."""
    if not (output_dir / "execution_complete.json").exists():
        raise ValueError("Final analysis must be complete before plot refresh")
    figures = []
    for mode in ("price_only", "gross_distribution"):
        ledger = pd.read_csv(output_dir / f"wealth_ledger_{mode}_10bps.csv")
        _render_wealth_plot(ledger, mode, output_dir)
        figures.append(f"wealth_paths_{mode}_10bps.png")
    record = {"status": "plots_refreshed_from_saved_ledgers",
              "runner_sha256": _sha256(Path(__file__)), "figures": figures}
    (output_dir / "figure_refresh.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def _fuzzy_final_checks(data: Dataset, cases: pd.DataFrame, rules: pd.DataFrame,
                        run_root: Path, out: Path) -> None:
    records = []
    for seed in SEEDS:
        key = ("Fuzzy2", "all", 2, "final_holdout", seed)
        case = cases.loc[cases.seed.eq(seed)].set_index("window_id")
        rule = rules.loc[rules.seed.eq(seed)]
        alpha = rule.pivot(index="window_id", columns="rule_id", values="alpha")
        support = rule.pivot(index="window_id", columns="rule_id", values="log_membership").max(axis=1)
        expert = rule.pivot(index="window_id", columns="rule_id", values="expert_raw")
        run_dir = run_root / "final_holdout" / "all" / "Fuzzy2" / f"seed_{seed}"
        checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
        model = build_model("Fuzzy2", 6)
        model.load_state_dict(checkpoint["state_dict"])
        prepared = data.prepare("final_holdout", "all")
        train_alpha, train_support = [], []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(prepared.x_train), BATCH_SIZE):
                _, diagnostics = model(torch.from_numpy(prepared.x_train[start:start+BATCH_SIZE]),
                                       return_diagnostics=True)
                train_alpha.append(diagnostics["alpha"].cpu().numpy())
                train_support.append(diagnostics["log_membership"].max(dim=1).values.cpu().numpy())
        train_alpha = np.concatenate(train_alpha)
        train_support = np.concatenate(train_support)
        support_cut = float(np.quantile(train_support, .05))
        actual = data.y[alpha.index.to_numpy()]
        for rule_id in (1, 2):
            cutoff = float(np.quantile(train_alpha[:, rule_id-1], .75))
            mask = alpha[rule_id].to_numpy() >= cutoff
            other_id = 3 - rule_id
            records.append(dict(zip(RUN_KEY, key)) | {
                "rule_id": rule_id, "n": len(alpha), "alpha_train_q75": cutoff,
                "high_activation_n": int(mask.sum()),
                "mean_max_alpha": float(alpha.max(axis=1).mean()),
                "fraction_max_alpha_gt_08": float((alpha.max(axis=1) > .8).mean()),
                "support_train_p05": support_cut,
                "fraction_below_train_support_p05": float((support < support_cut).mean()),
                "activated_expert_mae": float(np.mean(np.abs(expert[rule_id].to_numpy()[mask] - actual[mask]))) if mask.any() else np.nan,
                "other_expert_mae": float(np.mean(np.abs(expert[other_id].to_numpy()[mask] - actual[mask]))) if mask.any() else np.nan,
                "mixture_mae_same_rows": float(np.mean(np.abs(case.loc[alpha.index[mask], "y_mix_raw"].to_numpy() - actual[mask]))) if mask.any() else np.nan,
            })
    pd.DataFrame(records).to_csv(out / "fuzzy_local_checks.csv", index=False)


def _horizon_final(data: Dataset, predictions: pd.DataFrame, out: Path) -> None:
    joined = data.validate_predictions(predictions)
    daily = data.daily.sort_values("date").set_index("date")
    following_date = pd.Series(daily.index.to_series().shift(-1).to_numpy(), index=daily.index)
    following_open = daily.srt_open.shift(-1)
    rows = joined[RUN_KEY + ["window_id", "forecast_date", "target_date", "y_true", "y_pred"]].copy()
    rows["entry_open"] = rows.target_date.map(daily.srt_open)
    rows["following_open_date"] = rows.target_date.map(following_date)
    rows["following_open"] = rows.target_date.map(following_open)
    end_date = data.index.loc[data.ids("final_holdout", "test")[-1], "target_date"]
    rows = rows.loc[(rows.following_open_date <= end_date) &
                    rows.entry_open.gt(0) & rows.following_open.gt(0)].copy()
    rows["trade_log_return"] = np.log(rows.following_open / rows.entry_open)
    rows.to_csv(out / "three_return_horizon_rows.csv", index=False)
    correlations = []
    for key, group in rows.groupby(RUN_KEY):
        correlations.append(dict(zip(RUN_KEY, key)) | {
            "n": len(group),
            "corr_pred_vs_true_close": group.y_pred.corr(group.y_true),
            "corr_pred_vs_trade_open": group.y_pred.corr(group.trade_log_return),
            "corr_true_close_vs_trade_open": group.y_true.corr(group.trade_log_return),
        })
    pd.DataFrame(correlations).to_csv(out / "three_return_horizon_correlations.csv", index=False)


def run(config_path: Path, output_dir: Path, run_root: Path, device_name: str) -> dict:
    frozen, data = preflight(config_path)
    if output_dir.exists() or run_root.exists():
        raise FileExistsError("Final output or run directory already exists; inspect it before any rerun")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    device = torch.device(device_name)
    output_dir.mkdir(parents=True)
    run_root.mkdir(parents=True)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    started = {"status": "final_execution_started", "frozen_config_sha256": _sha256(config_path),
               "git_revision_before_phase4_code_commit": revision,
               "runner_sha256": _sha256(Path(__file__)), "device": device_name,
               "rule": "one final evaluation; no automatic rerun after a partial execution"}
    (output_dir / "execution_started.json").write_text(json.dumps(started, indent=2), encoding="utf-8")
    predictions, cases, rules, manifests = [], [], [], []
    for model in FINAL_MODELS:
        for seed in SEEDS:
            pred, case, rule, manifest = _fit_and_predict(data, frozen, model, seed,
                                                          run_root, device)
            predictions.append(pred)
            if case is not None:
                cases.append(case)
                rules.append(rule)
            manifests.append(manifest)
            print(f"Fitted {model} seed {seed}, {manifest['fixed_epochs']} epochs", flush=True)
    learned = pd.concat(predictions, ignore_index=True)
    baseline = data.zero_predictions(include_holdout=True)
    baseline = baseline.loc[baseline.fold.eq("final_holdout")]
    all_predictions = pd.concat([learned, baseline], ignore_index=True)
    if len(all_predictions) != 13 * 251:
        raise ValueError("Final predictions do not contain exactly 12 fits and one baseline")
    data.validate_predictions(all_predictions)
    case_table = pd.concat(cases, ignore_index=True)
    rule_table = pd.concat(rules, ignore_index=True)
    data.validate_fuzzy_diagnostics(all_predictions, case_table, rule_table)
    all_predictions.to_csv(output_dir / "predictions.csv", index=False)
    case_table.to_csv(output_dir / "cases.csv", index=False)
    rule_table.to_csv(output_dir / "rules.csv", index=False)
    forecast = _forecast_summary(data, all_predictions, output_dir)
    wealth = _wealth_outputs(data, all_predictions, frozen, output_dir)
    _fuzzy_final_checks(data, case_table, rule_table, run_root, output_dir)
    _horizon_final(data, all_predictions, output_dir)
    completed = {**started, "status": "final_execution_complete",
                 "learned_fits": len(manifests), "prediction_rows": len(all_predictions),
                 "fuzzy_cases": len(case_table), "fuzzy_rule_rows": len(rule_table),
                 "forecast_summary": forecast[["model_id", "return_rmse_mean"]].to_dict(orient="records"),
                 "primary_wealth": wealth.loc[wealth.wealth_mode.eq("price_only") &
                                              wealth.cost_bps_per_side.eq(10),
                                              ["model_id", "terminal_wealth_mean"]].to_dict(orient="records"),
                 "missing_open_treatment": frozen["missing_2025_10_24_open"]}
    (output_dir / "execution_complete.json").write_text(json.dumps(completed, indent=2), encoding="utf-8")
    return completed


def resume_analysis(config_path: Path, output_dir: Path, run_root: Path) -> dict:
    """Finish deterministic analysis from the first run's saved predictions only.

    This recovery path does not instantiate or fit any forecasting model. It is
    allowed only for an interrupted analysis with all 12 frozen run files saved.
    """
    frozen, data = preflight(config_path)
    started_path = output_dir / "execution_started.json"
    if not started_path.exists() or (output_dir / "execution_complete.json").exists():
        raise ValueError("Recovery requires an incomplete first execution")
    started = json.loads(started_path.read_text(encoding="utf-8"))
    if started["frozen_config_sha256"] != _sha256(config_path):
        raise ValueError("The original execution used a different frozen configuration")
    predictions_path = output_dir / "predictions.csv"
    cases_path = output_dir / "cases.csv"
    rules_path = output_dir / "rules.csv"
    predictions = pd.read_csv(predictions_path)
    cases = pd.read_csv(cases_path)
    rules = pd.read_csv(rules_path)
    expected_runs = {(model, seed) for model in FINAL_MODELS for seed in SEEDS}
    actual_runs = set(zip(predictions.loc[predictions.model_id.ne("Z0"), "model_id"],
                          predictions.loc[predictions.model_id.ne("Z0"), "seed"]))
    if len(predictions) != 13 * 251 or actual_runs != expected_runs or len(cases) != 3 * 251 or len(rules) != 6 * 251:
        raise ValueError("Saved first-run exports are incomplete")
    data.validate_predictions(predictions)
    data.validate_fuzzy_diagnostics(predictions, cases, rules)
    for model, seed in sorted(expected_runs):
        run_dir = run_root / "final_holdout" / "all" / model / f"seed_{seed}"
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if (manifest["status"] != "final_frozen_refit" or
            manifest["fixed_epochs"] != frozen["final_refit_epochs_by_model"][model] or
            manifest["dataset_sha256"] != frozen["dataset_sha256"] or
            manifest["model_source_sha256"] != frozen["model_source_sha256"] or
            manifest["train_source_sha256"] != frozen["train_source_sha256"]):
            raise ValueError(f"Saved run manifest disagrees with freeze: {run_dir}")
        run_prediction = pd.read_csv(run_dir / "predictions.csv").sort_values("window_id")
        aggregate = predictions.loc[predictions.model_id.eq(model) &
                                    predictions.seed.eq(seed)].sort_values("window_id")
        if (len(run_prediction) != 251 or
            not np.array_equal(run_prediction.window_id.to_numpy(), aggregate.window_id.to_numpy()) or
            not np.allclose(run_prediction.y_pred, aggregate.y_pred, atol=0, rtol=0)):
            raise ValueError(f"Saved aggregate differs from original run: {run_dir}")
    forecast = _forecast_summary(data, predictions, output_dir)
    wealth = _wealth_outputs(data, predictions, frozen, output_dir)
    _fuzzy_final_checks(data, cases, rules, run_root, output_dir)
    _horizon_final(data, predictions, output_dir)
    completed = {**started, "status": "final_analysis_complete_from_original_predictions",
                 "recovery_reason": "Control-table column-name collision after all 12 fits and exports",
                 "recovery_runner_sha256": _sha256(Path(__file__)),
                 "prediction_file_sha256": _sha256(predictions_path),
                 "cases_file_sha256": _sha256(cases_path),
                 "rules_file_sha256": _sha256(rules_path),
                 "learned_fits": 12, "prediction_rows": len(predictions),
                 "fuzzy_cases": len(cases), "fuzzy_rule_rows": len(rules),
                 "forecast_summary": forecast[["model_id", "return_rmse_mean"]].to_dict(orient="records"),
                 "primary_wealth": wealth.loc[wealth.wealth_mode.eq("price_only") &
                                              wealth.cost_bps_per_side.eq(10),
                                              ["model_id", "terminal_wealth_mean"]].to_dict(orient="records"),
                 "missing_open_treatment": frozen["missing_2025_10_24_open"]}
    (output_dir / "execution_complete.json").write_text(json.dumps(completed, indent=2), encoding="utf-8")
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume-analysis", action="store_true",
                        help="Complete an interrupted analysis from original saved predictions; never refit")
    parser.add_argument("--refresh-plots", action="store_true",
                        help="Render figures from saved final ledgers; never refit or rescore")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--out", type=Path, default=ROOT / "output/final")
    parser.add_argument("--runs", type=Path, default=ROOT / "output/runs/final")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if sum((args.preflight, args.resume_analysis, args.refresh_plots)) > 1:
        parser.error("Choose one of --preflight, --resume-analysis, or --refresh-plots")
    if args.preflight:
        frozen, data = preflight(args.config)
        print(json.dumps({"status": "preflight_pass", "train_windows": frozen["final_train_windows"],
                          "test_windows": frozen["final_eval_windows"],
                          "models": frozen["final_models"], "seeds": frozen["seeds"],
                          "epochs": frozen["final_refit_epochs_by_model"],
                          "final_outputs_already_exist": args.out.exists() or args.runs.exists()}, indent=2))
    elif args.resume_analysis:
        print(json.dumps(resume_analysis(args.config, args.out, args.runs), indent=2))
    elif args.refresh_plots:
        print(json.dumps(refresh_plots(args.out), indent=2))
    else:
        print(json.dumps(run(args.config, args.out, args.runs, args.device), indent=2))


if __name__ == "__main__":
    main()
