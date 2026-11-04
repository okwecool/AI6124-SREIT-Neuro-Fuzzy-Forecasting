"""Build and execute the single Group 1 course Notebook from frozen results."""

import argparse
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs" / "AI6124_Group1_Final_Project.ipynb"


def md(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


literature = (ROOT / "docs" / "literature_review_draft_en.md").read_text(encoding="utf-8")
literature = literature.replace("# Literature Review and Result Comparison — LSTM-Encoded Fuzzy Local Experts", "## Literature review and result comparison")

notebook = nbf.v4.new_notebook()
notebook.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
notebook.cells = [
    md("""
# AI6124 Group 1 — LSTM-Encoded Fuzzy Local Experts for SRT

**Zou Runze (Ryan) · Ngeow Shan Yong Destin**  
**Final course Notebook, 4 November 2026.**

This single Notebook contains the method, code entry points, chronological data checks, development and frozen final results, interpretation, trading simulation, literature comparison and conclusion. It reads the already saved final predictions; rerunning these cells does not refit a model or select settings on the final holdout. Protocol v1.2 is in `docs/experiment_protocol_lstm_fuzzy_local_experts.md`.

**Research question:** Can an LSTM learn three features from recent SRT/SORA/STI data that support a two-rule fuzzy mixture of local experts for next-day raw SRT return prediction?
"""),
    md(literature),
    md("""
## Data and information timing

The target is the SGD counter of the CSOP iEdge S-REIT Leaders Index ETF (`SRT.SI`). The primary label is the next-trading-day **raw close-to-close log return**. Each forecast uses the latest 20 SRT trading-day feature rows available after the forecast day's close. SORA is aligned by **publication date**. The saved 28 October snapshot combines SRT, STI and MAS SORA; its raw responses and source hashes are retained in `data/`. The 2025-10-24 SRT close was repaired with a documented second source; its open remains unavailable.

We use three expanding forward-time validation folds, followed by an untouched final holdout. No scaler is fitted on validation or test rows. `src/project_eval.py` provides the shared split and scoring contract.
"""),
    code("""
from pathlib import Path
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

project_root = next(
    (candidate for candidate in (Path.cwd(), Path.cwd().parent)
     if (candidate / "data" / "source_manifest.json").exists()),
    None,
)
if project_root is None:
    raise FileNotFoundError("Open this Notebook from the project root or docs/ directory")
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from src.project_eval import Dataset, FEATURE_GROUPS, VALIDATION_FOLDS

subprocess.run([sys.executable, "data/verify_dataset.py"], check=True)
data = Dataset()
print("X shape:", data.x.shape)
print("Features:", data.features)
"""),
    code("""
fold_summary = []
for fold in (*VALIDATION_FOLDS, "final_holdout"):
    role = "test" if fold == "final_holdout" else "validation"
    train_ids, eval_ids = data.ids(fold, "train"), data.ids(fold, role)
    fold_summary.append({
        "fold": fold,
        "train_windows": len(train_ids),
        "eval_windows": len(eval_ids),
        "first_eval_target": data.index.loc[eval_ids[0], "target_date"].date(),
        "last_eval_target": data.index.loc[eval_ids[-1], "target_date"].date(),
    })
display(pd.DataFrame(fold_summary))
"""),
    code("""
daily = data.daily.loc[data.daily.date <= pd.Timestamp("2025-09-30")].copy()
fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
axes[0].plot(daily.date, daily.srt_close, color="#0c6b76", linewidth=1.2)
axes[0].set_ylabel("SRT close (SGD)")
axes[1].plot(daily.date, daily.sora_rate_pct_asof, color="#d77931", linewidth=1.0)
axes[1].set_ylabel("Published SORA (%)")
axes[2].plot(daily.date, daily.sti_close_asof, color="#52699a", linewidth=1.0)
axes[2].set_ylabel("STI close")
axes[2].set_xlabel("Date")
fig.suptitle("Three source series in the development period")
plt.show()
"""),
    md("""
## Pretest data diagnostics

Inspect SRT close and daily log returns only through the last pretest target. Differencing removes much of the price-level trend but does not guarantee stationary returns or constant volatility. The missing-open date below comes from source-audit metadata; no final-holdout return pattern is plotted or used to revise the model.
"""),
    code("""
pretest_end = data.index.loc[data.ids("final_holdout", "train")[-1], "target_date"]
pretest = data.daily.loc[data.daily.date <= pretest_end, ["date", "srt_close", "srt_open"]].copy()
pretest["log_return"] = np.log(pretest.srt_close).diff()
pretest["rolling_mean_20"] = pretest.log_return.rolling(20).mean()
pretest["rolling_std_20"] = pretest.log_return.rolling(20).std()
diagnostic_dir = Path("output/runs/tmp/phase1_smoke")
diagnostic_dir.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
axes[0].plot(pretest.date, pretest.srt_close, linewidth=1, color="#0c6b76")
axes[0].set_ylabel("SRT close (SGD)")
axes[1].plot(pretest.date, pretest.log_return, linewidth=.8, color="#52699a")
axes[1].set_ylabel("Daily log return")
axes[2].plot(pretest.date, pretest.rolling_mean_20, label="20-day mean")
axes[2].plot(pretest.date, pretest.rolling_std_20, label="20-day std")
axes[2].legend()
axes[2].set_xlabel("Date")
fig.savefig(diagnostic_dir / "pretest_price_return_rolling.png", dpi=150)
plt.show()
returns = pretest.log_return.dropna()
fig, ax = plt.subplots(figsize=(8, 3))
ax.hist(returns, bins=40, color="#0c6b76")
ax.set_title("Pretest daily SRT log-return distribution")
fig.savefig(diagnostic_dir / "pretest_return_distribution.png", dpi=150)
plt.show()
print("Pretest rows:", len(pretest), "zero returns:", int(returns.eq(0).sum()),
      "zero fraction:", round(float(returns.eq(0).mean()), 4))
print("Missing SRT opens in frozen snapshot (date metadata only):",
      data.daily.loc[data.daily.srt_open.isna(), "date"].dt.date.tolist())
"""),
    md("""
## Models and benchmark

- `Z0`: predict zero next-day return (unchanged closing price).
- `LSTM32`: one-layer, 32-hidden-unit LSTM with a scalar return head.
- `LSTM3`: the same encoder with a three-dimensional `tanh` projection and scalar linear head.
- `Fuzzy2`: the same encoder and projection, then two diagonal Gaussian rules and two affine return experts.

The fuzzy system receives only the three learned latent values, not a flattened 20 × 6 input. `Fuzzy2` predicts a rule-weighted return. `Gate2` is the predeclared small non-Gaussian two-expert control. Implementations are in `src/models.py`; the inner-stopping/outer-refit runner is `src/train.py`. The encoder, fuzzy weights and experts are trained only on each fold's training rows; outer validation rows are used only for evaluation.
"""),
    code("""
# Example of the shared train-only scaling and inner stopping split.
inner = data.prepare("validation_1", "all", inner_stop=True)
outer = data.prepare("validation_1", "all", inner_stop=False)
print("Inner train / stopping:", inner.x_train.shape, inner.x_eval.shape)
print("Outer refit / evaluation:", outer.x_train.shape, outer.x_eval.shape)
print("Outer train scaled target mean:", round(float(outer.y_train.mean()), 6))
"""),
    md("""
## No-change baseline on the three development folds

The no-change predictor needs no fitting. The code below shows its three development folds; the frozen final comparison appears later. Directional accuracy excludes zero true returns; the majority-direction reference is learned from each fold's training labels.
"""),
    code("""
zero_predictions = data.zero_predictions()
zero_scores = data.forecast_scores(zero_predictions)
zero_pooled = data.pooled_validation_scores(zero_predictions)
display(zero_scores[[
    "fold", "n", "return_rmse", "return_mae", "price_rmse_sgd",
    "direction_n", "zero_actual_n", "train_majority_accuracy"
]])
display(zero_pooled)
"""),
    code("""
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(zero_scores.fold, zero_scores.return_rmse, color="#0c6b76")
ax.set_ylabel("Raw next-day log-return RMSE")
ax.set_title("Development-fold no-change baseline")
ax.tick_params(axis="x", rotation=15)
plt.tight_layout()
plt.show()
"""),
    md("""
## Trained-model predictions and result analysis

The development runner exports complete outer-validation predictions, combined at `output/development/predictions.csv`, with these columns:

`model_id,source_group,rule_count,fold,seed,window_id,y_pred`

`y_pred` must be in **raw log-return units**, after inverse target scaling. The evaluator validates the full seven-column key and fuzzy case/rule diagnostics. Do not infer that a rule is a particular market regime from its numeric label.
"""),
    code("""
prediction_file = Path("output/development/predictions.csv")
model_predictions = None
if prediction_file.exists():
    model_predictions = pd.read_csv(prediction_file)
    model_scores = data.forecast_scores(model_predictions)
    model_pooled = data.pooled_validation_scores(model_predictions)
    display(model_scores.sort_values(["model_id", "source_group", "rule_count", "fold", "seed"]))
    display(model_pooled.sort_values("return_rmse"))
    if {"Fuzzy2", "LSTM32"} <= set(model_predictions.model_id):
        display(data.paired_validation_rmse_difference(model_predictions))
else:
    print(f"Awaiting trained-model predictions: {prediction_file}")
"""),
    code("""
if model_predictions is not None:
    joined = data.validate_predictions(model_predictions)
    primary = joined.loc[
        (joined.model_id == "Fuzzy2") & (joined.source_group == "all")
        & (joined.seed == 11) & (joined.fold != "final_holdout")
    ].sort_values("target_date")
    if len(primary):
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
        axes[0].plot(primary.target_date, primary.y_true, label="Actual", linewidth=1)
        axes[0].plot(primary.target_date, primary.y_pred, label="Fuzzy2", linewidth=1)
        axes[0].set_ylabel("Next-day log return")
        axes[0].legend()
        overview = model_scores.loc[model_scores.fold != "final_holdout"].groupby(
            ["model_id", "source_group", "rule_count"], as_index=False
        ).return_rmse.mean().sort_values("return_rmse")
        labels = overview.model_id + " / " + overview.source_group + " / K=" + overview.rule_count.astype(str)
        axes[1].barh(labels, overview.return_rmse, color="#0c6b76")
        axes[1].set_xlabel("Mean fold return RMSE (descriptive; use pooled scores for ranking)")
        plt.show()
    else:
        print("Awaiting full-feature Fuzzy2, seed 11, for the forecast plot.")
"""),
    md("""
## Fuzzy-rule diagnostics

Protocol v1.2 requires a case table with the full `(model_id,source_group,rule_count,fold,seed,window_id)` key and `z1,z2,z3,y_mix_raw`, plus a rule-long table with that key and `rule_id,alpha,expert_raw,log_membership,distance2`. The validator checks one case and exactly `rule_count` rule rows per fuzzy forecast, including the raw-return weighted-mixture identity. The saved training-cut local-error analysis appears below; rule-number switching across seeds limits semantic labels.
"""),
    code("""
case_file = Path("output/development/cases.csv")
rule_file = Path("output/development/rules.csv")
if case_file.exists() and rule_file.exists() and model_predictions is not None:
    fuzzy_cases, fuzzy_rules = data.validate_fuzzy_diagnostics(
        model_predictions, pd.read_csv(case_file), pd.read_csv(rule_file))
    print("Validated fuzzy cases:", len(fuzzy_cases), "rule rows:", len(fuzzy_rules))
    display(fuzzy_rules.groupby(["model_id", "source_group", "rule_count", "rule_id"]).alpha.agg(["count", "mean", "min", "max"]))
elif case_file.exists() or rule_file.exists():
    print("Both fuzzy diagnostic files and the prediction file are required for validation.")
else:
    print(f"Awaiting fuzzy diagnostics: {case_file} and {rule_file}")
"""),
    md("""
## Portfolio application — causal observed-open ledger

A signal formed after close `t` is first executable at the next SRT open. The realized open-to-open interval differs from the close-to-close forecast label. The ledger trades only at observed opens, carries shares/cash across a missing opening quote and liquidates at the known fold boundary. It records model signals, a model-free five-day momentum signal, always-long and always-cash on identical dates. Costs are 10 basis points per side here. The nine SRT distribution events have issuer corroboration in `docs/srt_distribution_source_audit.md`; show gross-distribution and price-only wealth separately. The 2025-10-24 open remains missing, so any final-period trade path must be labelled a constrained-data simulation.
"""),
    code("""
zero_ledger, zero_wealth = data.portfolio_ledger(zero_predictions, cost_bps=10)
_, zero_total_wealth = data.portfolio_ledger(
    zero_predictions, cost_bps=10, wealth_mode="gross_distribution")
display(zero_wealth[["fold", "strategy", "terminal_wealth_multiple",
                     "max_drawdown", "one_way_trade_count"]])
display(zero_total_wealth[["fold", "strategy", "terminal_wealth_multiple",
                           "distribution_cash_sgd_per_initial_sgd"]])
first_fold = zero_ledger.loc[zero_ledger.fold == "validation_1"]
fig, ax = plt.subplots(figsize=(10, 4))
for strategy in ("always_long", "momentum_long_cash", "always_cash"):
    path = first_fold.loc[first_fold.strategy == strategy]
    ax.plot(path.date, path.wealth, label=strategy)
ax.legend()
ax.set_ylabel("Price-only wealth / initial SGD")
ax.set_title("Causal development wealth paths, validation 1")
plt.show()
if model_predictions is not None:
    model_wealth = pd.read_csv("output/development/wealth_summary_price_only_10bps.csv")
    display(model_wealth.loc[
        model_wealth.source_group.eq("all") & model_wealth.seed.eq(11)
        & model_wealth.strategy.eq("signal_long_cash"),
        ["model_id", "fold", "seed", "terminal_wealth_multiple"]])
else:
    print("Trained-model wealth paths await complete validation predictions.")
"""),
    md("""
## Development-only comparison and interpretation

These three forward validation folds were used to compare models and select the secondary trading threshold. They are **not** the reserved final test. The complete analysis and qualified conclusions are in `docs/development_results_2026-11-03.md`. In particular, the zero-return baseline currently has the lowest pooled RMSE, the untuned fuzzy strategy does not beat the model-free trading controls, and rule weights do not establish distinct economic regimes. The approved six-feature model remains the primary specification regardless of the small source-ablation differences.
"""),
    code("""
development_dir = Path("output/development")
if (development_dir / "forecast_primary_summary.csv").exists():
    display(pd.read_csv(development_dir / "forecast_primary_summary.csv"))
    display(pd.read_csv(development_dir / "forecast_paired_fuzzy2.csv"))
    display(pd.read_csv(development_dir / "fuzzy2_source_ablation.csv"))
    display(pd.read_csv(development_dir / "rule_count_sensitivity_seed11.csv"))
    display(pd.read_csv(development_dir / "fuzzy_coverage.csv").query(
        "model_id == 'Fuzzy2' and source_group == 'all'")[[
            "fold", "seed", "mean_max_alpha", "fraction_max_alpha_gt_08",
            "fraction_below_train_support_p05"]])
    print("Frozen development threshold choices:")
    print((development_dir / "selected_policy_thresholds.json").read_text(encoding="utf-8")[:850])
else:
    print("Development comparison files are not present yet.")
"""),
    md("""
## Frozen final evaluation and course conclusion

The next cells read `output/final/`. The 931/251 train/test split, four models, three seeds, fixed final epochs, baselines and policies were frozen before these 251 targets were scored. All final model predictions were generated once. A control-table error occurred after those predictions had been saved; the analysis resumed from the identical files without a second fit. The execution records and hashes preserve that sequence.
"""),
]

notebook.cells.extend([
    md("""
### Executable model and training interface

`src/models.py` contains the 32-unit LSTM encoder, compact 3D projection, two diagonal-Gaussian antecedents and affine experts. For `Fuzzy2`, `z∈R³`, `αₖ=softmax(-dₖ²/2)`, and `ŷ=Σₖαₖeₖ(z)`. The centers start at `(-0.25,0,0)` and `(0.25,0,0)`; there is no K-means initialization or auxiliary clustering loss. `src/train.py` trains standardized returns with AdamW, global MSE, chronological inner stopping and same-seed outer refit. Final refit epochs are fixed medians of the nine development best epochs, not chosen from final labels.

To reproduce training from scratch in a **new** directory, use `python -m src.tools.run_development_matrix --include-gate --device cpu --out output/runs/reproduction/development`, then the commands in `docs/README_DESTIN_HANDOFF.md`. The once-only final exports are already stored; this Notebook performs read-only final analysis.
"""),
    code("""
from src.models import build_model, parameter_count
from src.tools.run_final_holdout import preflight
frozen, _ = preflight()
model_sizes = pd.DataFrame([
    {"model": name, "parameters": parameter_count(build_model(name, 6)),
     "final_fixed_epochs": frozen["final_refit_epochs_by_model"][name]}
    for name in frozen["final_models"]
])
display(model_sizes)
print("Final train/test windows:", frozen["final_train_windows"], frozen["final_eval_windows"])
"""),
    md("""
### Final forecasting result

The primary metric is raw next-close log-return RMSE. `Z0` predicts zero return, equivalent to an unchanged closing price. Final `Fuzzy2` RMSE is 0.006349, about 0.44% below `Z0` and 0.55% below ordinary `LSTM32`; the absolute margins are small. `Z0` has the best MAE. The three seeds share one 251-day test period, so seed spread is not a cross-market uncertainty interval. The mechanical zero directional accuracy exported for `Z0` is not an informative directional benchmark.
"""),
    code("""
from IPython.display import Image
final_dir = Path("output/final")
subprocess.run([sys.executable, "-m", "src.tools.verify_final_outputs"], check=True)
display(pd.read_csv(final_dir / "forecast_summary.csv"))
display(pd.read_csv(final_dir / "forecast_paired_fuzzy2.csv"))
display(Image(filename=str(final_dir / "forecast_rmse.png")))
display(Image(filename=str(final_dir / "prediction_vs_actual_seed11.png")))
"""),
    md("""
### Why trading almost never occurred

The primary policy enters at the next observed SRT open only when the return prediction exceeds zero, with 10 bps cost per side. `Fuzzy2` predicts a narrow range around zero: seed 11 is positive on six of 251 target days; seeds 23 and 37 are never positive. The latter two policies remain in cash. The development-selected secondary thresholds make **all 12** final model-seed policies stay in cash; that is a sensitivity result, not successful trading. The single missing 2025-10-24 open uses the frozen no-trade-and-carry rule.
"""),
    code("""
display(pd.read_csv(final_dir / "fuzzy_prediction_distribution.csv"))
display(pd.read_csv(final_dir / "fuzzy_position_summary.csv"))
display(Image(filename=str(final_dir / "fuzzy_prediction_distribution.png")))
display(Image(filename=str(final_dir / "fuzzy_positions_by_seed.png")))
primary_wealth = pd.read_csv(final_dir / "wealth_comparison.csv")
display(primary_wealth.query("wealth_mode == 'price_only' and cost_bps_per_side == 10"))
display(pd.read_csv(final_dir / "wealth_selected_threshold_10bps.csv")[[
    "model_id", "seed", "threshold", "terminal_wealth_multiple", "one_way_trade_count"]])
display(Image(filename=str(final_dir / "wealth_paths_price_only_10bps.png")))
"""),
    md("""
### Rule behavior and local evidence

The fuzzy case/rule tables satisfy the weighted-mixture identity. Final maximum rule weights average only 0.511–0.563, with no case above 0.8; one rule has no cases above its **training-defined** high-activation cut in one seed. Rule contributions are inspectable, but rule index 1 or 2 does not have a stable bull/bear meaning. The examples below were chosen *after* evaluation for explanation only (largest absolute forecast, largest absolute error, strongest weight); they were not used to select models or policies.
"""),
    code("""
final_predictions = pd.read_csv(final_dir / "predictions.csv")
final_cases = pd.read_csv(final_dir / "cases.csv")
final_rules = pd.read_csv(final_dir / "rules.csv")
data.validate_fuzzy_diagnostics(final_predictions, final_cases, final_rules)
display(pd.read_csv(final_dir / "fuzzy_local_checks.csv"))
display(pd.read_csv(final_dir / "fuzzy_case_examples_seed11.csv"))
display(Image(filename=str(final_dir / "fuzzy_rules_contributions_seed11.png")))
"""),
    md("""
### Training choice, input sources and rule count

The curves below show **inner development stopping** for seed 11, not final-test fitting. The final epochs (1/6/3/4) are the predeclared nine-run medians for `LSTM32`/`LSTM3`/`Fuzzy2`/`Gate2`. Early stopping may reflect weak signal and noisy short history; it does not by itself prove a training bug or sufficient convergence. In development, adding SORA or STI changed `Fuzzy2` RMSE only slightly. The seed-11 K=1/2/3 diagnostic favored K=2 over K=3, but it did not support a broad rule-count claim.
"""),
    code("""
display(pd.read_csv(development_dir / "fuzzy2_source_ablation.csv"))
display(pd.read_csv(development_dir / "rule_count_sensitivity_seed11.csv")[[
    "model_id", "effective_k", "return_rmse"]])
display(Image(filename=str(final_dir / "development_inner_stopping.png")))
"""),
    md("""
### Conclusion and limitations

The simplified LSTM→3D→two-rule fuzzy mixture achieved a **small, metric-dependent** final RMSE advantage over the ordinary LSTM and unchanged-price reference. It did not improve MAE over the reference, did not beat cash after trading costs, and did not produce clear market-regime specialists. Its near-zero, mostly negative predictions explain much of the trading result. The executable next-open return also differs from the next-close forecast target; the 248-date horizon check is in `output/final/three_return_horizon_correlations.csv`.

This test covers one ETF and one future period. The 2025-10-24 opening quote is unavailable, so trading paths are constrained-data simulations. The full issuer-verified table has nine distribution events; **two** fall inside the final holdout, and price-only and gross-distribution wealth are reported separately. Gross receipts are not investor-specific after-tax returns. This frozen test cannot be used to tune another model. Future research should test local objectives that prevent empty rules, the stability of learned rule semantics across refits, and concept drift on new periods or instruments. A building-load application and its chronological test design are specified in the literature section above.
"""),
])

nbf.validate(notebook)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--no-execute", action="store_true", help="Write source cells without saved outputs")
args = parser.parse_args()
if not args.no_execute:
    notebook = NotebookClient(notebook, timeout=600, kernel_name="python3",
                              resources={"metadata": {"path": str(ROOT)}}).execute()
    nbf.validate(notebook)
nbf.write(notebook, OUTPUT)
print(f"{OUTPUT} ({'source only' if args.no_execute else 'executed with saved outputs'})")
