"""Build the Group 1 course Notebook scaffold from the reviewed handoff files."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs" / "AI6124_Group1_Project_Starter.ipynb"


def md(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


literature = (ROOT / "docs" / "literature_review_draft_en.md").read_text(encoding="utf-8")
literature = literature.replace("# Literature Review Draft — LSTM-Encoded Fuzzy Local Experts", "## Literature review")
literature = literature.replace(
    "**Status:** An English starting section for the final Jupyter Notebook, 30 October 2026. Verify the final bibliography format and add discussion of the team's actual results before submission.",
    "This section is a pretraining draft. Add the team's measured results and final discussion before submission.",
)

notebook = nbf.v4.new_notebook()
notebook.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
notebook.cells = [
    md("""
# AI6124 Group 1 — LSTM-Encoded Fuzzy Local Experts for SRT

**Zou Runze (Ryan) · Ngeow Shan Yong Destin**  
**Working Notebook scaffold, 30 October 2026.**

This Notebook starts the final course deliverable. It runs the frozen data audit and a no-change baseline. Trained-model predictions, interpretation, portfolio comparison, and conclusions still have to be added. The fixed protocol is in `docs/experiment_protocol_lstm_fuzzy_local_experts.md`.

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
## Models and benchmark

- `Z0`: predict zero next-day return (unchanged closing price).
- `LSTM32`: one-layer, 32-hidden-unit LSTM with a scalar return head.
- `LSTM3`: the same encoder with a three-dimensional `tanh` projection and scalar linear head.
- `Fuzzy2`: the same encoder and projection, then two diagonal Gaussian rules and two affine return experts.

The fuzzy system receives only the three learned latent values, not a flattened 20 × 6 input. `Fuzzy2` predicts a rule-weighted return. The exact initialization, optimizer, stopping policy, seeds and ablations are fixed in the protocol. Destin will insert model and training cells here. The encoder, fuzzy weights and experts must be trained only on each fold's training rows; the outer validation rows are used only for evaluation.
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

The no-change predictor needs no fitting and is evaluated only on the development folds here. The final test remains reserved for the frozen set of trained models. Directional accuracy excludes zero true returns; the majority-direction reference is learned from each fold's training labels.
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

Destin's training script should write `output/model_predictions_validation.csv` with one row per outer-validation forecast and these columns:

`model_id,source_group,fold,seed,window_id,y_pred`

`y_pred` must be in **raw log-return units**, after inverse target scaling. The evaluation module rejects wrong fold IDs, duplicate rows and incomplete folds. Models should also save `z`, rule weights and expert outputs in a separate diagnostic file keyed by the same identifiers. Do not infer that a rule is a particular market regime from its numeric label.
"""),
    code("""
prediction_file = Path("output/model_predictions_validation.csv")
if prediction_file.exists():
    model_predictions = pd.read_csv(prediction_file)
    model_scores = data.forecast_scores(model_predictions)
    model_pooled = data.pooled_validation_scores(model_predictions)
    display(model_scores.sort_values(["model_id", "source_group", "fold", "seed"]))
    display(model_pooled.sort_values("return_rmse"))
    if {"Fuzzy2", "LSTM32"} <= set(model_predictions.model_id):
        display(data.paired_validation_rmse_difference(model_predictions))
else:
    print(f"Awaiting trained-model predictions: {prediction_file}")
"""),
    code("""
if prediction_file.exists():
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
            ["model_id", "source_group"], as_index=False
        ).return_rmse.mean().sort_values("return_rmse")
        labels = overview.model_id + " / " + overview.source_group
        axes[1].barh(labels, overview.return_rmse, color="#0c6b76")
        axes[1].set_xlabel("Mean fold return RMSE (descriptive; use pooled scores for ranking)")
        plt.show()
    else:
        print("Awaiting full-feature Fuzzy2, seed 11, for the forecast plot.")
"""),
    md("""
## Fuzzy-rule diagnostics

If `output/fuzzy_diagnostics_validation.csv` exists, it should use the same `fold,seed,window_id` keys as the prediction file and contain `z1,z2,z3,alpha1,alpha2,expert1_raw,expert2_raw`. The expert columns are in raw-return units. Rule-number switching across seeds must be considered before attaching market labels.
"""),
    code("""
diagnostic_file = Path("output/fuzzy_diagnostics_validation.csv")
if diagnostic_file.exists():
    diagnostics = pd.read_csv(diagnostic_file)
    required = {"fold", "seed", "window_id", "z1", "z2", "z3", "alpha1", "alpha2", "expert1_raw", "expert2_raw"}
    assert required <= set(diagnostics.columns)
    assert np.allclose(diagnostics.alpha1 + diagnostics.alpha2, 1, atol=1e-5)
    display(diagnostics[["alpha1", "alpha2", "expert1_raw", "expert2_raw"]].describe())
else:
    print(f"Awaiting fuzzy diagnostics: {diagnostic_file}")
"""),
    md("""
## Portfolio application — exploratory price-only evaluation

A signal formed after close `t` is first executable at the next SRT open. The portfolio code measures the following open-to-open interval, so its realized return has a different horizon from the close-to-close forecast label. Compare signal long/cash, always-long and always-cash on identical eligible intervals. The one-way cost assumptions are 0, 10 and 20 bps. Intervals requiring the missing 2025-10-24 open are omitted for every strategy. Cash distributions are not yet fully verified, so price-only figures must **not** be called ETF total returns.
"""),
    code("""
if prediction_file.exists():
    portfolio_results = pd.concat(
        [data.portfolio_scores(model_predictions, cost_bps=cost) for cost in (0, 10, 20)],
        ignore_index=True,
    )
    display(portfolio_results)
else:
    print("Portfolio comparison will run after the trained predictions are added.")
"""),
    md("""
## Discussion and conclusion — complete after training

1. **Forecast comparison.** Report each development fold, three seeds, pooled validation scores and the once-opened final holdout. Explain the size and variability of `Fuzzy2 − LSTM32` and `Fuzzy2 − LSTM3` differences.
2. **Sources and parameters.** Explain SRT-only versus +SORA versus +STI, and the one/two/three-rule diagnostic. Describe when additional inputs or rules fail to help.
3. **Interpretability.** Plot both rule weights and expert predictions; check their distribution and relationship with *past-only* volatility and momentum. State if rules collapse or swap labels across seeds.
4. **Trading.** Compare price-only next-open performance at three costs with always-long/cash, and acknowledge the forecast/trading-horizon mismatch and unverified distributions.
5. **Limitations and future research.** Address short ETF history, possible overfitting, learned latent features without fixed economic meaning, one missing open, distribution verification, and possible future tests on a longer REIT series.

This section is deliberately left without numerical conclusions until all results have been produced.
"""),
]

nbf.validate(notebook)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
