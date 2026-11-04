"""Render final interpretation figures from frozen CSVs and development manifests."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.project_eval import Dataset


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/final"


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)


def main() -> None:
    data = Dataset(ROOT)
    predictions = pd.read_csv(OUT / "predictions.csv")
    joined = data.validate_predictions(predictions)
    fuzzy = joined.loc[joined.model_id.eq("Fuzzy2")].copy()
    stats = fuzzy.groupby("seed", as_index=False).agg(
        predicted_mean=("y_pred", "mean"), predicted_sd=("y_pred", "std"),
        predicted_min=("y_pred", "min"), predicted_max=("y_pred", "max"),
        positive_count=("y_pred", lambda series: int((series > 0).sum())),
        n=("y_pred", "size"))
    stats["positive_fraction"] = stats.positive_count / stats.n
    stats.to_csv(OUT / "fuzzy_prediction_distribution.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    for seed, group in fuzzy.groupby("seed"):
        ax.hist(group.y_pred, bins=35, alpha=.45, label=f"seed {seed}")
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Predicted next-close log return")
    ax.set_ylabel("Target days")
    ax.set_title("Fuzzy2 final predictions cluster near zero")
    ax.legend()
    save(fig, "fuzzy_prediction_distribution.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    sample = fuzzy.loc[fuzzy.seed.eq(11)].sort_values("target_date")
    ax.plot(sample.target_date, sample.y_true, label="Actual return", linewidth=.8)
    ax.plot(sample.target_date, sample.y_pred, label="Fuzzy2 prediction", linewidth=1.2)
    ax.axhline(0, color="gray", linewidth=.7)
    ax.set_ylabel("Next-close log return")
    ax.set_title("Final holdout: actual and predicted returns, seed 11")
    ax.legend()
    save(fig, "prediction_vs_actual_seed11.png")

    only_fuzzy = predictions.loc[predictions.model_id.eq("Fuzzy2")]
    ledger, wealth = data.portfolio_ledger(only_fuzzy, cost_bps=10)
    position = ledger.loc[ledger.strategy.eq("signal_long_cash") & ledger.observed_open].copy()
    position["date"] = pd.to_datetime(position.date)
    fig, axes = plt.subplots(3, 1, figsize=(10, 5), sharex=True)
    for ax, seed in zip(axes, (11, 23, 37)):
        group = position.loc[position.seed.eq(seed)].sort_values("date")
        ax.step(group.date, group.actual_position, where="post")
        ax.set_ylim(-.1, 1.15)
        ax.set_ylabel(f"seed {seed}")
    axes[0].set_title("Fuzzy2 position at observed opens; threshold 0, 10 bps")
    axes[-1].set_xlabel("Date")
    save(fig, "fuzzy_positions_by_seed.png")
    position.groupby("seed", as_index=False).agg(
        observed_opens=("date", "size"), invested_opens=("actual_position", "sum")
    ).merge(wealth.loc[wealth.strategy.eq("signal_long_cash"),
                       ["seed", "terminal_wealth_multiple", "one_way_trade_count"]],
            on="seed", validate="one_to_one").to_csv(OUT / "fuzzy_position_summary.csv", index=False)

    cases = pd.read_csv(OUT / "cases.csv")
    rules = pd.read_csv(OUT / "rules.csv")
    case = cases.loc[cases.seed.eq(11)].merge(
        sample[["window_id", "target_date", "y_true", "y_pred"]],
        on="window_id", validate="one_to_one").sort_values("target_date")
    rule = rules.loc[rules.seed.eq(11)].merge(
        case[["window_id", "target_date"]], on="window_id", validate="many_to_one")
    rule["contribution"] = rule.alpha * rule.expert_raw
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for rule_id, group in rule.groupby("rule_id"):
        group = group.sort_values("target_date")
        axes[0].plot(group.target_date, group.alpha, label=f"rule {rule_id}")
        axes[1].plot(group.target_date, group.contribution, label=f"rule {rule_id} contribution")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Normalized weight")
    axes[0].legend()
    axes[1].plot(case.target_date, case.y_pred, color="black", linestyle="--",
                 linewidth=1, label="mixture")
    axes[1].set_ylabel("Raw log-return contribution")
    axes[1].set_xlabel("Target date")
    axes[1].legend()
    axes[0].set_title("Fuzzy2 rule weights and expert contributions, seed 11")
    save(fig, "fuzzy_rules_contributions_seed11.png")

    wide = rule.pivot(index="window_id", columns="rule_id",
                      values=["alpha", "expert_raw", "contribution"])
    wide.columns = [f"{name}_{rule_id}" for name, rule_id in wide.columns]
    examples = case.join(wide, on="window_id")
    examples["absolute_error"] = (examples.y_pred - examples.y_true).abs()
    examples["max_alpha"] = examples[["alpha_1", "alpha_2"]].max(axis=1)
    chosen = {
        "largest_abs_prediction": examples.y_pred.abs().idxmax(),
        "largest_abs_error": examples.absolute_error.idxmax(),
        "strongest_rule_weight": examples.max_alpha.idxmax(),
    }
    selected = []
    for reason, index in chosen.items():
        record = examples.loc[index, ["target_date", "window_id", "y_true", "y_pred",
                                      "absolute_error", "alpha_1", "alpha_2",
                                      "expert_raw_1", "expert_raw_2",
                                      "contribution_1", "contribution_2"]].to_dict()
        selected.append({"selection_reason_descriptive_only": reason, **record})
    pd.DataFrame(selected).to_csv(OUT / "fuzzy_case_examples_seed11.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=False)
    for ax, model in zip(axes.flat, ("LSTM32", "LSTM3", "Fuzzy2", "Gate2")):
        for fold in ("validation_1", "validation_2", "validation_3"):
            path = ROOT / "output/runs/development" / fold / "all" / model / "seed_11/manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            trace = pd.DataFrame(manifest["inner_trace"])
            ax.plot(trace.epoch, trace.stop_mse_scaled, linewidth=1, label=fold[-1])
            ax.axvline(manifest["best_epoch"], alpha=.2, linewidth=.8)
        ax.set_title(model)
        ax.set_xlabel("Inner-fit epoch")
        ax.set_ylabel("Stopping MSE, standardized")
        ax.legend(title="fold", fontsize=8)
    fig.suptitle("Development inner stopping, seed 11; final epochs use nine-run medians")
    save(fig, "development_inner_stopping.png")
    print(json.dumps({"status": "pass", "figures": 5, "case_examples": len(selected),
                      "prediction_stats": stats.to_dict(orient="records")}, indent=2))


if __name__ == "__main__":
    main()
