"""Render the saved V9 regime reversal and forecast-error comparison."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/v9/signal_audit"


def render() -> Path:
    splits = pd.read_csv(OUT / "feature_median_splits.csv")
    splits = splits.loc[splits.feature.eq("srt_volatility_20d")].sort_values("fold")
    scores = pd.read_csv(OUT / "saved_model_by_regime_summary.csv")
    fuzzy = scores.loc[scores.model_id.eq("Fuzzy2"), ["fold", "state", "rmse"]]
    zero = scores.loc[scores.model_id.eq("Z0"), ["fold", "state", "rmse"]]
    paired = fuzzy.merge(zero, on=["fold", "state"], suffixes=("_fuzzy", "_zero"), validate="one_to_one")
    names = ["2024 Apr–Sep", "2024 Oct–2025 Mar", "2025 Apr–Sep"]
    x = np.arange(3)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    axes[0].axhline(0, color="#65717c", linewidth=0.9)
    axes[0].plot(x, 10000 * splits.train_high_minus_low.to_numpy(), "o-", color="#176789",
                 label="Training period", linewidth=2)
    axes[0].plot(x, 10000 * splits.eval_high_minus_low.to_numpy(), "s-", color="#b15a26",
                 label="Later validation", linewidth=2)
    axes[0].set_title("High minus low volatility: next-day return")
    axes[0].set_ylabel("Mean-return difference (basis points)")
    axes[0].legend(frameon=False)
    axes[1].axhline(0, color="#65717c", linewidth=0.9)
    for offset, state, color in ((-0.18, "high", "#176789"), (0.18, "low", "#b15a26")):
        values = paired.loc[paired.state.eq(state)].sort_values("fold")
        axes[1].bar(x + offset, 10000 * (values.rmse_fuzzy - values.rmse_zero).to_numpy(),
                    width=0.34, label=state.title() + " volatility", color=color)
    axes[1].set_title("Saved Fuzzy2 RMSE minus zero-return RMSE")
    axes[1].set_ylabel("RMSE difference (basis points; lower is better)")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.set_xticks(x, names, rotation=15, ha="right")
        axis.grid(axis="y", alpha=0.16)
        axis.spines[["top", "right"]].set_visible(False)
    path = OUT / "regime_signal_diagnostic.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path


if __name__ == "__main__":
    print(render())
