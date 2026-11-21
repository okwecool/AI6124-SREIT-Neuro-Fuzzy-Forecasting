"""Publication-style summary of saved development results; no fitting."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/v4/neural_preopen"


def run():
    forecasts = pd.read_csv(OUT / "comparison_pooled.csv").set_index("arm")
    wealth = pd.read_csv(OUT / "portfolio_summary.csv")
    seed_wealth = pd.read_csv(OUT / "portfolio_by_seed.csv")
    arms = ["DivSchedule", "preopen/latent/gaussian", "preopen/direct_us/gaussian", "preopen_us"]
    labels = [
        "Announced distributions only",
        "LSTM + 3D fuzzy experts",
        "Fuzzy experts + direct US inputs",
        "Two-US-input Ridge",
    ]
    colors = ["#7b8794", "#3666a5", "#459788", "#c88727"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), layout="constrained")
    for i, (arm, label, color) in enumerate(zip(arms, labels, colors)):
        value = forecasts.loc[arm, "return_rmse"]
        axes[0].scatter(value, i, color=color, s=75, zorder=3)
        axes[0].text(value + 0.000012, i, f"{value:.6f}", va="center", fontsize=9)
    axes[0].axvline(forecasts.loc["DivSchedule", "return_rmse"], color=colors[0], ls="--", lw=1)
    axes[0].set(
        yticks=np.arange(4),
        yticklabels=labels,
        xlim=(0.00870, 0.00932),
        ylim=(3.6, -0.6),
        xlabel="Raw next-close return RMSE (lower is better)",
        title="Forecast gains: information versus model",
    )
    axes[0].ticklabel_format(axis="x", style="plain", useOffset=False)
    axes[0].tick_params(axis="x", labelrotation=20)
    entries = [
        ("preopen/latent/gaussian", "always_cash", "Cash"),
        ("preopen/latent/gaussian", "always_long", "Always long"),
        ("preopen/latent/gaussian", "momentum_long_cash", "Momentum"),
        ("preopen/latent/linear", "signal_long_cash", "3D linear head"),
        ("preopen/latent/gate", "signal_long_cash", "3D ordinary gate"),
        ("preopen/latent/gaussian", "signal_long_cash", "3D fuzzy experts"),
        ("preopen/direct_us/gaussian", "signal_long_cash", "Direct-US fuzzy experts"),
        ("preopen_us", "signal_long_cash", "Two-US-input Ridge"),
    ]
    for i, (arm, strategy, label) in enumerate(entries):
        selected = wealth.loc[
            wealth.arm.eq(arm)
            & wealth.strategy.eq(strategy)
            & wealth.wealth_mode.eq("gross_distribution")
        ]
        value = selected.three_fold_compounded_wealth.item()
        seed = seed_wealth.loc[
            seed_wealth.arm.eq(arm)
            & seed_wealth.strategy.eq(strategy)
            & seed_wealth.wealth_mode.eq("gross_distribution")
        ]
        axes[1].scatter(
            seed.three_fold_compounded_wealth, np.full(len(seed), i), color="#b7c7d7", s=25
        )
        axes[1].scatter(
            value, i, color="#c88727" if arm == "preopen_us" else "#224b72", marker="D", s=50
        )
        axes[1].text(value + 0.006, i - 0.2, f"{value:.3f}", va="center", fontsize=9)
    axes[1].axvline(1, color="#7b8794", ls="--", lw=1)
    axes[1].set(
        yticks=np.arange(len(entries)),
        yticklabels=[e[2] for e in entries],
        ylim=(len(entries) - 0.3, -0.7),
        xlim=(0.98, 1.25),
        xlabel="Three-fold compounded terminal wealth",
        title="Fixed policy: 10 bps/side, gross distributions",
    )
    for ax in axes:
        ax.grid(axis="x", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Development exploration only | 378 dates | unchanged SRT next-close target\n"
        "New information at next-session 08:00 Singapore time; this changes forecast lead time",
        fontsize=12,
    )
    axes[1].text(
        0,
        -0.19,
        "Diamonds: seed means; pale dots: individual seeds (not confidence intervals).",
        transform=axes[1].transAxes,
        fontsize=8,
    )
    fig.savefig(OUT / "development_summary.png", dpi=180)
    fig.savefig(OUT / "development_summary.pdf")
    plt.close(fig)


if __name__ == "__main__":
    run()
