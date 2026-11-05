"""Visualize saved V5 forecasts and ablations without model fitting."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.v5.shared_local import OUT


def run():
    score = pd.read_csv(OUT / "comparison_pooled.csv").set_index("arm")
    fold = pd.read_csv(OUT / "comparison_by_fold.csv").groupby(["arm", "fold"]).rmse.mean()
    arms = [
        "Z0",
        "V1_Fuzzy2",
        "gaussian_independent",
        "gaussian_common_matched",
        "gaussian_shared_local",
        "linear",
        "gate_shared_local",
    ]
    labels = [
        "Zero-return reference",
        "Original V1 Fuzzy2",
        "Independent experts",
        "Match common penalty",
        "Shared + shrunk local (primary)",
        "Matched linear head",
        "Matched ordinary gate",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), layout="constrained")
    colors = ["#777777", "#687fa2", "#687fa2", "#687fa2", "#167d8d", "#687fa2", "#687fa2"]
    for i, (arm, color) in enumerate(zip(arms, colors)):
        value = score.loc[arm, "rmse"]
        axes[0].scatter(value, i, color=color, s=65)
        axes[0].text(value + 0.000002, i, f"{value:.6f}", va="center", fontsize=9)
    axes[0].axvline(score.loc["Z0", "rmse"], color="#777777", ls="--", lw=1)
    axes[0].set(
        yticks=range(len(arms)),
        yticklabels=labels,
        ylim=(6.7, -0.7),
        xlim=(0.009420, 0.009530),
        xlabel="Seed-mean pooled RMSE (lower is better)",
        title="Small gain over V1; still worse than zero",
    )
    axes[0].ticklabel_format(axis="x", style="plain", useOffset=False)
    axes[0].tick_params(axis="x", labelrotation=30)
    x = np.arange(3)
    for arm, label, offset, color in [
        ("Z0", "Zero return", -0.22, "#777777"),
        ("V1_Fuzzy2", "Original Fuzzy2", 0.0, "#687fa2"),
        ("gaussian_shared_local", "New shared/local", 0.22, "#167d8d"),
    ]:
        values = [
            100 * (fold.loc[(arm, f"validation_{j}")] / fold.loc[("Z0", f"validation_{j}")] - 1)
            for j in (1, 2, 3)
        ]
        axes[1].bar(x + offset, values, 0.21, label=label, color=color)
    axes[1].axhline(0, color="#777777", lw=1)
    axes[1].set(
        xticks=x,
        xticklabels=["Fold 1", "Fold 2", "Fold 3"],
        ylabel="RMSE change versus zero return (%)",
        title="Positive values mean worse error",
    )
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].grid(axis="x", alpha=0.2)
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle(
        "V5 model-only exploration | same original 20x6 inputs, forecast time and raw target\n"
        "378 development dates | 3 seeds | primary uses zero fallback in 6 of 9 runs",
        fontsize=12,
    )
    fig.savefig(OUT / "model_only_summary.png", dpi=180)
    fig.savefig(OUT / "model_only_summary.pdf")
    plt.close(fig)


if __name__ == "__main__":
    run()
