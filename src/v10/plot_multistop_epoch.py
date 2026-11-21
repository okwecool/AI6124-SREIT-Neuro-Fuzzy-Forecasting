"""Plot saved V10 forward-block curves and outer-fold RMSE changes."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/v10/multistop_epoch"


def render() -> Path:
    curves = pd.read_csv(OUT / "epoch_curves.csv")
    selection = pd.read_csv(OUT / "selected_epochs.csv")
    by_fold = pd.read_csv(OUT / "comparison_by_fold.csv")
    example = curves.loc[curves.model_id.eq("Fuzzy2") & curves.fold.eq("validation_1") &
                         curves.seed.eq(11) & curves.epoch.le(60)]
    chosen = selection.loc[selection.model_id.eq("Fuzzy2") &
                           selection.fold.eq("validation_1") & selection.seed.eq(11)].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    colors = {"A": "#176789", "B": "#b15a26"}
    for block, data in example.groupby("block"):
        axes[0].plot(data.epoch, data.stop_to_zero_mse, label=f"Forward stop {block}",
                     color=colors[block], linewidth=2)
    axes[0].axvline(chosen.v1_selected_epoch, color="#53626f", linestyle=":",
                    label=f"V1 selected: {int(chosen.v1_selected_epoch)}")
    axes[0].axvline(chosen.selected_epoch, color="#166b54", linestyle="--",
                    label=f"V10 selected: {int(chosen.selected_epoch)}")
    axes[0].set_title("Fuzzy2 example: fold 1, seed 11")
    axes[0].set_xlabel("Training epoch")
    axes[0].set_ylabel("Stop MSE / zero-return MSE")
    axes[0].legend(frameon=False, fontsize=9)
    x = np.arange(3)
    for offset, old, new, color in ((-0.24, "Fuzzy2", "V10Fuzzy2", "#176789"),
                                    (0.0, "Gate2", "V10Gate2", "#b15a26"),
                                    (0.24, "LSTM3", "V10LSTM3", "#55785b")):
        original = by_fold.loc[by_fold.model_id.eq(old)].sort_values("fold").rmse.to_numpy()
        updated = by_fold.loc[by_fold.model_id.eq(new)].sort_values("fold").rmse.to_numpy()
        axes[1].bar(x + offset, 10000 * (updated - original), width=0.23,
                    label=old, color=color)
    axes[1].axhline(0, color="#53626f", linewidth=0.9)
    axes[1].set_title("Outer RMSE change: two stops vs V1")
    axes[1].set_ylabel("RMSE difference (basis points; lower is better)")
    axes[1].set_xticks(x, ["Fold 1", "Fold 2", "Fold 3"])
    axes[1].legend(frameon=False, fontsize=9)
    for axis in axes:
        axis.grid(axis="y", alpha=0.16)
        axis.spines[["top", "right"]].set_visible(False)
    path = OUT / "two_stop_epoch_diagnostic.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(render())
