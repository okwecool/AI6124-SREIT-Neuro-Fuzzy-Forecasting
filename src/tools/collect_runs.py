"""Validate and combine complete development runs into reviewable CSV outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.project_eval import Dataset


def collect(run_root: Path, output_dir: Path, *, cost_bps: int = 10,
            wealth_mode: str = "price_only") -> dict:
    manifests = sorted(Path(run_root).rglob("manifest.json"))
    if not manifests:
        raise ValueError(f"No run manifests under {run_root}")
    data = Dataset()
    predictions, cases, rules = [], [], []
    dataset_hashes, fold_hashes = set(), set()
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["status"] != "development":
            raise ValueError(f"Not a development run: {path}")
        dataset_hashes.add(manifest["dataset_sha256"])
        fold_hashes.add(manifest["fold_assignments_sha256"])
        run_dir = path.parent
        prediction = pd.read_csv(run_dir / "predictions.csv")
        if len(prediction) != manifest["outer_validation_rows"]:
            raise ValueError(f"Prediction row count disagrees with manifest: {path}")
        identity = ("model_id", "source_group", "rule_count", "fold", "seed")
        if any(not prediction[column].eq(manifest[column]).all() for column in identity):
            raise ValueError(f"Prediction run key disagrees with manifest: {path}")
        data.validate_predictions(prediction)
        predictions.append(prediction)
        if manifest["rule_count"]:
            case = pd.read_csv(run_dir / "cases.csv")
            rule = pd.read_csv(run_dir / "rules.csv")
            data.validate_fuzzy_diagnostics(prediction, case, rule)
            cases.append(case)
            rules.append(rule)
    if len(dataset_hashes) != 1 or len(fold_hashes) != 1:
        raise ValueError("Cannot mix different dataset or fold-assignment snapshots")
    prediction_table = pd.concat(predictions, ignore_index=True)
    data.validate_predictions(prediction_table)
    case_table = pd.concat(cases, ignore_index=True) if cases else None
    rule_table = pd.concat(rules, ignore_index=True) if rules else None
    if case_table is not None:
        data.validate_fuzzy_diagnostics(prediction_table, case_table, rule_table)
    scores = data.forecast_scores(prediction_table)
    ledger, wealth = data.portfolio_ledger(prediction_table, cost_bps=cost_bps,
                                           wealth_mode=wealth_mode)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_table.to_csv(output_dir / "predictions.csv", index=False)
    scores.to_csv(output_dir / "forecast_scores.csv", index=False)
    ledger.to_csv(output_dir / f"wealth_ledger_{wealth_mode}_{cost_bps}bps.csv", index=False)
    wealth.to_csv(output_dir / f"wealth_summary_{wealth_mode}_{cost_bps}bps.csv", index=False)
    if case_table is not None:
        case_table.to_csv(output_dir / "cases.csv", index=False)
        rule_table.to_csv(output_dir / "rules.csv", index=False)
    # Review plot: one declared source group, first development fold and seed.
    plot_fold = sorted(ledger.fold.unique())[0]
    plot_source = "all" if "all" in set(ledger.source_group) else sorted(ledger.source_group.unique())[0]
    plot_seed = int(sorted(ledger.seed.unique())[0])
    plotted = ledger.loc[(ledger.fold == plot_fold) &
                         (ledger.source_group == plot_source) &
                         (ledger.seed == plot_seed)]
    fig, ax = plt.subplots(figsize=(10, 5))
    for model_id in sorted(plotted.model_id.unique()):
        path = plotted.loc[(plotted.model_id == model_id) &
                           (plotted.strategy == "signal_long_cash")]
        ax.plot(path.date, path.wealth, label=f"{model_id} signal")
    control_id = sorted(plotted.model_id.unique())[0]
    for strategy in ("momentum_long_cash", "always_long", "always_cash"):
        path = plotted.loc[(plotted.model_id == control_id) & (plotted.strategy == strategy)]
        ax.plot(path.date, path.wealth, label=strategy, linestyle="--")
    ax.set_title(f"{plot_fold}, {plot_source}, seed {plot_seed}: {wealth_mode}, {cost_bps} bps")
    ax.set_ylabel("Wealth / initial SGD")
    ax.set_xlabel("Observed SRT date")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    figure_path = output_dir / f"wealth_paths_{wealth_mode}_{cost_bps}bps.png"
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    result = {"runs": len(manifests), "prediction_rows": len(prediction_table),
              "fuzzy_cases": len(case_table) if case_table is not None else 0,
              "fuzzy_rule_rows": len(rule_table) if rule_table is not None else 0,
              "ledger_rows": len(ledger), "wealth_mode": wealth_mode,
              "cost_bps": cost_bps, "output_dir": str(output_dir),
              "review_figure": str(figure_path)}
    (output_dir / f"collection_{wealth_mode}_{cost_bps}bps.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cost-bps", type=int, default=10)
    parser.add_argument("--wealth-mode", choices=("price_only", "gross_distribution"),
                        default="price_only")
    args = parser.parse_args()
    print(json.dumps(collect(args.runs, args.out, cost_bps=args.cost_bps,
                             wealth_mode=args.wealth_mode), indent=2))


if __name__ == "__main__":
    main()
