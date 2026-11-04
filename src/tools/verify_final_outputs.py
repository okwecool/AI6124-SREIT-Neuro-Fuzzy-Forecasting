"""Read-only audit of the saved final predictions, metrics and causal ledgers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.project_eval import Dataset, RUN_KEY
from src.train import _sha256
from src.tools.run_final_holdout import CONFIG, FINAL_MODELS, ROOT, SEEDS, preflight


def validate_completion_record(complete: dict) -> None:
    if complete.get("status") not in (
        "final_execution_complete", "final_analysis_complete_from_original_predictions"
    ):
        raise ValueError("Final execution is incomplete or has unexpected status")
    for field in ("prediction_file_sha256", "cases_file_sha256", "rules_file_sha256"):
        if not isinstance(complete.get(field), str) or len(complete[field]) != 64:
            raise ValueError(f"Final completion record lacks a valid {field}")


def verify(folder: Path = ROOT / "output/final") -> dict:
    frozen, data = preflight(CONFIG)
    complete = json.loads((folder / "execution_complete.json").read_text(encoding="utf-8"))
    validate_completion_record(complete)
    for name, key in (("predictions.csv", "prediction_file_sha256"),
                      ("cases.csv", "cases_file_sha256"),
                      ("rules.csv", "rules_file_sha256")):
        if _sha256(folder / name) != complete[key]:
            raise ValueError(f"Final export changed after the recovery audit: {name}")
    predictions = pd.read_csv(folder / "predictions.csv")
    cases = pd.read_csv(folder / "cases.csv")
    rules = pd.read_csv(folder / "rules.csv")
    if len(predictions) != 3263 or len(cases) != 753 or len(rules) != 1506:
        raise ValueError("Final row counts differ from the frozen 12-fit plan")
    data.validate_predictions(predictions)
    data.validate_fuzzy_diagnostics(predictions, cases, rules)
    expected = {(model, seed) for model in FINAL_MODELS for seed in SEEDS}
    observed = set(zip(predictions.loc[predictions.model_id.ne("Z0"), "model_id"],
                       predictions.loc[predictions.model_id.ne("Z0"), "seed"]))
    if observed != expected:
        raise ValueError("Final model/seed matrix is incomplete")
    recomputed = data.forecast_scores(predictions).sort_values(RUN_KEY).reset_index(drop=True)
    saved = pd.read_csv(folder / "forecast_by_seed.csv").sort_values(RUN_KEY).reset_index(drop=True)
    if not recomputed[RUN_KEY].equals(saved[RUN_KEY]) or not np.allclose(
        recomputed.return_rmse, saved.return_rmse, atol=1e-12, rtol=0
    ):
        raise ValueError("Saved forecast scores disagree with original predictions")
    ledger_checks = 0
    for mode, cost in (("price_only", 0), ("price_only", 10),
                       ("price_only", 20), ("gross_distribution", 10)):
        ledger_path = folder / f"wealth_ledger_{mode}_{cost}bps.csv"
        if not ledger_path.exists():
            continue  # Full event ledgers are local reproducible files, ignored by Git.
        ledger = pd.read_csv(ledger_path)
        summary = pd.read_csv(folder / f"wealth_summary_{mode}_{cost}bps.csv")
        keys = RUN_KEY + ["strategy"]
        last = ledger.groupby(keys, as_index=False).tail(1)
        joined = last.merge(summary, on=keys, validate="one_to_one")
        if len(joined) != len(summary) or not np.allclose(
            joined.wealth, joined.terminal_wealth_multiple, atol=1e-10, rtol=0
        ):
            raise ValueError(f"Ledger endpoints differ from wealth summaries: {mode}/{cost}")
        missing = ledger.loc[ledger.date.eq("2025-10-24")]
        if len(missing) != len(summary) or missing.observed_open.astype(str).str.lower().ne("false").any() or not missing.action.eq("missing_open_carry").all():
            raise ValueError("Missing-open no-trade-and-carry rule was violated")
        ledger_checks += 1
    return {"status": "pass", "frozen_refit_epochs": frozen["final_refit_epochs_by_model"],
            "prediction_rows": len(predictions), "fuzzy_cases": len(cases),
            "fuzzy_rule_rows": len(rules), "forecast_runs": len(saved),
            "local_ledger_variants_checked": ledger_checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=ROOT / "output/final")
    args = parser.parse_args()
    print(json.dumps(verify(args.folder), indent=2))


if __name__ == "__main__":
    main()
