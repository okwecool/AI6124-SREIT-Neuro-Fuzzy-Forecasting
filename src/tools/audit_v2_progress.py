"""Read saved V2 predictions and explain their limits; never fit a model.

Run with --out in a new directory to preserve the review evidence. Hashes here
describe the files at review time, not a retroactive pre-experiment freeze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from src.project_eval import Dataset, VALIDATION_FOLDS


ROOT = Path(__file__).resolve().parents[2]


def audit() -> dict:
    data = Dataset(ROOT)
    fingerprints = {}

    def read_csv(relative: str) -> pd.DataFrame:
        path = ROOT / relative
        fingerprints[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return pd.read_csv(path)

    # Recompute RMSE/MAE directly from keyed predictions, independently of the
    # existing scoring functions. Average seed scores, not seed predictions.
    screens = {
        "h1": ("h1/predictions.csv", {}),
        "analysis": ("analysis/h3_predictions.csv", {"Fuzzy2": "Fuzzy2+local"}),
        "h1b_h2_screen": ("h1b_h2_screen/predictions.csv", {}),
        "announced_distribution_screen": ("announced_distribution_screen/predictions.csv", {}),
        "residual_fuzzy": ("residual_fuzzy/predictions.csv", {"Fuzzy2": "ResidualFuzzy2"}),
        "clr_screen": ("clr_screen/predictions.csv", {}),
        "trend_rate_screen": ("trend_rate_screen/predictions.csv", {}),
    }
    comparisons = {}
    prediction_rows = 0
    maximum_error = 0.0
    for folder, (prediction_file, rename) in screens.items():
        prediction = read_csv(f"output/v2/{prediction_file}")
        if set(prediction.fold) != set(VALIDATION_FOLDS):
            raise ValueError(f"Unexpected evaluation folds in {folder}")
        checked = data.validate_predictions(prediction)
        prediction_rows += len(checked)
        records = []
        for (model, seed), part in checked.groupby(["model_id", "seed"]):
            if len(part) != 378 or part.window_id.nunique() != 378:
                raise ValueError(f"Incomplete common evaluation calendar in {folder}")
            error = part.y_pred.to_numpy() - part.y_true.to_numpy()
            records.append({"model_id": rename.get(model, model), "seed": int(seed),
                            "rmse": float(np.sqrt(np.mean(error ** 2))),
                            "mae": float(np.mean(np.abs(error)))})
        actual = pd.DataFrame(records).groupby("model_id")[["rmse", "mae"]].mean()
        saved = read_csv(f"output/v2/{folder}/comparison_pooled.csv")
        saved = saved.rename(columns={"return_rmse": "rmse", "return_mae": "mae",
                                      "mean_pooled_rmse": "rmse", "mean_pooled_mae": "mae"})
        expected = saved.groupby("model_id")[["rmse", "mae"]].mean().loc[actual.index]
        difference = float(np.max(np.abs(actual.to_numpy() - expected.to_numpy())))
        if difference > 1e-12:
            raise ValueError(f"Saved score mismatch in {folder}: {difference}")
        maximum_error = max(maximum_error, difference)
        comparisons[folder] = actual.reset_index().to_dict(orient="records")

    fuzzy_runs = 0
    for folder, pattern in (("h3", "validation_*/all/Fuzzy2/seed_*"),
                            ("residual_fuzzy", "runs/validation_*/seed_*")):
        runs = sorted((ROOT / "output/v2" / folder).glob(pattern))
        if len(runs) != 9:
            raise ValueError(f"Expected nine complete runs in {folder}")
        for run in runs:
            tables = [read_csv((run / name).relative_to(ROOT).as_posix())
                      for name in ("predictions.csv", "cases.csv", "rules.csv")]
            data.validate_fuzzy_diagnostics(*tables)
            fuzzy_runs += 1

    schedule = read_csv("output/v2/announced_distribution_screen/predictions.csv")
    schedule = data.validate_predictions(schedule.loc[schedule.model_id.eq("DivSchedule")])
    events = schedule.loc[schedule.y_pred.ne(0)].copy()
    if len(events) != 3:
        raise ValueError("Expected three announced distribution adjustments")
    events["sse_gain_vs_zero"] = events.y_true ** 2 - (events.y_true - events.y_pred) ** 2
    event_records = [{"target_date": str(row.target_date.date()), "actual": float(row.y_true),
                      "prediction": float(row.y_pred), "sse_gain_vs_zero": float(row.sse_gain_vs_zero)}
                     for row in events.itertuples()]

    daily = data.daily.loc[data.daily.date.le("2025-09-30")].copy()
    gross_return = np.log((daily.srt_close + daily.srt_dividend_sgd_on_date) / daily.srt_close.shift(1))
    gross_vol = gross_return.rolling(20, min_periods=20).std(ddof=0)
    input_events = []
    for i in daily.index[daily.srt_dividend_sgd_on_date.gt(0)]:
        input_events.append({"date": str(daily.date.loc[i].date()),
                             "raw_return": float(daily.srt_log_return_raw.loc[i]),
                             "gross_return": float(gross_return.loc[i]),
                             "raw_vol20": float(daily.srt_volatility_20.loc[i]),
                             "gross_vol20": float(gross_vol.loc[i])})

    # Exact existing residual-label definition, restricted to development dates.
    index = data.index.loc[data.index.target_date.le("2025-09-30")]
    close = index.forecast_date.map(daily.set_index("date").srt_close).to_numpy()
    distribution = index.target_date.map(daily.set_index("date").srt_dividend_sgd_on_date).to_numpy()
    residual = data.y[index.window_id].astype(float) - np.log1p(-distribution / close)
    mean_floors = []
    for fold in VALIDATION_FOLDS:
        train_ids, eval_ids = data.ids(fold, "train"), data.ids(fold, "validation")
        mu = float(residual[train_ids].mean())
        actual = residual[eval_ids]
        mean_floors.append({"fold": fold, "train_residual_mean": mu,
                            "evaluation_residual_mean": float(actual.mean()),
                            "zero_residual_rmse": float(np.sqrt(np.mean(actual ** 2))),
                            "train_mean_residual_rmse": float(np.sqrt(np.mean((actual - mu) ** 2)))})

    notebook = json.loads((ROOT / "docs/AI6124_Group1_Final_Project.ipynb").read_text(encoding="utf-8"))
    notebook_text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    with ZipFile(ROOT / "output/AI6124_Group1_Final_Handoff_2026-11-04.zip") as archive:
        v2_members = [n for n in archive.namelist() if "/v2/" in n or "v2_" in n]
    return {"review_date": "2026-11-05", "scope": "saved-output audit; no new fits or final-period ranking",
            "prediction_rows_checked": prediction_rows, "fuzzy_runs_checked": fuzzy_runs,
            "max_recomputed_score_difference": maximum_error,
            "scores": comparisons, "distribution_event_gain": event_records,
            "historical_input_events": input_events, "unpenalized_intercept_limit": mean_floors,
            "initial_fuzzy_max_weight_bound": float(1 / (1 + np.exp(-0.5))),
            "delivery": {"notebook_mentions_DivSchedule": "DivSchedule" in notebook_text,
                         "old_handoff_v2_members": len(v2_members)},
            "review_time_input_sha256": fingerprints,
            "limitations": ["Seven one-day experiment exports checked; weekly-target metrics were not recomputed here.",
                            "Metric agreement and diagnostic identities do not prove historical data availability or generalization.",
                            "Review-time hashes are not original execution dependency hashes."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = audit()
    result = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as destination:
            destination.write(result + "\n")
        print(json.dumps({"status": "pass", "output": str(args.out),
                          "prediction_rows_checked": report["prediction_rows_checked"],
                          "fuzzy_runs_checked": report["fuzzy_runs_checked"],
                          "max_score_difference": report["max_recomputed_score_difference"]}, indent=2))
    else:
        print(result)
