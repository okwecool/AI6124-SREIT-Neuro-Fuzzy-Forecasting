"""Evaluate a causal, SGX-announced distribution adjustment on V1 development dates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.project_eval import Dataset, PREDICTION_COLUMNS, VALIDATION_FOLDS


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs/v2_announced_distribution_contract_2026-11-04.json"
EVENTS = ROOT / "data/reference/srt_announced_distributions_dev.csv"
OUT = ROOT / "output/v2/announced_distribution_screen"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def announced_adjustments(data: Dataset, announcements: pd.DataFrame) -> pd.DataFrame:
    """One row per development forecast, using information announced before t."""
    expected = pd.concat([data.zero_predictions()], ignore_index=True)
    known = data.validate_predictions(expected)[["fold", "window_id", "forecast_date", "target_date", "close_t", "y_true"]]
    announcement = announcements.copy()
    announcement["announcement_date"] = pd.to_datetime(announcement.announcement_date)
    announcement["ex_date"] = pd.to_datetime(announcement.ex_date)
    if announcement.ex_date.duplicated().any() or (announcement.announcement_date >= announcement.ex_date).any():
        raise ValueError("Announcement dates must precede unique ex-dates")
    if (announcement.distribution_sgd <= 0).any() or not np.isfinite(announcement.distribution_sgd).all():
        raise ValueError("Distribution amounts must be finite and positive")
    frame = known.merge(announcement, left_on="target_date", right_on="ex_date", how="left", validate="many_to_one")
    event = frame.ex_date.notna()
    if not (frame.loc[event, "announcement_date"] < frame.loc[event, "forecast_date"]).all():
        raise ValueError("Distribution announcement unavailable at forecast time")
    if (frame.loc[event, "close_t"] <= frame.loc[event, "distribution_sgd"]).any():
        raise ValueError("Announced distribution exceeds current SRT close")
    if len(frame) != 378 or set(frame.fold) != set(VALIDATION_FOLDS):
        raise ValueError("The V1 development calendar changed")
    if int(event.sum()) != len(announcement):
        raise ValueError("An announced ex-date is not in the development evaluation calendar")
    # Check the independent SGX amount against the existing historical label audit.
    yahoo = data.daily[["date", "srt_dividend_sgd_on_date"]]
    matched = frame.loc[event, ["target_date", "distribution_sgd"]].merge(
        yahoo, left_on="target_date", right_on="date", validate="one_to_one")
    if not np.allclose(matched.distribution_sgd, matched.srt_dividend_sgd_on_date, atol=1e-6):
        raise ValueError("SGX announced amounts disagree with historical dividend audit")
    frame["adjustment"] = 0.0
    frame.loc[event, "adjustment"] = np.log1p(-frame.loc[event, "distribution_sgd"] / frame.loc[event, "close_t"])
    return frame


def run() -> dict:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refuse to overwrite results: {OUT}")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract["experiment_id"] != "V2-H8-announced-distribution-baseline-01":
        raise ValueError("Unexpected contract")
    data = Dataset(ROOT)
    announcement = pd.read_csv(EVENTS)
    adjustment = announced_adjustments(data, announcement)
    schedule = pd.DataFrame([
        ("DivSchedule", "all", 0, row.fold, 0, int(row.window_id), float(row.adjustment))
        for row in adjustment.itertuples()], columns=PREDICTION_COLUMNS)
    old = pd.read_csv(ROOT / "output/development/predictions.csv")
    old = old.loc[old.source_group.eq("all") & old.model_id.eq("Fuzzy2")]
    old = old.merge(adjustment[["fold", "window_id", "adjustment"]],
                    on=["fold", "window_id"], validate="many_to_one")
    hybrid = old.loc[:, PREDICTION_COLUMNS].copy()
    hybrid["model_id"] = "SchedulePlusFuzzy2"
    hybrid["rule_count"] = 0
    hybrid["y_pred"] += old.adjustment
    comparison = pd.concat([data.zero_predictions(), schedule,
                            old.loc[:, PREDICTION_COLUMNS], hybrid], ignore_index=True)
    data.validate_predictions(comparison)
    pooled = data.pooled_validation_scores(comparison).sort_values("return_rmse")
    by_fold = data.forecast_scores(comparison).sort_values(["model_id", "fold", "seed"])
    z0 = pooled.loc[pooled.model_id.eq("Z0")].iloc[0]
    selected = pooled.loc[pooled.model_id.eq("DivSchedule")].iloc[0]
    joined = by_fold.loc[by_fold.model_id.eq("DivSchedule"), ["fold", "return_rmse"]].merge(
        by_fold.loc[by_fold.model_id.eq("Z0"), ["fold", "return_rmse"]],
        on="fold", suffixes=("_schedule", "_z0"), validate="one_to_one")
    wins = int((joined.return_rmse_schedule < joined.return_rmse_z0).sum())
    decision = {"experiment_id": contract["experiment_id"],
                "event_count": int(adjustment.ex_date.notna().sum()),
                "z0_rmse": float(z0.return_rmse), "schedule_rmse": float(selected.return_rmse),
                "rmse_improvement_pct": float(100 * (1 - selected.return_rmse / z0.return_rmse)),
                "z0_mae": float(z0.return_mae), "schedule_mae": float(selected.return_mae),
                "mae_improvement_pct": float(100 * (1 - selected.return_mae / z0.return_mae)),
                "fold_wins": wins,
                "passes_development_gate": bool(selected.return_rmse <= 0.99 * z0.return_rmse and
                                                wins >= 2 and selected.return_mae <= 1.01 * z0.return_mae),
                "attribution": "The announced distribution is external deterministic information; this decision does not establish a fuzzy-model gain."}
    OUT.mkdir(parents=True, exist_ok=False)
    comparison.to_csv(OUT / "predictions.csv", index=False)
    pooled.to_csv(OUT / "comparison_pooled.csv", index=False)
    by_fold.to_csv(OUT / "comparison_by_fold.csv", index=False)
    adjustment.loc[adjustment.ex_date.notna()].to_csv(OUT / "event_days.csv", index=False)
    (OUT / "manifest.json").write_text(json.dumps({"contract_sha256": sha256(CONTRACT),
        "event_source_sha256": sha256(EVENTS), "source_sha256": sha256(Path(__file__)),
        "v1_development_predictions_sha256": sha256(ROOT / "output/development/predictions.csv"),
        "v1_data_sha256": sha256(ROOT / "data/processed/srt_windows_unscaled.npz"),
        "fold_assignments_sha256": sha256(ROOT / "data/processed/fold_assignments.csv")}, indent=2), encoding="utf-8")
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return decision


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
