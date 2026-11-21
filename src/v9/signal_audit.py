"""Inspect saved 20x6 inputs and forecasts without fitting any new model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.project_eval import Dataset, PREDICTION_COLUMNS, VALIDATION_FOLDS


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs/v9_signal_audit_contract_2026-11-05.json"
OUT = ROOT / "output/v9/signal_audit"
FEATURE_LABELS = ("srt_return_1d", "srt_momentum_5d", "srt_volatility_20d",
                  "sti_return_1d", "sora_level", "sora_change_20obs")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def association(x: np.ndarray, y: np.ndarray, *, rank: bool = False) -> float:
    """Descriptive correlation; NaN if a variable is constant in this slice."""
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(a) != len(b) or len(a) < 3 or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Aligned, finite association arrays with >=3 rows required")
    if rank:
        a = pd.Series(a).rank(method="average").to_numpy()
        b = pd.Series(b).rank(method="average").to_numpy()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def mean_split(values: np.ndarray, returns: np.ndarray, threshold: float) -> dict:
    high = values >= threshold
    low = ~high
    if not high.any() or not low.any():
        return {"n_low": int(low.sum()), "n_high": int(high.sum()),
                "mean_low": None, "mean_high": None, "high_minus_low": None}
    mean_low, mean_high = float(returns[low].mean()), float(returns[high].mean())
    return {"n_low": int(low.sum()), "n_high": int(high.sum()),
            "mean_low": mean_low, "mean_high": mean_high,
            "high_minus_low": mean_high - mean_low}


def raw_feature_tables(data: Dataset):
    correlations, splits, regimes, within = [], [], [], []
    for fold in VALIDATION_FOLDS:
        train = data.ids(fold, "train")
        evaluation = data.ids(fold, "validation")
        for index, label in enumerate(FEATURE_LABELS):
            train_x, eval_x = data.x[train, -1, index], data.x[evaluation, -1, index]
            train_y, eval_y = data.y[train], data.y[evaluation]
            threshold = float(np.median(train_x))
            train_split, eval_split = mean_split(train_x, train_y, threshold), mean_split(eval_x, eval_y, threshold)
            correlations.append({"fold": fold, "feature_index": index, "feature": label,
                "n_train": len(train), "n_eval": len(evaluation),
                "train_pearson": association(train_x, train_y),
                "eval_pearson": association(eval_x, eval_y),
                "train_spearman": association(train_x, train_y, rank=True),
                "eval_spearman": association(eval_x, eval_y, rank=True)})
            splits.append({"fold": fold, "feature_index": index, "feature": label,
                           "train_median_threshold": threshold,
                           **{f"train_{key}": val for key, val in train_split.items()},
                           **{f"eval_{key}": val for key, val in eval_split.items()}})
        cut = float(np.median(data.x[train, -1, 2]))
        for role, ids in (("train", train), ("validation", evaluation)):
            high = data.x[ids, -1, 2] >= cut
            y = data.y[ids]
            for state, mask in (("low", ~high), ("high", high)):
                if not mask.any():
                    raise ValueError("Volatility regime has no cases")
                sub = y[mask]
                regimes.append({"fold": fold, "role": role, "state": state,
                    "train_median_volatility": cut, "n": int(mask.sum()),
                    "mean_next_return": float(sub.mean()),
                    "std_next_return": float(sub.std(ddof=0)),
                    "mean_abs_next_return": float(np.abs(sub).mean()),
                    "positive_fraction": float((sub > 0).mean())})
                for feature in (1, 3):
                    predictor = data.x[ids[mask], -1, feature]
                    within.append({"fold": fold, "role": role, "state": state,
                        "predictor": FEATURE_LABELS[feature], "n": len(sub),
                        "pearson_to_next_return": association(predictor, sub),
                        "spearman_to_next_return": association(predictor, sub, rank=True)})
    return tuple(pd.DataFrame(rows) for rows in (correlations, splits, regimes, within))


def saved_model_regime_scores(data: Dataset, splits: pd.DataFrame):
    old = pd.read_csv(ROOT / "output/development/predictions.csv")
    old = old.loc[old.model_id.eq("Fuzzy2") & old.source_group.eq("all"), PREDICTION_COLUMNS]
    v7 = pd.read_csv(ROOT / "output/v7/staged_experts/predictions.csv")
    v7 = v7.loc[v7.model_id.eq("V7StagedGaussian"), PREDICTION_COLUMNS]
    saved = pd.concat((old, v7, data.zero_predictions()), ignore_index=True)
    data.validate_predictions(saved)
    rows = []
    for fold in VALIDATION_FOLDS:
        cut = float(splits.loc[(splits.fold == fold) & splits.feature.eq("srt_volatility_20d"),
                               "train_median_threshold"].iloc[0])
        group = saved.loc[saved.fold.eq(fold)]
        for (model, seed), run in group.groupby(["model_id", "seed"], sort=True):
            ids = run.window_id.to_numpy(dtype=int)
            high = data.x[ids, -1, 2] >= cut
            error = run.y_pred.to_numpy(dtype=float) - data.y[ids]
            for state, mask in (("low", ~high), ("high", high)):
                rows.append({"fold": fold, "model_id": model, "seed": int(seed),
                             "state": state, "n": int(mask.sum()),
                             "rmse": float(np.sqrt(np.mean(error[mask] ** 2))),
                             "mae": float(np.mean(np.abs(error[mask])))})
    detailed = pd.DataFrame(rows)
    summary = detailed.groupby(["fold", "model_id", "state"], as_index=False).agg(
        n=("n", "first"), rmse=("rmse", "mean"), mae=("mae", "mean"), seeds=("seed", "nunique"))
    return detailed, summary


def v7_expert_regions(data: Dataset):
    components = pd.read_csv(ROOT / "output/v7/staged_experts/components.csv")
    components = components.loc[components.arm.eq("V7StagedGaussian")]
    runs = json.loads((ROOT / "output/v7/staged_experts/manifests.json").read_text(encoding="utf-8"))["runs"]
    accepted = {(r["fold"], r["seed"]): r["selections"]["gaussian"]["chosen_epoch"]
                for r in runs if r["selections"]["gaussian"]["chosen_epoch"] >= 0}
    rows = []
    for (fold, seed), chosen_epoch in accepted.items():
        group = components.loc[components.fold.eq(fold) & components.seed.eq(seed)]
        if len(group) != len(data.ids(fold, "validation")):
            raise ValueError("Incomplete saved V7 expert components")
        y = data.y[group.window_id.to_numpy(dtype=int)]
        for rule in (1, 2):
            own = f"expert{rule}_raw"
            other = f"expert{3 - rule}_raw"
            mask = group[f"alpha{rule}"].to_numpy(dtype=float) >= 0.8
            n = int(mask.sum())
            own_mae = float(np.abs(group.loc[mask, own].to_numpy() - y[mask]).mean()) if n else None
            other_mae = float(np.abs(group.loc[mask, other].to_numpy() - y[mask]).mean()) if n else None
            rows.append({"fold": fold, "seed": seed, "selected_epoch": chosen_epoch,
                         "rule_id": rule, "activation_cut": 0.8, "n": n,
                         "small_region": n < 20, "own_mae": own_mae,
                         "other_mae": other_mae,
                         "own_better": bool(own_mae < other_mae) if n >= 20 else None})
    return pd.DataFrame(rows)


def interpret(correlations: pd.DataFrame, splits: pd.DataFrame,
              regime_scores: pd.DataFrame, experts: pd.DataFrame) -> dict:
    feature_summary = []
    for feature, group in correlations.groupby("feature", sort=False):
        feature_splits = splits.loc[splits.feature.eq(feature)]
        pearson_signs = np.sign(group.eval_pearson.to_numpy())
        split_signs = np.sign(feature_splits.eval_high_minus_low.to_numpy(dtype=float))
        feature_summary.append({"feature": feature,
            "train_eval_pearson_same_sign_folds": int((np.sign(group.train_pearson) == pearson_signs).sum()),
            "eval_pearson_all_three_same_sign": bool(np.all(pearson_signs == pearson_signs[0])),
            "train_eval_median_difference_same_sign_folds": int((
                np.sign(feature_splits.train_high_minus_low.to_numpy(dtype=float)) == split_signs).sum()),
            "eval_median_difference_all_three_same_sign": bool(np.all(split_signs == split_signs[0]))})
    model_wins = {}
    for model in ("Fuzzy2", "V7StagedGaussian"):
        comparison = regime_scores.loc[regime_scores.model_id.eq(model), ["fold", "state", "rmse"]].merge(
            regime_scores.loc[regime_scores.model_id.eq("Z0"), ["fold", "state", "rmse"]],
            on=["fold", "state"], suffixes=("_model", "_z0"), validate="one_to_one")
        model_wins[model] = {state: int((g.rmse_model < g.rmse_z0).sum())
                             for state, g in comparison.groupby("state")}
    qualified = experts.loc[~experts.small_region]
    expert_summary = {f"rule_{rule}": {"qualifying_regions": int(len(group)),
        "own_better_regions": int(group.own_better.sum())}
        for rule, group in qualified.groupby("rule_id")}
    for rule in (1, 2):
        expert_summary.setdefault(f"rule_{rule}", {"qualifying_regions": 0, "own_better_regions": 0})
    return {"experiment_id": "V9-H21-original-input-signal-readonly-audit-01",
            "feature_sign_consistency": feature_summary,
            "saved_model_rmse_fold_wins_vs_z0_by_regime": model_wins,
            "v7_selected_gaussian_runs": int(experts[["fold", "seed"]].drop_duplicates().shape[0]),
            "v7_expert_high_activation_summary": expert_summary,
            "scope": "read-only descriptive development analysis; no new fit or final-period score"}


def run() -> dict:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refuse to overwrite V9: {OUT}")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract["experiment_id"] != "V9-H21-original-input-signal-readonly-audit-01":
        raise ValueError("Unexpected V9 contract")
    source_files = [CONTRACT, Path(__file__), ROOT / "src/project_eval.py",
        ROOT / "data/processed/srt_windows_unscaled.npz",
        ROOT / "data/processed/window_index.csv",
        ROOT / "data/processed/fold_assignments.csv",
        ROOT / "output/development/predictions.csv",
        ROOT / "output/v7/staged_experts/predictions.csv",
        ROOT / "output/v7/staged_experts/components.csv",
        ROOT / "output/v7/staged_experts/manifests.json"]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in source_files}
    data = Dataset(ROOT)
    if tuple(data.features) != ("srt_log_return_raw", "srt_momentum_5", "srt_volatility_20",
                                "sti_log_return_asof", "sora_rate_pct_asof",
                                "sora_change_20_publications_pp"):
        raise ValueError(f"Unexpected original input order: {data.features}")
    correlations, splits, regimes, within = raw_feature_tables(data)
    model_seed, model_summary = saved_model_regime_scores(data, splits)
    experts = v7_expert_regions(data)
    summary = interpret(correlations, splits, model_summary, experts)
    if hashes != {str(p.relative_to(ROOT)): digest(p) for p in source_files}:
        raise RuntimeError("A V9 source changed during read-only analysis")
    OUT.mkdir(parents=True, exist_ok=False)
    for name, table in (("feature_correlations.csv", correlations),
                        ("feature_median_splits.csv", splits),
                        ("volatility_regimes.csv", regimes),
                        ("within_regime_correlations.csv", within),
                        ("saved_model_by_regime_seed.csv", model_seed),
                        ("saved_model_by_regime_summary.csv", model_summary),
                        ("v7_expert_regions.csv", experts)):
        table.to_csv(OUT / name, index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "source_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
