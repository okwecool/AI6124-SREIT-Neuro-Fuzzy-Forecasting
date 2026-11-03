"""Shared, leakage-aware evaluation contract for the AI6124 SRT project.

This module deliberately has no model-training code. It prepares chronological
folds, checks prediction files, scores forecasts, and runs the predeclared
price-only next-open portfolio exercise.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FEATURE_GROUPS = {
    "srt": (0, 1, 2),
    "srt_sora": (0, 1, 2, 4, 5),
    "all": (0, 1, 2, 3, 4, 5),
}
VALIDATION_FOLDS = ("validation_1", "validation_2", "validation_3")
PREDICTION_COLUMNS = (
    "model_id", "source_group", "fold", "seed", "window_id", "y_pred"
)


@dataclass(frozen=True)
class Standardizer:
    """Train-only columnwise input and target z-score transformations."""

    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray) -> "Standardizer":
        if x.ndim != 3 or y.ndim != 1 or len(x) != len(y) or len(y) == 0:
            raise ValueError("Expected nonempty X=(N,20,F), y=(N,)")
        x_mean = x.mean(axis=(0, 1), dtype=np.float64)
        x_std = x.std(axis=(0, 1), dtype=np.float64)
        x_std = np.where(x_std == 0, 1.0, x_std)
        y_mean = float(y.mean(dtype=np.float64))
        y_std = float(y.std(dtype=np.float64)) or 1.0
        return cls(x_mean, x_std, y_mean, y_std)

    def x_transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.x_mean) / self.x_std).astype(np.float32)

    def y_transform(self, y: np.ndarray) -> np.ndarray:
        return ((y - self.y_mean) / self.y_std).astype(np.float32)

    def y_inverse(self, y_scaled: np.ndarray) -> np.ndarray:
        return np.asarray(y_scaled, dtype=np.float64) * self.y_std + self.y_mean


@dataclass(frozen=True)
class PreparedFold:
    train_ids: np.ndarray
    eval_ids: np.ndarray
    x_train: np.ndarray
    y_train: np.ndarray
    x_eval: np.ndarray
    y_eval_raw: np.ndarray
    scaler: Standardizer


class Dataset:
    """The fixed saved snapshot. Array row position equals window_id."""

    def __init__(self, root: Path = ROOT):
        processed = Path(root) / "data" / "processed"
        arrays = np.load(processed / "srt_windows_unscaled.npz", allow_pickle=False)
        self.x = arrays["X"]
        self.y = arrays["y_raw"]
        self.features = tuple(str(value) for value in arrays["feature_names"])
        self.index = pd.read_csv(
            processed / "window_index.csv",
            parse_dates=["window_start_date", "forecast_date", "target_date"],
        )
        self.assignments = pd.read_csv(processed / "fold_assignments.csv")
        self.daily = pd.read_csv(processed / "srt_daily_aligned.csv", parse_dates=["date"])
        if not np.array_equal(self.index.window_id.to_numpy(), np.arange(len(self.x))):
            raise ValueError("window_index.csv must have one sequential row per X entry")
        if len(self.x) != len(self.y) or not np.isfinite(self.x).all() or not np.isfinite(self.y).all():
            raise ValueError("Window arrays are misaligned or nonfinite")

    def ids(self, fold: str, role: str) -> np.ndarray:
        selected = self.assignments.loc[
            (self.assignments.fold == fold) & (self.assignments.role == role), "window_id"
        ].to_numpy(dtype=np.int64)
        if len(selected) == 0 or len(np.unique(selected)) != len(selected):
            raise ValueError(f"Missing or duplicate assignment: {fold}/{role}")
        return np.sort(selected)

    def prepare(self, fold: str, source_group: str = "all", *, inner_stop: bool = False) -> PreparedFold:
        """Use inner_stop=True for early stopping; False for full outer refit."""
        if source_group not in FEATURE_GROUPS:
            raise ValueError(f"Unknown source group: {source_group}")
        train_ids = self.ids(fold, "train")
        if inner_stop:
            count = math.ceil(0.15 * len(train_ids))
            fit_ids, eval_ids = train_ids[:-count], train_ids[-count:]
        else:
            fit_ids = train_ids
            eval_ids = self.ids(fold, "test" if fold == "final_holdout" else "validation")
        if len(fit_ids) == 0 or self.index.loc[fit_ids, "target_date"].max() >= self.index.loc[
            eval_ids, "target_date"
        ].min():
            raise ValueError("Fold chronology is invalid")
        columns = list(FEATURE_GROUPS[source_group])
        x_fit = self.x[fit_ids][:, :, columns]
        x_eval = self.x[eval_ids][:, :, columns]
        scaler = Standardizer.fit(x_fit, self.y[fit_ids])
        return PreparedFold(
            fit_ids, eval_ids, scaler.x_transform(x_fit), scaler.y_transform(self.y[fit_ids]),
            scaler.x_transform(x_eval), self.y[eval_ids].copy(), scaler,
        )

    def zero_predictions(self, *, include_holdout: bool = False) -> pd.DataFrame:
        folds = VALIDATION_FOLDS + (("final_holdout",) if include_holdout else ())
        rows = []
        for fold in folds:
            role = "test" if fold == "final_holdout" else "validation"
            for window_id in self.ids(fold, role):
                rows.append(("Z0", "all", fold, 0, int(window_id), 0.0))
        return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)

    def validate_predictions(self, predictions: pd.DataFrame, *, complete: bool = True) -> pd.DataFrame:
        """Attach known labels/prices and reject IDs outside a fold's evaluation set."""
        missing = set(PREDICTION_COLUMNS) - set(predictions.columns)
        if missing:
            raise ValueError(f"Prediction file lacks columns: {sorted(missing)}")
        frame = predictions.copy()
        frame["window_id"] = pd.to_numeric(frame.window_id, errors="raise").astype(int)
        frame["seed"] = pd.to_numeric(frame.seed, errors="raise").astype(int)
        frame["y_pred"] = pd.to_numeric(frame.y_pred, errors="raise").astype(float)
        if not np.isfinite(frame.y_pred).all():
            raise ValueError("Predictions must be finite raw log returns")
        if not frame.source_group.isin(FEATURE_GROUPS).all():
            raise ValueError("Unknown source_group in prediction file")
        key = ["model_id", "source_group", "fold", "seed", "window_id"]
        if frame.duplicated(key).any():
            raise ValueError("Duplicate prediction key")
        for (_, _, fold, _), group in frame.groupby(key[:-1], sort=False):
            if fold not in VALIDATION_FOLDS + ("final_holdout",):
                raise ValueError(f"Unknown fold: {fold}")
            expected = set(self.ids(fold, "test" if fold == "final_holdout" else "validation"))
            supplied = set(group.window_id)
            if not supplied <= expected or (complete and supplied != expected):
                raise ValueError(f"Wrong window IDs for {fold}: supplied={len(supplied)}, expected={len(expected)}")
        meta = self.index[["window_id", "forecast_date", "target_date"]].copy()
        meta["y_true"] = self.y.astype(float)
        closes = self.daily.set_index("date")["srt_close"]
        meta["close_t"] = meta.forecast_date.map(closes)
        meta["close_target"] = meta.target_date.map(closes)
        result = frame.merge(meta, on="window_id", how="left", validate="many_to_one")
        if result[["y_true", "close_t", "close_target"]].isna().any().any():
            raise ValueError("Prediction metadata could not be aligned")
        result["close_pred"] = result.close_t * np.exp(result.y_pred)
        return result.sort_values(key).reset_index(drop=True)

    def forecast_scores(self, predictions: pd.DataFrame) -> pd.DataFrame:
        data = self.validate_predictions(predictions)
        records = []
        for (model_id, source_group, fold, seed), group in data.groupby(
            ["model_id", "source_group", "fold", "seed"], sort=True
        ):
            error = group.y_pred.to_numpy() - group.y_true.to_numpy()
            price_error = group.close_pred.to_numpy() - group.close_target.to_numpy()
            nonzero = group.y_true.to_numpy() != 0
            majority_train = self.y[self.ids(fold, "train")]
            majority_sign = 1 if np.sum(majority_train > 0) >= np.sum(majority_train < 0) else -1
            records.append({
                "model_id": model_id,
                "source_group": source_group,
                "fold": fold,
                "seed": int(seed),
                "n": len(group),
                "return_rmse": float(np.sqrt(np.mean(error**2))),
                "return_mae": float(np.mean(np.abs(error))),
                "price_rmse_sgd": float(np.sqrt(np.mean(price_error**2))),
                "price_mae_sgd": float(np.mean(np.abs(price_error))),
                "direction_n": int(nonzero.sum()),
                "zero_actual_n": int((~nonzero).sum()),
                "direction_accuracy": float(np.mean(np.sign(group.y_pred.to_numpy()[nonzero]) == np.sign(group.y_true.to_numpy()[nonzero]))),
                "train_majority_direction": majority_sign,
                "train_majority_accuracy": float(np.mean(majority_sign == np.sign(group.y_true.to_numpy()[nonzero]))),
            })
        return pd.DataFrame(records)

    def pooled_validation_scores(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Score each model/seed on all 378 nonoverlapping validation targets."""
        data = self.validate_predictions(predictions)
        data = data.loc[data.fold.isin(VALIDATION_FOLDS)]
        rows = []
        for (model_id, source_group, seed), group in data.groupby(
            ["model_id", "source_group", "seed"], sort=True
        ):
            if set(group.fold) != set(VALIDATION_FOLDS) or len(group) != 378:
                raise ValueError(f"Pooled score needs all three validation folds: {model_id}/{source_group}/{seed}")
            error = group.y_pred.to_numpy() - group.y_true.to_numpy()
            price_error = group.close_pred.to_numpy() - group.close_target.to_numpy()
            nonzero = group.y_true.to_numpy() != 0
            rows.append({
                "model_id": model_id,
                "source_group": source_group,
                "seed": int(seed),
                "n": len(group),
                "return_rmse": float(np.sqrt(np.mean(error**2))),
                "return_mae": float(np.mean(np.abs(error))),
                "price_rmse_sgd": float(np.sqrt(np.mean(price_error**2))),
                "price_mae_sgd": float(np.mean(np.abs(price_error))),
                "direction_n": int(nonzero.sum()),
                "zero_actual_n": int((~nonzero).sum()),
                "direction_accuracy": float(np.mean(np.sign(group.y_pred.to_numpy()[nonzero]) == np.sign(group.y_true.to_numpy()[nonzero]))),
            })
        return pd.DataFrame(rows)

    def paired_validation_rmse_difference(
        self, predictions: pd.DataFrame, *, challenger: str = "Fuzzy2", benchmark: str = "LSTM32",
        source_group: str = "all",
    ) -> pd.DataFrame:
        """Positive delta means the challenger has a larger (worse) RMSE."""
        data = self.validate_predictions(predictions)
        data = data.loc[data.fold.isin(VALIDATION_FOLDS) & (data.source_group == source_group)]
        left = data.loc[data.model_id == challenger, ["seed", "fold", "window_id", "y_true", "y_pred"]]
        right = data.loc[data.model_id == benchmark, ["seed", "fold", "window_id", "y_pred"]]
        merged = left.merge(right, on=["seed", "fold", "window_id"], suffixes=("_challenger", "_benchmark"), validate="one_to_one")
        if len(merged) != len(left) or len(merged) != len(right):
            raise ValueError("Paired models must have the same seeds and validation target IDs")
        rows = []
        for seed, group in merged.groupby("seed"):
            if set(group.fold) != set(VALIDATION_FOLDS) or len(group) != 378:
                raise ValueError("Paired comparison needs all three validation folds")
            yc = group.y_pred_challenger.to_numpy() - group.y_true.to_numpy()
            yb = group.y_pred_benchmark.to_numpy() - group.y_true.to_numpy()
            rows.append({
                "challenger": challenger,
                "benchmark": benchmark,
                "source_group": source_group,
                "seed": int(seed),
                "n": len(group),
                "challenger_rmse": float(np.sqrt(np.mean(yc**2))),
                "benchmark_rmse": float(np.sqrt(np.mean(yb**2))),
                "delta_rmse": float(np.sqrt(np.mean(yc**2)) - np.sqrt(np.mean(yb**2))),
                "delta_mae": float(np.mean(np.abs(yc)) - np.mean(np.abs(yb))),
            })
        return pd.DataFrame(rows)

    def portfolio_scores(self, predictions: pd.DataFrame, *, cost_bps: int = 10) -> pd.DataFrame:
        """Price-only next-open strategy; missing-open intervals are common omissions."""
        if cost_bps < 0:
            raise ValueError("Transaction cost must be nonnegative")
        data = self.validate_predictions(predictions)
        calendar = self.daily[["date", "srt_open"]].sort_values("date").reset_index(drop=True)
        next_open = calendar.srt_open.shift(-1)
        next_date = calendar.date.shift(-1)
        by_date = calendar.set_index("date")
        results = []
        for (model_id, source_group, fold, seed), group in data.groupby(
            ["model_id", "source_group", "fold", "seed"], sort=True
        ):
            positions = {
                "signal_long_cash": (group.y_pred.to_numpy() > 0).astype(int),
                "always_long": np.ones(len(group), dtype=int),
                "always_cash": np.zeros(len(group), dtype=int),
            }
            intervals = group[["target_date"]].copy()
            intervals["entry_open"] = intervals.target_date.map(by_date.srt_open)
            intervals["exit_open"] = intervals.target_date.map(pd.Series(next_open.to_numpy(), index=calendar.date))
            intervals["exit_date"] = intervals.target_date.map(pd.Series(next_date.to_numpy(), index=calendar.date))
            # An evaluation fold ends at its last target date. Its final signal
            # cannot borrow the following fold's price to close a trade.
            within_fold = intervals.exit_date <= group.target_date.max()
            intervals["valid"] = intervals.entry_open.notna() & intervals.exit_open.notna() & within_fold
            if (intervals.loc[intervals.valid, ["entry_open", "exit_open"]] <= 0).any().any():
                raise ValueError("Open prices must be positive")
            order = np.argsort(group.target_date.to_numpy())
            intervals = intervals.iloc[order].reset_index(drop=True)
            for strategy, raw_position in positions.items():
                position = raw_position[order]
                returns = []
                turnover = 0
                trades = 0
                equity = 1.0
                peak = 1.0
                max_drawdown = 0.0
                prev_position = 0
                prev_exit = None
                first_entry = None
                last_exit = None
                for i, row in intervals.iterrows():
                    if not row.valid:
                        continue
                    if first_entry is None:
                        first_entry = row.target_date
                    if prev_exit is not None and row.target_date != prev_exit:
                        prev_position = 0  # common cash gap; prior segment was liquidated
                    entry_change = abs(int(position[i]) - prev_position)
                    next_valid_and_contiguous = (
                        i + 1 < len(intervals) and bool(intervals.loc[i + 1, "valid"])
                        and intervals.loc[i + 1, "target_date"] == row.exit_date
                    )
                    exit_change = int(position[i]) if not next_valid_and_contiguous else 0
                    interval_return = (
                        int(position[i]) * (float(row.exit_open) / float(row.entry_open) - 1)
                        - (cost_bps / 10000) * (entry_change + exit_change)
                    )
                    returns.append(interval_return)
                    turnover += entry_change + exit_change
                    trades += int(entry_change > 0) + int(exit_change > 0)
                    equity *= 1 + interval_return
                    peak = max(peak, equity)
                    max_drawdown = max(max_drawdown, 1 - equity / peak)
                    prev_position = 0 if exit_change else int(position[i])
                    prev_exit = row.exit_date
                    last_exit = row.exit_date
                if not returns:
                    raise ValueError(f"No tradable intervals for {fold}")
                years = (last_exit - first_entry).days / 365.25
                results.append({
                    "model_id": model_id,
                    "source_group": source_group,
                    "fold": fold,
                    "seed": int(seed),
                    "strategy": strategy,
                    "cost_bps_per_side": cost_bps,
                    "eligible_intervals": len(returns),
                    "omitted_intervals": len(intervals) - len(returns),
                    "cumulative_price_return": equity - 1,
                    "calendar_annualized_price_return": equity ** (1 / years) - 1 if years > 0 else np.nan,
                    "max_drawdown": max_drawdown,
                    "one_way_turnover": turnover,
                    "one_way_trade_count": trades,
                    "dividends_included": False,
                })
        return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("zero", "score", "pooled", "portfolio"))
    parser.add_argument("--input", type=Path, help="Prediction CSV for score/portfolio")
    parser.add_argument("--out", type=Path, help="Where to save CSV output")
    parser.add_argument("--cost-bps", type=int, default=10)
    args = parser.parse_args()
    data = Dataset()
    if args.action == "zero":
        result = data.zero_predictions()
    else:
        if args.input is None:
            parser.error("--input is required for score and portfolio")
        predictions = pd.read_csv(args.input)
        if args.action == "score":
            result = data.forecast_scores(predictions)
        elif args.action == "pooled":
            result = data.pooled_validation_scores(predictions)
        else:
            result = data.portfolio_scores(predictions, cost_bps=args.cost_bps)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.out, index=False)
        print(f"Saved {len(result)} rows to {args.out}")
    else:
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
