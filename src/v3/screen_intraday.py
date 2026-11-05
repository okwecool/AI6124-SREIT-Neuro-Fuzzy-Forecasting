"""One bounded, explicitly changed target; no modification to the V1 experiment."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.project_eval import Dataset, Standardizer, VALIDATION_FOLDS
from src.v3.hybrid_training import inputs_and_residual


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs/v3_intraday_screen_contract_2026-11-05.json"
OUT = ROOT / "output/v3/intraday_screen"


def targets(data: Dataset) -> tuple[np.ndarray, np.ndarray]:
    daily = data.daily.set_index("date")
    intraday = np.log(daily.srt_close / daily.srt_open)
    eligible = data.index.target_date.le(pd.Timestamp("2025-09-30"))
    y, lag = np.full(len(data.y), np.nan), np.full(len(data.y), np.nan)
    rows = data.index.loc[eligible]
    y[rows.window_id] = rows.target_date.map(intraday).to_numpy()
    lag[rows.window_id] = rows.forecast_date.map(intraday).to_numpy()
    if not np.isfinite(y[eligible]).all() or not np.isfinite(lag[eligible]).all():
        raise ValueError("Missing development OHLC; fixed sample cannot be changed")
    return y, lag


def ridge_predict(x, y, train, evaluate, penalty):
    scale = Standardizer.fit(x[train], y[train])
    a = scale.x_transform(x[train]).reshape(len(train), -1).astype(float)
    b = scale.x_transform(x[evaluate]).reshape(len(evaluate), -1).astype(float)
    a, b = np.column_stack((a, np.ones(len(a)))), np.column_stack((b, np.ones(len(b))))
    target = y[train] / scale.y_std
    weights = np.linalg.solve(a.T @ a / len(a) + penalty * np.eye(a.shape[1]), a.T @ target / len(a))
    return (b @ weights) * scale.y_std


def run():
    if OUT.exists():
        raise FileExistsError("Do not overwrite this target screen")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract["lambda_grid"] != [0.1, 1.0, 10.0]:
        raise ValueError("Unexpected grid")
    data = Dataset(ROOT)
    y, previous = targets(data)
    dependencies = [CONTRACT, Path(__file__), ROOT / "src/v3/hybrid_training.py", ROOT / "src/project_eval.py",
                    *[ROOT / "data/processed" / p for p in ("srt_daily_aligned.csv", "srt_windows_unscaled.npz",
                                                            "window_index.csv", "fold_assignments.csv")]]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "pre_execution.json").write_text(json.dumps({"contract": contract, "dependency_sha256": hashes}, indent=2), encoding="utf-8")
    rows, fits = [], []

    def append(model, fold, ids, prediction):
        for i, p in zip(ids, prediction):
            rows.append({"model": model, "fold": fold, "window_id": int(i),
                         "forecast_date": str(data.index.forecast_date.iloc[i].date()),
                         "target_date": str(data.index.target_date.iloc[i].date()),
                         "y_true": float(y[i]), "y_pred": float(p)})

    for fold in VALIDATION_FOLDS:
        train, evaluate = data.ids(fold, "train"), data.ids(fold, "validation")
        n = math.ceil(0.15 * len(train))
        inner, stop = train[:-n], train[-n:]
        for left, right in ((inner, stop), (train, evaluate)):
            if data.index.target_date.iloc[left].max() > data.index.forecast_date.iloc[right].min():
                raise ValueError("Unavailable intraday training labels")
        for model, prediction in (("ZeroIntraday", np.zeros(len(evaluate))),
                                  ("MeanTrainIntraday", np.full(len(evaluate), y[train].mean())),
                                  ("PreviousIntraday", previous[evaluate]),
                                  ("ReversePreviousIntraday", -previous[evaluate])):
            append(model, fold, evaluate, prediction)
        for arm in contract["input_arms"]:
            x, _ = inputs_and_residual(data, arm)
            choices = [{"lambda": None, "inner_rmse": float(np.sqrt(np.mean(y[stop] ** 2)))}]
            for penalty in contract["lambda_grid"]:
                prediction = ridge_predict(x, y, inner, stop, penalty)
                choices.append({"lambda": penalty, "inner_rmse": float(np.sqrt(np.mean((prediction - y[stop]) ** 2)))})
            chosen = min(choices, key=lambda c: (c["inner_rmse"], -float("inf") if c["lambda"] is None else -c["lambda"]))
            prediction = (np.zeros(len(evaluate)) if chosen["lambda"] is None else
                          ridge_predict(x, y, train, evaluate, chosen["lambda"]))
            append("RidgeIntraday" + arm.title(), fold, evaluate, prediction)
            fits.append({"fold": fold, "arm": arm, "choices": choices, "chosen": chosen})
    predictions = pd.DataFrame(rows)
    if predictions.duplicated(["model", "fold", "window_id"]).any() or not (predictions.groupby("model").size() == 378).all():
        raise ValueError("Duplicate or incomplete prediction keys")
    def score(g):
        e = g.y_pred.to_numpy() - g.y_true.to_numpy()
        return {"n": len(g), "rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e)))}
    pooled = pd.DataFrame([{"model": name, **score(g)} for name, g in predictions.groupby("model")])
    folds = pd.DataFrame([{"model": name, "fold": fold, **score(g)} for (name, fold), g in predictions.groupby(["model", "fold"])])
    reference = pooled.loc[~pooled.model.str.startswith("Ridge")].sort_values("rmse").iloc[0]
    decisions = []
    for row in pooled.loc[pooled.model.str.startswith("Ridge")].itertuples():
        a = folds.loc[folds.model.eq(row.model)].set_index("fold")
        b = folds.loc[folds.model.eq(reference.model)].set_index("fold")
        wins = int((a.rmse < b.rmse).sum())
        decisions.append({"model": row.model, "reference": reference.model,
                          "gain_pct": float(100 * (1 - row.rmse / reference.rmse)), "fold_wins": wins,
                          "passes": bool(row.rmse <= 0.99 * reference.rmse and wins >= 2 and row.mae <= 1.01 * reference.mae)})
    for name, frame in (("predictions.csv", predictions), ("comparison_pooled.csv", pooled), ("comparison_by_fold.csv", folds)):
        frame.to_csv(OUT / name, index=False)
    (OUT / "fits.json").write_text(json.dumps(fits, indent=2), encoding="utf-8")
    (OUT / "decision.json").write_text(json.dumps(decisions, indent=2), encoding="utf-8")
    if hashes != {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}:
        raise RuntimeError("Dependencies changed during screen")
    print(pooled.to_string(index=False))
    print(json.dumps(decisions, indent=2))


if __name__ == "__main__":
    run()
