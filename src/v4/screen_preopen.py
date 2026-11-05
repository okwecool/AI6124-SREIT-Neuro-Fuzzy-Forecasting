"""Fixed information-timing screen before any V4 neural training."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.project_eval import Dataset, PREDICTION_COLUMNS, VALIDATION_FOLDS
from src.v2.screen_distributions import EVENTS, announced_adjustments
from src.v3.hybrid_training import partition
from src.v3.screen_intraday import ridge_predict
from src.v4.preopen_data import build


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/v4/preopen_screen"
CONTRACT = ROOT / "docs/v4_preopen_screen_contract_2026-11-05.json"


def run():
    if OUT.exists():
        raise FileExistsError("Do not overwrite information-screen results")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    data = Dataset(ROOT)
    x, y, alignment = build(data)
    designs = {"base_endpoint": x["close"][:, -1:, :6], "close_us": x["close"][:, -1:, 6:],
               "preopen_us": x["preopen"][:, -1:, 6:], "close_all": x["close"][:, -1:, :],
               "preopen_all": x["preopen"][:, -1:, :]}
    if list(designs) != contract["configs"]:
        raise ValueError("Contract configs differ")
    deps = [CONTRACT, Path(__file__), ROOT / "src/v4/preopen_data.py", ROOT / "src/v3/hybrid_training.py",
            ROOT / "src/v3/screen_intraday.py", ROOT / "src/project_eval.py", ROOT / "src/v2/screen_distributions.py", EVENTS,
            *[ROOT / "data/processed" / n for n in ("srt_daily_aligned.csv", "srt_windows_unscaled.npz", "window_index.csv", "fold_assignments.csv")],
            *[ROOT / f"data/raw/yahoo_{s}_20210901_20250930.json" for s in ("vnq", "spy")],
            *[ROOT / f"data/source_us_{s}_manifest.json" for s in ("vnq", "spy")]]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in deps}
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "pre_execution.json").write_text(json.dumps({"contract": contract, "dependency_sha256": hashes}, indent=2), encoding="utf-8")
    alignment.to_csv(OUT / "time_alignment.csv", index=False)
    offset = announced_adjustments(data, pd.read_csv(EVENTS)).set_index("window_id").adjustment
    frames, fits = [], []
    def append(model, fold, ids, residual):
        pred = residual + offset.loc[ids].to_numpy()
        frames.append(pd.DataFrame([(model, "all", 0, fold, 0, int(i), float(p)) for i, p in zip(ids, pred)], columns=PREDICTION_COLUMNS))
    for fold in VALIDATION_FOLDS:
        inner, stop, train, evaluate = partition(data, fold)
        append("DivSchedule", fold, evaluate, np.zeros(len(evaluate)))
        append("MeanTrainResidualPlusSchedule", fold, evaluate, np.full(len(evaluate), y[train].mean()))
        for name, matrix in designs.items():
            trials = [{"lambda": None, "rmse": float(np.sqrt(np.mean(y[stop] ** 2)))}]
            for penalty in (0.1, 1.0, 10.0):
                pred = ridge_predict(matrix, y, inner, stop, penalty)
                trials.append({"lambda": penalty, "rmse": float(np.sqrt(np.mean((pred - y[stop]) ** 2)))})
            chosen = min(trials, key=lambda c: (c["rmse"], -float("inf") if c["lambda"] is None else -c["lambda"]))
            pred = np.zeros(len(evaluate)) if chosen["lambda"] is None else ridge_predict(matrix, y, train, evaluate, chosen["lambda"])
            append(name, fold, evaluate, pred)
            fits.append({"config": name, "fold": fold, "trials": trials, "chosen": chosen})
    prediction = pd.concat(frames, ignore_index=True)
    pooled = data.pooled_validation_scores(prediction)
    folds = data.forecast_scores(prediction)
    scores = pooled.set_index("model_id")
    decisions = []
    for name in ("preopen_us", "preopen_all"):
        candidate, reference, earlier = scores.loc[name], scores.loc["DivSchedule"], scores.loc[name.replace("preopen", "close")]
        a = folds.loc[folds.model_id.eq(name)].set_index("fold")
        b = folds.loc[folds.model_id.eq("DivSchedule")].set_index("fold")
        wins = int((a.return_rmse < b.return_rmse - 1e-12).sum())
        decisions.append({"config": name, "gain_vs_schedule_pct": float(100 * (1 - candidate.return_rmse / reference.return_rmse)),
            "gain_vs_close_pct": float(100 * (1 - candidate.return_rmse / earlier.return_rmse)), "fold_wins": wins,
            "advance": bool(candidate.return_rmse <= .99 * reference.return_rmse and candidate.return_rmse <= .99 * earlier.return_rmse
                            and wins >= 2 and candidate.return_mae <= 1.01 * reference.return_mae)})
    for filename, frame in (("predictions.csv", prediction), ("comparison_pooled.csv", pooled), ("comparison_by_fold.csv", folds)):
        frame.to_csv(OUT / filename, index=False)
    (OUT / "fits.json").write_text(json.dumps(fits, indent=2), encoding="utf-8")
    (OUT / "decision.json").write_text(json.dumps(decisions, indent=2), encoding="utf-8")
    if hashes != {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in deps}:
        raise RuntimeError("Dependencies changed during screen")
    print(pooled[["model_id", "return_rmse", "return_mae"]].sort_values("return_rmse").to_string(index=False))
    print(json.dumps(decisions, indent=2))


if __name__ == "__main__":
    run()
