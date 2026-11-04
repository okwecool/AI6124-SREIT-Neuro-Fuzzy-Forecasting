"""Run the frozen Phase 3 matrix without opening the final holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.project_eval import Dataset, VALIDATION_FOLDS
from src.train import SEEDS, _sha256, run_development


def matrix(include_gate: bool) -> list[tuple[str, str, str, int]]:
    specs = []
    for fold in VALIDATION_FOLDS:
        for seed in SEEDS:
            specs.extend((model_id, "all", fold, seed)
                         for model_id in ("LSTM32", "LSTM3", "Fuzzy2"))
            specs.extend(("Fuzzy2", group, fold, seed)
                         for group in ("srt", "srt_sora"))
            if include_gate:
                specs.append(("Gate2", "all", fold, seed))
        specs.append(("Fuzzy3", "all", fold, 11))
    if len(specs) != (57 if include_gate else 48):
        raise AssertionError("Unexpected run count")
    return specs


def reusable_run(run_dir: Path, model_id: str) -> bool:
    manifest_file = run_dir / "manifest.json"
    if not manifest_file.exists():
        return False
    previous = json.loads(manifest_file.read_text(encoding="utf-8"))
    required = [run_dir / "predictions.csv", run_dir / "checkpoint.pt"]
    if model_id.startswith("Fuzzy"):
        required.extend((run_dir / "cases.csv", run_dir / "rules.csv"))
    return (previous.get("status") == "development" and
            previous.get("train_source_sha256") == _sha256(Path(run_development.__code__.co_filename)) and
            previous.get("model_source_sha256") == _sha256(Path(__file__).resolve().parents[1] / "models.py") and
            all(path.exists() for path in required))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("output/runs/reproduction/development"))
    parser.add_argument("--include-gate", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    specs = matrix(args.include_gate)
    plan = {"core_runs": 48, "optional_gate_runs": 9 if args.include_gate else 0,
            "k1_reference": "LSTM3 at seed 11 is functionally Fuzzy1",
            "specs": [dict(model_id=model, source_group=group, fold=fold, seed=seed)
                      for model, group, fold, seed in specs]}
    (args.out / "matrix_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    data = Dataset()
    for number, (model_id, group, fold, seed) in enumerate(specs, start=1):
        run_dir = args.out / fold / group / model_id / f"seed_{seed}"
        manifest_file = run_dir / "manifest.json"
        if manifest_file.exists():
            if reusable_run(run_dir, model_id):
                print(f"[{number}/{len(specs)}] verified existing {model_id} {group} {fold} seed={seed}", flush=True)
                continue
            raise ValueError(f"Incomplete or stale run at {run_dir}; use a new --out reproduction directory")
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError(f"Unmanifested files at {run_dir}; use a new --out reproduction directory")
        result = run_development(data, model_id, group, fold, seed, args.out, device=args.device)
        print(f"[{number}/{len(specs)}] {model_id} {group} {fold} seed={seed} "
              f"best={result['best_epoch']} inner={result['inner_seconds']:.2f}s "
              f"refit={result['refit_seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
