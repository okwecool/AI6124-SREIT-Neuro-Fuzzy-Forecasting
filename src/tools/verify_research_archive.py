"""Read-only verification of archived V3--V5 research; never train or fetch data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.analysis_utils import artifact_path, verify_hashes
from src.project_eval import Dataset

ROOT = Path(__file__).resolve().parents[2]
NEURAL_RUNS = {
    "output/v3/hybrid_training": 54,
    "output/v3/transfer_probe": 54,
    "output/v4/neural_preopen": 108,
    "output/v5/shared_local": 45,
}
SCREENS = ("output/v3/intraday_screen", "output/v4/preopen_screen")
ANALYSES = {
    "output/v4/neural_preopen/independent_audit.json": "src/v4/audit_improvement.py",
    "output/v5/shared_local/analysis_record.json": "src/v5/analyze_shared_local.py",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify(root: Path) -> dict:
    root = root.resolve()
    data = Dataset(root)
    hashes = 0
    gaussian_runs = 0
    recorded_runs = 0
    for relative in (*NEURAL_RUNS, *SCREENS):
        directory = artifact_path(root, relative)
        pre = read_json(directory / "pre_execution.json")
        hashes += verify_hashes(root, pre["dependency_sha256"])
        if relative in NEURAL_RUNS:
            expected = NEURAL_RUNS[relative]
            complete = read_json(directory / "complete.json")
            manifests = read_json(directory / "manifests.json")
            if complete["runs"] != expected or len(manifests) != expected:
                raise ValueError(f"Incomplete experiment: {relative}")
            hashes += verify_hashes(directory, {"predictions.csv": complete["prediction_sha256"]})
            if "manifests_sha256" in complete:
                hashes += verify_hashes(directory, {"manifests.json": complete["manifests_sha256"]})
            saved_runs = []
            for path in sorted(directory.rglob("manifest.json")):
                manifest = read_json(path)
                if "output_sha256" not in manifest:
                    # Transfer-source training records are not SRT evaluation runs.
                    continue
                hashes += verify_hashes(path.parent, manifest["output_sha256"])
                saved_runs.append(manifest)
                if (path.parent / "cases.csv").exists():
                    data.validate_fuzzy_diagnostics(
                        *[
                            pd.read_csv(path.parent / name)
                            for name in ("predictions.csv", "cases.csv", "rules.csv")
                        ]
                    )
                    gaussian_runs += 1
            if len(saved_runs) != expected:
                raise ValueError(f"Missing per-run manifests: {relative}")
            canonical = lambda record: json.dumps(record, sort_keys=True)
            if sorted(map(canonical, saved_runs)) != sorted(map(canonical, manifests)):
                raise ValueError(f"Aggregate/per-run manifests differ: {relative}")
            recorded_runs += expected

    for relative, source in ANALYSES.items():
        path = artifact_path(root, relative)
        record = read_json(path)
        hashes += verify_hashes(
            root,
            {
                source: record["analysis_source_sha256"],
                "src/analysis_utils.py": record["analysis_utils_sha256"],
            },
        )
        hashes += verify_hashes(root, record.get("reference_sha256", {}))
        hashes += verify_hashes(path.parent, record.get("output_sha256", {}))
    for symbol in ("spy", "vnq"):
        manifest = read_json(root / "data" / f"source_us_{symbol}_manifest.json")
        hashes += verify_hashes(root / "data", {manifest["file"]: manifest["sha256"]})

    return {
        "status": "pass",
        "experiments": len(NEURAL_RUNS) + len(SCREENS),
        "neural_runs": recorded_runs,
        "gaussian_runs": gaussian_runs,
        "file_hash_checks": hashes,
        "final_period_retrained_or_scored": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=ROOT, help="Verify another checkout without modifying it"
    )
    print(json.dumps(verify(parser.parse_args().root), indent=2))
