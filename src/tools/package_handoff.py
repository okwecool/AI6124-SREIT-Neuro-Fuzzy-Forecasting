"""Package the frozen course results, final deliverables, source and data."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "output/AI6124_Group1_Final_Handoff_2026-11-04.zip"
DIRECTORIES = (
    "data", "docs", "src", "tests", "output/development",
    "output/final", "output/runs/development", "output/runs/final",
)
SINGLE_FILES = (
    "requirements.txt",
    "output/zero_baseline_validation.csv",
    "output/zero_baseline_validation_scores.csv",
    "output/zero_baseline_validation_pooled.csv",
    "output/final_slides/AI6124_Group1_Final_Results_2026-11-04_v2.pptx",
)
EXCLUDED_SUFFIXES = {".pt", ".pth", ".ckpt", ".pyc"}


def members() -> list[Path]:
    paths = [ROOT / name for name in SINGLE_FILES]
    for directory in DIRECTORIES:
        paths.extend(path for path in (ROOT / directory).rglob("*") if path.is_file())
    result = []
    for path in sorted(set(paths)):
        if not path.exists():
            raise FileNotFoundError(f"Handoff member is missing: {path}")
        if path.name == "AI6124_Group1_Project_Starter.ipynb":
            continue  # Only the completed Notebook belongs in the final submission package.
        if path.suffix in EXCLUDED_SUFFIXES or any(
            part in {"__pycache__", ".ipynb_checkpoints"} for part in path.parts
        ):
            continue
        result.append(path)
    return result


def build(output: Path) -> dict:
    files = members()
    manifest = {"package_date": "2026-11-04", "purpose": "frozen final course handoff",
                "excluded": "model checkpoints and Python caches", "files": {}}
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing handoff: {output}")
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for source in files:
            relative = source.relative_to(ROOT).as_posix()
            data = source.read_bytes()
            archive.writestr(relative, data)
            manifest["files"][relative] = {"bytes": len(data),
                                            "sha256": sha256(data).hexdigest()}
        archive.writestr("HANDOFF_MANIFEST.json", json.dumps(manifest, indent=2))
    with ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Handoff ZIP failed CRC integrity check")
        found = set(archive.namelist())
        if found != set(manifest["files"]) | {"HANDOFF_MANIFEST.json"}:
            raise RuntimeError("Handoff member list is incomplete or duplicated")
        for relative, metadata in manifest["files"].items():
            if sha256(archive.read(relative)).hexdigest() != metadata["sha256"]:
                raise RuntimeError(f"Handoff SHA-256 check failed: {relative}")
    return {"path": str(output), "bytes": output.stat().st_size,
            "files": len(files), "status": "verified"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.out), indent=2))


if __name__ == "__main__":
    main()
