"""Create a compact, self-contained handoff package for Destin."""

from hashlib import sha256
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import json


root = Path(__file__).resolve().parents[2]
output = root / "output" / "AI6124_Group1_Early_Work_Handoff_2026-11-03.zip"
members = [
    "docs/README.md",
    "docs/README_DESTIN_HANDOFF.md",
    "docs/experiment_protocol_lstm_fuzzy_local_experts.md",
    "docs/literature_review_draft_en.md",
    "docs/lstm_fuzzy_local_experts_design_en.md",
    "docs/lstm_fuzzy_local_experts_design_zh.md",
    "docs/development_plan.md",
    "docs/design_review_2026-11-03_zh.md",
    "docs/pre_git_cleanup.md",
    "docs/AI6124_Group1_Project_Starter.ipynb",
    "src/project_eval.py",
    "src/__init__.py",
    "tests/test_project_eval.py",
    "src/tools/build_starter_notebook.py",
    "src/tools/verify_starter_notebook.py",
    "output/zero_baseline_validation.csv",
    "output/zero_baseline_validation_scores.csv",
    "output/zero_baseline_validation_pooled.csv",
]
members.extend(
    str(path.relative_to(root)).replace("\\", "/")
    for path in sorted((root / "data").rglob("*"))
    if path.is_file() and "__pycache__" not in path.parts
)

manifest = {"package_date": "2026-11-03", "files": {}}
output.parent.mkdir(parents=True, exist_ok=True)
with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
    for relative in members:
        source = root / relative
        data = source.read_bytes()
        archive.writestr(relative.replace("\\", "/"), data)
        manifest["files"][relative] = {"bytes": len(data), "sha256": sha256(data).hexdigest()}
    archive.writestr("HANDOFF_MANIFEST.json", json.dumps(manifest, indent=2))

with ZipFile(output) as archive:
    if archive.testzip() is not None:
        raise RuntimeError("Handoff ZIP failed integrity check")
print(f"{output} ({output.stat().st_size} bytes, {len(members) + 1} members)")
