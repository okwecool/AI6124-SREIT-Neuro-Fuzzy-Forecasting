"""A saved export without a checkpoint cannot stand in for a completed fit."""

import json
import tempfile
import unittest
from pathlib import Path

from src.train import _sha256, run_development
from src.tools.run_development_matrix import reusable_run


TEMP_ROOT = Path(__file__).resolve().parents[1] / "output/runs/tmp"


class ReproductionTest(unittest.TestCase):
    def test_missing_checkpoint_is_not_skipped(self) -> None:
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
            run_dir = Path(folder)
            manifest = {
                "status": "development",
                "train_source_sha256": _sha256(Path(run_development.__code__.co_filename)),
                "model_source_sha256": _sha256(Path(__file__).resolve().parents[1] / "src/models.py"),
            }
            (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            for name in ("predictions.csv", "cases.csv", "rules.csv"):
                (run_dir / name).write_text("saved", encoding="utf-8")
            self.assertFalse(reusable_run(run_dir, "Fuzzy2"))
            (run_dir / "checkpoint.pt").write_bytes(b"checkpoint")
            self.assertTrue(reusable_run(run_dir, "Fuzzy2"))
            (run_dir / "rules.csv").unlink()
            self.assertFalse(reusable_run(run_dir, "Fuzzy2"))


if __name__ == "__main__":
    unittest.main()
