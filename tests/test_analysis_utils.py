"""Integrity failures and fold boundaries in the shared analysis helpers."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.analysis_utils import artifact_path, circular_block_interval, sha256_file, verify_hashes


class AnalysisUtilityTests(unittest.TestCase):
    def test_hashes_detect_changes_and_support_windows_recorded_paths(self):
        scratch = Path(__file__).resolve().parents[1] / "output/runs/tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="ai6124-analysis-", dir=scratch) as directory:
            root = Path(directory).resolve()
            self.assertTrue(root.is_relative_to(scratch.resolve()))
            path = root / "nested" / "table.csv"
            path.parent.mkdir()
            path.write_bytes(b"column\r\n1\r\n")
            digest = sha256_file(path)
            self.assertEqual(verify_hashes(root, {"nested\\table.csv": digest}), 1)
            path.write_bytes(b"column\n1\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_hashes(root, {"nested/table.csv": digest})
            with self.assertRaisesRegex(ValueError, "escapes"):
                artifact_path(root, "../outside.csv")

    def test_bootstrap_keeps_folds_separate_and_weights_each_date(self):
        # Any within-fold resample has mean (2*1 + 3*3)/5 = 2.2;
        # mixing dates across fold boundaries would produce varying means.
        result = circular_block_interval([np.ones(2), np.full(3, 3.0)], 4, resamples=100)
        np.testing.assert_allclose(result, [2.2, 2.2, 2.2], atol=0, rtol=0)

    def test_bootstrap_seed_and_invalid_input(self):
        chunks = [np.arange(7, dtype=float), np.array([-5.0, 1.0, 4.0])]
        first = circular_block_interval(chunks, 3, resamples=100, seed=42)
        self.assertEqual(first, circular_block_interval(chunks, 3, resamples=100, seed=42))
        self.assertAlmostEqual(first[0], np.concatenate(chunks).mean())
        for bad in ([], [np.array([])], [np.array([np.nan])], [np.ones((2, 2))]):
            with self.assertRaises(ValueError):
                circular_block_interval(bad, 3)
        with self.assertRaises(ValueError):
            circular_block_interval(chunks, 0)


if __name__ == "__main__":
    unittest.main()
