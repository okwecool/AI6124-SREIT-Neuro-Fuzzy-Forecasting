"""Determinism and sealed-holdout checks for the development runner."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.project_eval import Dataset
from src.train import run_development


class DevelopmentRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.data = Dataset()

    def test_same_seed_repeats_prediction_and_manifest(self):
        local_tmp = Path("output/runs/tmp")
        local_tmp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=local_tmp) as first, tempfile.TemporaryDirectory(dir=local_tmp) as second:
            manifests = []
            forecasts = []
            for location in (first, second):
                manifest = run_development(
                    self.data, "LSTM32", "srt", "validation_1", 11,
                    Path(location), max_epochs=2, patience=2)
                manifests.append(manifest)
                forecasts.append(pd.read_csv(manifest["prediction_file"]))
            self.assertEqual(manifests[0]["best_epoch"], manifests[1]["best_epoch"])
            self.assertEqual(manifests[0]["dataset_sha256"], manifests[1]["dataset_sha256"])
            np.testing.assert_array_equal(forecasts[0].y_pred, forecasts[1].y_pred)
            self.assertEqual(len(forecasts[0]), 126)

    def test_development_entry_point_rejects_final_holdout(self):
        with self.assertRaisesRegex(ValueError, "sealed"):
            run_development(self.data, "LSTM32", "all", "final_holdout", 11, Path("unused"))


if __name__ == "__main__":
    unittest.main()
