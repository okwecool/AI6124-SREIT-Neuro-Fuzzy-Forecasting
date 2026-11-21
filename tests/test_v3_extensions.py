"""Information-boundary and transfer mapping checks for the named extensions."""

import unittest

import numpy as np
import torch

from src.project_eval import Dataset
from src.train import seed_everything
from src.v3.hybrid_training import HybridHead
from src.v3.screen_intraday import targets
from src.v3.transfer_probe import source_windows, source_model, initialized


class ExtensionTests(unittest.TestCase):
    def test_intraday_label_and_previous_session_use_distinct_dates(self):
        data = Dataset()
        y, previous = targets(data)
        row = data.index.iloc[int(data.ids("validation_1", "validation")[0])]
        daily = data.daily.set_index("date")
        actual = daily.loc[row.target_date]
        known = daily.loc[row.forecast_date]
        self.assertAlmostEqual(y[row.window_id], np.log(actual.srt_close / actual.srt_open))
        self.assertAlmostEqual(previous[row.window_id], np.log(known.srt_close / known.srt_open))
        self.assertTrue(np.isnan(y[data.ids("final_holdout", "test")]).all())

    def test_source_windows_end_before_target_history(self):
        x, y, dates = source_windows()
        self.assertEqual(x.shape, (len(y), 20, 3))
        self.assertTrue(np.isfinite(x).all() and np.isfinite(y).all())
        self.assertLessEqual(max(dates), "2021-09-30")
        self.assertLess(max(dates), str(Dataset().daily.date.min().date()))

    def test_transfer_replaces_only_shared_input_columns_and_freezes_encoder(self):
        torch.set_num_threads(1)
        seed_everything(11)
        baseline = HybridHead("gaussian")
        source = source_model(11)
        with torch.no_grad():
            source.encoder.lstm.weight_ih_l0.add_(0.01)
        x = torch.randn(30, 20, 6)
        model = initialized("gaussian", 11, source, x)
        torch.testing.assert_close(
            model.encoder.lstm.weight_ih_l0[:, :3], source.encoder.lstm.weight_ih_l0
        )
        torch.testing.assert_close(
            model.encoder.lstm.weight_ih_l0[:, 3:], baseline.encoder.lstm.weight_ih_l0[:, 3:]
        )
        self.assertFalse(any(p.requires_grad for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
