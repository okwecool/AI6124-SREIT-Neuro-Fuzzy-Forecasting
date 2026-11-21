"""Timing checks for the analyst-guided multi-horizon development screen."""

import unittest

import numpy as np
import pandas as pd

from src.project_eval import Dataset, VALIDATION_FOLDS
from src.v11.horizon_screen import fold_ids, labels_and_momentum


class HorizonScreenTimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Dataset()
        cls.cutoff = pd.Timestamp("2025-09-30")

    def test_all_fit_labels_end_before_the_next_forecast_period(self):
        for horizon in (20, 60):
            _, _, eligible, label_end = labels_and_momentum(self.data, horizon, self.cutoff)
            for fold in VALIDATION_FOLDS:
                fit, stop, train, evaluation = fold_ids(self.data, eligible, label_end, fold)
                stop_start = np.datetime64(self.data.index.forecast_date.iloc[stop[0]])
                eval_start = np.datetime64(self.data.index.forecast_date.iloc[evaluation[0]])
                self.assertLess(label_end[fit].max(), stop_start)
                self.assertLess(label_end[train].max(), eval_start)
                self.assertLessEqual(label_end[evaluation].max(), np.datetime64(self.cutoff))
                self.assertTrue(np.isfinite(self.data.x[evaluation]).all())

    def test_future_close_changes_label_but_not_past_momentum(self):
        horizon = 20
        original_y, original_momentum, eligible, _ = labels_and_momentum(self.data, horizon, self.cutoff)
        window_id = int(self.data.ids("validation_1", "validation")[10])
        self.assertTrue(eligible[window_id])
        future_row = int(self.data.index.daily_end_row.iloc[window_id]) + horizon
        daily_copy = self.data.daily
        try:
            self.data.daily = daily_copy.copy()
            self.data.daily.loc[future_row, "srt_close"] *= 1.1
            changed_y, changed_momentum, _, _ = labels_and_momentum(self.data, horizon, self.cutoff)
        finally:
            self.data.daily = daily_copy
        self.assertNotAlmostEqual(original_y[window_id], changed_y[window_id])
        self.assertAlmostEqual(original_momentum[window_id], changed_momentum[window_id])


if __name__ == "__main__":
    unittest.main()
