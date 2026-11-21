"""Causal timing and identity checks for the bounded V2 follow-up experiments."""

import unittest

import numpy as np
import pandas as pd

from src.project_eval import Dataset, VALIDATION_FOLDS
from src.v2.run_residual_fuzzy import ResidualDataset
from src.v2.screen_clr import aligned_clr_windows
from src.v2.screen_distributions import EVENTS, announced_adjustments
from src.v2.screen_ohlcv import daily_feature_groups
from src.v2.screen_weekly_target import make_target, split_fold


class FollowupTimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = Dataset()

    def test_ohlcv_features_do_not_change_before_a_perturbed_future_row(self) -> None:
        original = daily_feature_groups(self.data.daily)
        altered = self.data.daily.copy()
        altered.loc[300, "srt_volume"] *= 100
        altered.loc[300, "srt_open"] *= 1.01
        changed = daily_feature_groups(altered)
        for group in ("price", "volume"):
            np.testing.assert_allclose(original[group][:300], changed[group][:300], equal_nan=True)

    def test_weekly_labels_are_purged_before_each_evaluation_forecast(self) -> None:
        cutoff = pd.Timestamp("2025-09-30")
        _, valid, end_dates = make_target(self.data, cutoff)
        for fold in VALIDATION_FOLDS:
            inner_fit, inner_stop, outer_fit, evaluation = split_fold(self.data, valid, end_dates, fold)
            self.assertLess(end_dates[inner_fit].max(), np.datetime64(self.data.index.forecast_date.iloc[inner_stop[0]]))
            self.assertLess(end_dates[outer_fit].max(), np.datetime64(self.data.index.forecast_date.iloc[evaluation[0]]))
            self.assertTrue((end_dates[evaluation] <= np.datetime64(cutoff)).all())

    def test_announced_adjustment_rejects_information_not_yet_available(self) -> None:
        announcements = pd.read_csv(EVENTS)
        adjustment = announced_adjustments(self.data, announcements)
        self.assertEqual(int(adjustment.ex_date.notna().sum()), 3)
        first = adjustment.loc[adjustment.ex_date.notna()].iloc[0]
        self.assertAlmostEqual(float(first.adjustment), float(np.log1p(-first.distribution_sgd / first.close_t)))
        late = announcements.copy()
        late.loc[0, "announcement_date"] = "2024-07-18"
        with self.assertRaises(ValueError):
            announced_adjustments(self.data, late)

    def test_residual_label_reconstructs_raw_label_on_all_development_evaluations(self) -> None:
        residual = ResidualDataset(self.data.root)
        for fold in VALIDATION_FOLDS:
            ids = self.data.ids(fold, "validation")
            reconstructed = residual.y[ids].astype(float) + residual.realized_adjustment[ids]
            np.testing.assert_allclose(reconstructed, self.data.y[ids], atol=2e-9)

    def test_clr_feature_is_the_same_day_observed_close_return(self) -> None:
        x, aligned = aligned_clr_windows(self.data)
        self.assertEqual(len(aligned), 971)
        self.assertFalse(aligned.clr_close.isna().any())
        for fold in VALIDATION_FOLDS:
            window_id = int(self.data.ids(fold, "validation")[0])
            end = int(self.data.index.daily_end_row.iloc[window_id])
            self.assertEqual(aligned.date.iloc[end], self.data.index.forecast_date.iloc[window_id])
            expected = np.log(aligned.clr_close.iloc[end] / aligned.clr_close.iloc[end - 1])
            self.assertAlmostEqual(float(x[window_id, -1, -1]), float(expected), places=7)


if __name__ == "__main__":
    unittest.main()
