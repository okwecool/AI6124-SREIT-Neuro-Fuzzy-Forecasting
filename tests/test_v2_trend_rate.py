"""Timing, warm-up, and source checks for the fixed trend/rate screen."""

import unittest

import numpy as np
import pandas as pd

from data.fetch_sgs10y_source import RAW, parse_rows
from src.project_eval import Dataset, VALIDATION_FOLDS
from src.v2.run_residual_fuzzy import ResidualDataset
from src.v2.screen_trend_rate import SGS, design_matrices, rate_features, trend_features


class TrendRateTimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = ResidualDataset(Dataset().root)
        cls.sgs = pd.read_csv(SGS)

    def test_mas_snapshot_round_trips_to_saved_normalized_series(self) -> None:
        rows = parse_rows(RAW.read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 983)
        self.assertEqual(rows[0][0], "2021-11-01")
        self.assertEqual(rows[-1][0], "2025-09-30")
        self.assertEqual(rows, list(self.sgs.itertuples(index=False, name=None)))

    def test_trend_is_causal_and_uses_distribution_on_its_own_date(self) -> None:
        original = trend_features(self.data.daily)
        changed = self.data.daily.copy()
        changed.loc[300, "srt_close"] *= 1.05
        changed.loc[300, "srt_dividend_sgd_on_date"] += 0.01
        result = trend_features(changed)
        np.testing.assert_allclose(original.iloc[:300, 1:].to_numpy(),
                                   result.iloc[:300, 1:].to_numpy(), equal_nan=True)
        self.assertTrue(np.isnan(original.ma50_200_log_ratio.iloc[198]))
        self.assertTrue(np.isfinite(original.ma50_200_log_ratio.iloc[199]))
        self.assertTrue(np.isnan(original.ma50_200_log_ratio_change_5.iloc[203]))
        self.assertTrue(np.isfinite(original.ma50_200_log_ratio_change_5.iloc[204]))

    def test_sgs_alignment_uses_strictly_prior_source_dates(self) -> None:
        aligned = rate_features(self.data.daily, self.sgs)
        development = aligned.date.le("2025-09-30")
        self.assertTrue((aligned.loc[development, "sgs_source_date"] < aligned.loc[development, "date"]).all())
        self.assertTrue((aligned.loc[development, "sgs_stale_calendar_days"] >= 1).all())
        self.assertTrue(aligned.loc[~development, "sgs10y_change_5_srt_days_pp"].isna().all())
        source = pd.to_datetime(self.sgs.date)
        same_day = aligned.loc[development & aligned.date.isin(source)]
        self.assertFalse((same_day.sgs_source_date == same_day.date).any())

    def test_all_groups_use_same_train_ids_and_original_development_evaluation(self) -> None:
        matrices, valid = design_matrices(self.data, trend_features(self.data.daily),
                                         rate_features(self.data.daily, self.sgs))
        self.assertEqual(tuple(matrices), ("base", "trend", "rate", "trend_rate"))
        self.assertEqual([array.shape[1] for array in matrices.values()], [120, 122, 122, 124])
        self.assertEqual(int(valid.sum()), 766)
        self.assertEqual(int(valid[self.data.ids("validation_1", "train")].sum()), 388)
        for fold in VALIDATION_FOLDS:
            self.assertTrue(valid[self.data.ids(fold, "validation")].all())
        self.assertFalse(valid[self.data.ids("final_holdout", "test")].any())


if __name__ == "__main__":
    unittest.main()
