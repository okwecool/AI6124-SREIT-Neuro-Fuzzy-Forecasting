"""Guard against a one-day or daylight-saving leak in the overseas inputs."""

import unittest

import numpy as np
import pandas as pd

from src.project_eval import Dataset, VALIDATION_FOLDS
from src.v4.preopen_data import asof_values, build, us_frame


class PreopenTimeTests(unittest.TestCase):
    def test_new_york_dst_and_conservative_availability(self):
        source = us_frame("VNQ").set_index("session_date")
        self.assertEqual(
            source.loc["2024-03-08", "nominal_close_utc"], pd.Timestamp("2024-03-08 21:00:00Z")
        )
        self.assertEqual(
            source.loc["2024-03-11", "nominal_close_utc"], pd.Timestamp("2024-03-11 20:00:00Z")
        )
        self.assertEqual(
            source.loc["2024-03-11", "assumed_available_at"], pd.Timestamp("2024-03-11 22:00:00Z")
        )

    def test_same_date_us_close_unavailable_at_singapore_close(self):
        times = pd.Series(pd.to_datetime(["2024-03-11 10:00:00Z", "2024-03-12 00:00:00Z"]))
        joined = asof_values(times, us_frame("SPY"))
        self.assertEqual(joined.session_date.iloc[0], pd.Timestamp("2024-03-08"))
        self.assertEqual(joined.session_date.iloc[1], pd.Timestamp("2024-03-11"))

    def test_future_source_changes_cannot_change_earlier_features(self):
        source = us_frame("VNQ")
        times = pd.Series(pd.to_datetime(["2024-03-11 10:00:00Z", "2024-03-12 00:00:00Z"]))
        original = asof_values(times, source)
        changed = source.copy()
        changed.loc[changed.session_date.ge("2024-03-12"), "log_total_return_index"] += 100
        observed = asof_values(times, changed)
        np.testing.assert_array_equal(
            original.log_total_return_index, observed.log_total_return_index
        )

    def test_unchanged_session_means_zero_new_return(self):
        times = pd.Series(pd.to_datetime(["2024-03-09 00:00:00Z", "2024-03-11 00:00:00Z"]))
        frame = asof_values(times, us_frame("SPY"))
        self.assertEqual(frame.session_date.iloc[0], frame.session_date.iloc[1])
        self.assertEqual(frame.return_since_previous_cutoff.iloc[1], 0)

    def test_common_training_calendar_and_no_final_period_features(self):
        data = Dataset()
        inputs, y, alignment = build(data)
        for clock in inputs:
            for fold in VALIDATION_FOLDS:
                ids = np.concatenate([data.ids(fold, role) for role in ("train", "validation")])
                self.assertTrue(np.isfinite(inputs[clock][ids]).all())
                self.assertTrue(np.isfinite(y[ids]).all())
            self.assertTrue(np.isnan(inputs[clock][data.ids("final_holdout", "test")]).all())
        self.assertTrue((alignment.assumed_available_at <= alignment.cutoff).all())
        np.testing.assert_array_equal(inputs["close"][:931, :, :6], inputs["preopen"][:931, :, :6])


if __name__ == "__main__":
    unittest.main()
