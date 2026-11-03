"""Contract checks for the shared forecasting evaluation layer."""

import unittest

import numpy as np
import pandas as pd

from src.project_eval import Dataset, VALIDATION_FOLDS, FUZZY_CASE_COLUMNS, FUZZY_RULE_COLUMNS


class EvaluationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Dataset()

    def test_outer_and_inner_folds_respect_target_chronology(self):
        for fold in VALIDATION_FOLDS:
            for inner in (False, True):
                split = self.data.prepare(fold, "all", inner_stop=inner)
                train_dates = self.data.index.loc[split.train_ids, "target_date"]
                eval_dates = self.data.index.loc[split.eval_ids, "target_date"]
                self.assertLess(train_dates.max(), eval_dates.min())
                self.assertEqual(split.x_train.shape[1:], (20, 6))
                self.assertEqual(split.x_eval.shape[1:], (20, 6))
                self.assertAlmostEqual(float(split.y_train.mean()), 0, places=5)

    def test_validation_data_never_affects_scaler(self):
        split = self.data.prepare("validation_1", "srt_sora")
        original_mean = split.scaler.x_mean.copy()
        first_eval = int(split.eval_ids[0])
        original = self.data.x[first_eval].copy()
        try:
            self.data.x[first_eval] = 1_000_000
            changed = self.data.prepare("validation_1", "srt_sora")
            np.testing.assert_allclose(changed.scaler.x_mean, original_mean)
        finally:
            self.data.x[first_eval] = original

    def test_prediction_contract_rejects_wrong_fold_ids(self):
        predictions = self.data.zero_predictions()
        self.assertEqual(len(predictions), 378)
        bad = predictions.copy()
        bad.loc[0, "window_id"] = int(self.data.ids("final_holdout", "test")[0])
        with self.assertRaises(ValueError):
            self.data.validate_predictions(bad)

    def test_zero_forecast_is_unchanged_price(self):
        self.assertTrue((self.data.zero_predictions().rule_count == 0).all())
        frame = self.data.validate_predictions(self.data.zero_predictions())
        np.testing.assert_allclose(frame.close_pred, frame.close_t)
        scores = self.data.forecast_scores(self.data.zero_predictions())
        self.assertEqual(set(scores.fold), set(VALIDATION_FOLDS))
        self.assertTrue((scores.return_rmse > 0).all())
        pooled = self.data.pooled_validation_scores(self.data.zero_predictions()).iloc[0]
        expected = np.sqrt(np.mean(frame.y_true.to_numpy() ** 2))
        self.assertAlmostEqual(pooled.return_rmse, expected)
        self.assertEqual(pooled.n, 378)

    def test_legacy_zero_only_migration_and_rule_count_validation(self):
        legacy = self.data.zero_predictions().drop(columns="rule_count")
        self.assertTrue((self.data.validate_predictions(legacy).rule_count == 0).all())
        invalid = legacy.assign(model_id="Fuzzy2")
        with self.assertRaisesRegex(ValueError, "rule_count"):
            self.data.validate_predictions(invalid)
        invalid = self.data.zero_predictions().assign(model_id="Fuzzy2")
        with self.assertRaisesRegex(ValueError, "rule_count"):
            self.data.validate_predictions(invalid)

    def test_fuzzy_full_keys_and_mixture_identity(self):
        base = self.data.zero_predictions().query("fold == 'validation_1'").head(2).copy()
        base["model_id"] = "Fuzzy2"
        base["rule_count"] = 2
        base["source_group"] = "srt"
        base["y_pred"] = 0.02
        other = base.copy()
        other["source_group"] = "all"
        predictions = pd.concat([base, other], ignore_index=True)
        case_rows, rule_rows = [], []
        for row in predictions.itertuples(index=False):
            key = (row.model_id, row.source_group, row.rule_count, row.fold, row.seed, row.window_id)
            case_rows.append((*key, 0.0, 0.0, 0.0, 0.02))
            rule_rows.extend([(*key, 1, 0.5, 0.01, 0.0, 0.0), (*key, 2, 0.5, 0.03, 0.0, 0.0)])
        cases = pd.DataFrame(case_rows, columns=FUZZY_CASE_COLUMNS)
        rules = pd.DataFrame(rule_rows, columns=FUZZY_RULE_COLUMNS)
        self.data.validate_fuzzy_diagnostics(predictions, cases, rules, complete=False)
        bad = rules.copy()
        bad.loc[0, "expert_raw"] = 0.2
        with self.assertRaisesRegex(ValueError, "weighted experts"):
            self.data.validate_fuzzy_diagnostics(predictions, cases, bad, complete=False)
        bad = rules.iloc[1:].copy()
        with self.assertRaisesRegex(ValueError, "number or IDs"):
            self.data.validate_fuzzy_diagnostics(predictions, cases, bad, complete=False)

    def test_missing_open_omits_two_adjacent_intervals(self):
        # Model a one-day data-source gap inside a validation period.
        copied = Dataset()
        predictions = copied.zero_predictions()
        before = copied.portfolio_scores(predictions)
        target_day = copied.index.loc[int(copied.ids("validation_1", "validation")[40]), "target_date"]
        copied.daily.loc[copied.daily.date == target_day, "srt_open"] = np.nan
        after = copied.portfolio_scores(predictions)
        baseline = before.loc[(before.fold == "validation_1") & (before.strategy == "always_long")].iloc[0]
        repaired = after.loc[(after.fold == "validation_1") & (after.strategy == "always_long")].iloc[0]
        self.assertEqual(repaired.omitted_intervals - baseline.omitted_intervals, 2)
        self.assertEqual(repaired.eligible_intervals, baseline.eligible_intervals - 2)

    def test_portfolio_does_not_cross_validation_boundary(self):
        result = self.data.portfolio_scores(self.data.zero_predictions())
        for fold, expected in (("validation_1", 125), ("validation_2", 123), ("validation_3", 127)):
            subset = result.loc[(result.fold == fold) & (result.strategy == "always_long")].iloc[0]
            self.assertEqual(subset.eligible_intervals, expected)
            self.assertEqual(subset.omitted_intervals, 1)


if __name__ == "__main__":
    unittest.main()
