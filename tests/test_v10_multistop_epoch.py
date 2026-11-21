"""Chronological stopping blocks and V1 outer-refit replay for V10."""

import unittest

import numpy as np
import pandas as pd
import torch

from src.project_eval import Dataset
from src.v10.multistop_epoch import forward_blocks, outer_refit


class ForwardStopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.data = Dataset()

    def test_latest_block_is_original_v1_stop_and_both_are_forward(self):
        for fold in ("validation_1", "validation_2", "validation_3"):
            blocks = forward_blocks(self.data, fold)
            original = self.data.prepare(fold, "all", inner_stop=True)
            np.testing.assert_array_equal(blocks[1][1], original.train_ids)
            np.testing.assert_array_equal(blocks[1][2], original.eval_ids)
            self.assertLess(blocks[0][1][-1], blocks[0][2][0])
            self.assertLess(blocks[0][2][-1], blocks[1][2][0])

    def test_fixed_epoch_refit_replays_saved_v1_fuzzy_prediction(self):
        current, cases, rules, training = outer_refit(self.data, "Fuzzy2", "validation_1", 11, 6)
        historical = pd.read_csv("output/development/predictions.csv")
        historical = historical.loc[historical.model_id.eq("Fuzzy2") &
                                    historical.source_group.eq("all") &
                                    historical.fold.eq("validation_1") &
                                    historical.seed.eq(11)]
        compare = current.merge(historical, on=["fold", "seed", "window_id"],
                                suffixes=("_new", "_old"), validate="one_to_one")
        self.assertEqual(len(compare), 126)
        self.assertLess(float(np.max(np.abs(compare.y_pred_new - compare.y_pred_old))), 1e-7)
        self.assertEqual(len(training), 6)
        self.assertEqual(len(cases), 126)
        self.assertEqual(len(rules), 252)


if __name__ == "__main__":
    unittest.main()
