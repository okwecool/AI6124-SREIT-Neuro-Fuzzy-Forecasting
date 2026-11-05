"""Causal split, fixed common model and mixture checks for V7."""

import unittest

import numpy as np
import torch

from src.project_eval import Dataset
from src.v5.shared_local import original_development_arrays
from src.v7.staged_experts import (chronological_parts, fitted_local, local_step,
                                   new_local, stage_arrays, train_common)


class StagedExpertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.data = Dataset()

    def test_chronological_splits_withhold_future_labels(self):
        for fold in ("validation_1", "validation_2", "validation_3"):
            for inner in (True, False):
                common, calibration, target = chronological_parts(self.data, fold, inner=inner)
                self.assertLess(common[-1], calibration[0])
                self.assertLess(calibration[-1], target[0])
                self.assertEqual(len(set(common) & set(calibration)), 0)
                self.assertEqual(len(set(calibration) & set(target)), 0)

    def test_outer_target_labels_are_not_read_by_scaler_or_fit(self):
        x, y = original_development_arrays(self.data)
        common, calibration, target = chronological_parts(self.data, "validation_1", inner=False)
        a = stage_arrays(x, y, common, calibration, target, target_labels=False)
        y[target] = np.nan
        b = stage_arrays(x, y, common, calibration, target, target_labels=False)
        for left, right in zip(a[:5], b[:5]):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertIsNone(a[5])
        self.assertIsNone(b[5])

    def test_local_step_changes_only_gate_and_exactly_mixes_experts(self):
        generator = torch.Generator().manual_seed(7)
        x = torch.randn(30, 20, 6, generator=generator)
        y = torch.randn(30, generator=generator)
        common, b, _ = train_common(x[:20], y[:20], 11)
        model = new_local(common, x[:20], "gaussian", 11)
        before = {name: p.detach().clone() for name, p in model.encoder.named_parameters()}
        centers = model.centers.detach().clone()
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.001)
        loss = local_step(model, optimizer, x[20:25], y[20:25], b)
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(torch.equal(centers, model.centers))
        for name, p in model.encoder.named_parameters():
            torch.testing.assert_close(before[name], p, rtol=0, atol=0)
        with torch.no_grad():
            pred, common_pred, d, diag = fitted_local(model, x[20:25], y[20:25], b, x[25:])
            q = diag["affine"].double()
            experts = torch.stack((q @ (b + d), q @ (b - d)), 1)
            torch.testing.assert_close(pred, (diag["alpha"].double() * experts).sum(1))
            self.assertEqual(tuple(diag["alpha"].shape), (5, 2))
            self.assertEqual(tuple(common_pred.shape), (5,))


if __name__ == "__main__":
    unittest.main()
