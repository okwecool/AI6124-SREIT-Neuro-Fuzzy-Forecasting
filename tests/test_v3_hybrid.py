"""Meaningful numerical and time-boundary checks before the V3 experiments."""

import unittest

import numpy as np
import torch

from src.project_eval import Dataset
from src.train import seed_everything
from src.v3.hybrid_training import (
    HybridHead,
    arrays,
    corrected_daily,
    inputs_and_residual,
    partition,
    ridge_coefficients,
    step,
)


class HybridTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.data = Dataset()

    def test_ridge_optimality_gradient_and_zero_residual(self):
        torch.manual_seed(7)
        phi = torch.randn(19, 8, dtype=torch.float64, requires_grad=True)
        y = torch.randn(19, dtype=torch.float64)
        beta = ridge_coefficients(phi, y, 0.1)
        gradient = phi.T @ (phi @ beta - y) / len(y) + 0.1 * beta
        torch.testing.assert_close(gradient, torch.zeros_like(beta), atol=1e-12, rtol=0)
        objective = ((phi @ beta - y) ** 2).mean() + 0.1 * beta.square().sum()
        objective.backward()
        self.assertTrue(torch.isfinite(phi.grad).all())
        self.assertGreater(float(phi.grad.abs().sum()), 0)
        zero = ridge_coefficients(phi.detach(), torch.zeros_like(y), 0.1)
        torch.testing.assert_close(zero, torch.zeros_like(zero), atol=0, rtol=0)

    def test_scaling_and_initialization_ignore_evaluation_data(self):
        x, y = inputs_and_residual(self.data, "gross")
        fit, evaluation, _, _ = partition(self.data, "validation_1")
        xt, yt, _, scale = arrays(x, y, fit, evaluation)
        altered_x, altered_y = x.copy(), y.copy()
        altered_x[evaluation] *= 100
        altered_y[evaluation] += 10
        xt2, yt2, _, scale2 = arrays(altered_x, altered_y, fit, evaluation)
        torch.testing.assert_close(xt, xt2)
        torch.testing.assert_close(yt, yt2)
        self.assertEqual(scale.y_std, scale2.y_std)
        np.testing.assert_array_equal(scale.y_inverse(np.zeros(3)), np.zeros(3))
        states = []
        for train_x in (xt, xt2):
            seed_everything(11)
            model = HybridHead("gaussian")
            model.initialize_from_training(train_x)
            states.append(model.state_dict())
        for key in states[0]:
            torch.testing.assert_close(states[0][key], states[1][key], atol=0, rtol=0)

    def test_corporate_action_features_are_causal_and_final_labels_hidden(self):
        original = corrected_daily(self.data.daily)
        changed = self.data.daily.copy()
        changed.loc[300, "srt_dividend_sgd_on_date"] += 0.1
        changed.loc[300, "srt_close"] *= 1.1
        np.testing.assert_allclose(original[:300], corrected_daily(changed)[:300], equal_nan=True)
        _, y = inputs_and_residual(self.data, "gross")
        ids = self.data.ids("final_holdout", "test")
        self.assertTrue(np.isnan(y[ids]).all())
        with self.assertRaises(ValueError):
            partition(self.data, "final_holdout")

    def test_all_heads_have_trainable_finite_paths_and_matched_initial_encoder(self):
        states = []
        torch.manual_seed(7)
        x, y = torch.randn(24, 20, 6), torch.randn(24)
        for head in ("linear", "gaussian", "gate"):
            seed_everything(11)
            model = HybridHead(head)
            model.initialize_from_training(x)
            states.append({k: v.clone() for k, v in model.encoder.state_dict().items()})
            before = model.projection.weight.detach().clone()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            loss = step(model, optimizer, x, y, 0.1)
            self.assertTrue(np.isfinite(loss))
            self.assertFalse(torch.equal(before, model.projection.weight))
        for key in states[0]:
            torch.testing.assert_close(states[0][key], states[1][key], atol=0, rtol=0)
            torch.testing.assert_close(states[0][key], states[2][key], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
