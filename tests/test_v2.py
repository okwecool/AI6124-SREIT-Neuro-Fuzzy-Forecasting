"""Checks for the new development-only baseline and local-loss contracts."""

import unittest

import numpy as np
import torch

from src.project_eval import Dataset
from src.v2.baselines import fit_ridge, predict_ridge, summarize
from src.v2.local_loss import local_expert_loss, loss_components
from src.v2.run_local import run_one


class V2ContractTests(unittest.TestCase):
    def test_h1_summary_includes_deterministic_zero_baseline(self) -> None:
        data = Dataset()
        dummy = data.zero_predictions().copy()
        dummy["model_id"] = "DummyReference"
        pooled, _, _, _ = summarize(data, dummy)
        self.assertIn("Z0", set(pooled.model_id))

    def test_ridge_does_not_penalize_constant_intercept(self) -> None:
        rng = np.random.default_rng(6124)
        x = rng.normal(size=(30, 20, 6))
        y = np.full(30, 4.0)
        fitted = fit_ridge(x, y, alpha=10.0)
        self.assertTrue(np.allclose(predict_ridge(x, fitted), 4.0, atol=1e-10))
        self.assertAlmostEqual(float(np.linalg.norm(fitted[0])), 0.0, places=10)

    def test_local_loss_detaches_weights_but_updates_experts(self) -> None:
        alpha = torch.tensor([[0.4, 0.6], [0.7, 0.3]], requires_grad=True)
        experts = torch.tensor([[0.2, -0.1], [0.3, 0.1]], requires_grad=True)
        target = torch.tensor([0.0, 0.2])
        local = local_expert_loss(alpha, experts, target)
        local.backward()
        self.assertIsNone(alpha.grad)
        self.assertIsNotNone(experts.grad)
        self.assertTrue(torch.isfinite(experts.grad).all())
        self.assertGreater(float(experts.grad.abs().sum()), 0.0)

    def test_mixed_loss_keeps_gate_gradient(self) -> None:
        alpha = torch.tensor([[0.4, 0.6], [0.7, 0.3]], requires_grad=True)
        experts = torch.tensor([[0.2, -0.1], [0.3, 0.1]], requires_grad=True)
        target = torch.tensor([0.0, 0.2])
        prediction = (alpha * experts).sum(dim=1)
        total, mixed, local = loss_components(prediction, experts, alpha, target,
                                               local_weight=0.1)
        self.assertAlmostEqual(float(total.detach()), float((mixed + 0.1 * local).detach()), places=7)
        total.backward()
        self.assertGreater(float(alpha.grad.abs().sum()), 0.0)

    def test_h3_rejects_final_holdout(self) -> None:
        data = Dataset()
        with self.assertRaises(ValueError):
            run_one(data, "final_holdout", 11, data.root / "output/v2/forbidden")


if __name__ == "__main__":
    unittest.main()
