"""Algebra, regularization, leakage boundaries and trainable paths for V5."""

import unittest

import numpy as np
import torch

from src.project_eval import Dataset
from src.train import seed_everything
from src.v3.hybrid_training import HybridHead, arrays, partition
from src.v5.shared_local import components, design, original_development_arrays, solve, step


class SharedLocalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.data = Dataset()

    def test_reparameterization_preserves_predictions_objective_and_gradient(self):
        torch.manual_seed(5)
        q = torch.randn(40, 4, dtype=torch.float64, requires_grad=True)
        logits = torch.randn(40, 2, dtype=torch.float64, requires_grad=True)
        alpha = logits.softmax(-1)
        y = torch.randn(40, dtype=torch.float64)
        independent = (alpha[:, :, None] * q[:, None, :]).reshape(40, 8)
        shared = torch.cat((q, (alpha[:, :1] - alpha[:, 1:]) * q), 1)
        beta = solve(independent, y, [0.1] * 8)
        theta = solve(shared, y, [0.2] * 8)
        torch.testing.assert_close(independent @ beta, shared @ theta, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(theta[:4], (beta[:4] + beta[4:]) / 2, rtol=1e-12, atol=1e-12)
        objectives = [
            ((independent @ beta - y) ** 2).mean() + 0.1 * beta.square().sum(),
            ((shared @ theta - y) ** 2).mean() + 0.2 * theta.square().sum(),
        ]
        torch.testing.assert_close(*objectives, rtol=1e-12, atol=1e-12)
        gradients = [torch.autograd.grad(v, (q, logits), retain_graph=True) for v in objectives]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-12)

    def test_balanced_independent_gate_has_double_single_head_penalty(self):
        torch.manual_seed(6)
        q, y = torch.randn(40, 4, dtype=torch.float64), torch.randn(40, dtype=torch.float64)
        independent = torch.cat((q / 2, q / 2), 1)
        expected = q @ solve(q, y, [0.2] * 4)
        actual = independent @ solve(independent, y, [0.1] * 8)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        self.assertGreater(float((actual - q @ solve(q, y, [0.1] * 4)).abs().max()), 1e-4)

    def test_stronger_local_penalty_reduces_local_coefficient_norm(self):
        torch.manual_seed(7)
        q = torch.randn(80, 4, dtype=torch.float64)
        local = torch.randn(80, 4, dtype=torch.float64)
        phi = torch.cat((q, local), 1)
        y = q[:, 0] + local[:, 0] + 0.1 * torch.randn(80, dtype=torch.float64)
        low = solve(phi, y, [0.1] * 4 + [0.2] * 4)
        high = solve(phi, y, [0.1] * 4 + [1.0] * 4)
        self.assertLess(float(high[4:].norm()), float(low[4:].norm()))

    def test_original_input_target_values_and_final_boundary(self):
        x, y = original_development_arrays(self.data)
        keep = np.isfinite(y)
        np.testing.assert_array_equal(x[keep], self.data.x[keep])
        np.testing.assert_array_equal(y[keep], self.data.y[keep])
        final = self.data.ids("final_holdout", "test")
        self.assertTrue(np.isnan(x[final]).all() and np.isnan(y[final]).all())
        with self.assertRaises(ValueError):
            partition(self.data, "final_holdout")

    def test_outer_evaluation_cannot_change_scaling_or_initialization(self):
        x, y = original_development_arrays(self.data)
        _, _, train, evaluation = partition(self.data, "validation_1")
        xa, ya, _, sa = arrays(x, y, train, evaluation)
        x[evaluation] *= 1000
        y[evaluation] += 10
        xb, yb, _, sb = arrays(x, y, train, evaluation)
        torch.testing.assert_close(xa, xb, rtol=0, atol=0)
        torch.testing.assert_close(ya, yb, rtol=0, atol=0)
        self.assertEqual(sa.y_std, sb.y_std)
        self.assertEqual(sa.y_mean, 0.0)
        states = []
        for xt in (xa, xb):
            seed_everything(11)
            model = HybridHead("gaussian")
            model.initialize_from_training(xt)
            states.append(model.state_dict())
        for key in states[0]:
            torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)

    def test_all_arms_share_encoder_and_have_finite_gradient_and_mixture(self):
        configs = [
            dict(head="linear", basis="shared", lambda_common=0.1),
            dict(head="gaussian", basis="independent", lambda_expert=0.1),
            dict(head="gaussian", basis="shared", lambda_common=0.1, lambda_local=0.2),
            dict(head="gaussian", basis="shared", lambda_common=0.1, lambda_local=1.0),
            dict(head="gate", basis="shared", lambda_common=0.1, lambda_local=1.0),
        ]
        torch.manual_seed(8)
        x, y = torch.randn(40, 20, 6), torch.randn(40)
        states = []
        for config in configs:
            seed_everything(11)
            model = HybridHead(config["head"])
            model.initialize_from_training(x)
            states.append({k: v.clone() for k, v in model.encoder.state_dict().items()})
            before = model.projection.weight.detach().clone()
            loss = step(model, torch.optim.AdamW(model.parameters(), lr=0.001), x, y, config)
            self.assertTrue(np.isfinite(loss))
            self.assertFalse(torch.equal(before, model.projection.weight))
            phi, diag, penalties = design(model, x, config)
            beta = solve(phi, y, penalties)
            common, local, experts = components(beta, diag, config)
            torch.testing.assert_close(common + local, phi.double() @ beta, rtol=1e-5, atol=1e-7)
            if experts is not None:
                torch.testing.assert_close(
                    (diag["alpha"].double() * experts).sum(1), common + local, rtol=1e-5, atol=1e-7
                )
        for state in states[1:]:
            for key in states[0]:
                torch.testing.assert_close(states[0][key], state[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
