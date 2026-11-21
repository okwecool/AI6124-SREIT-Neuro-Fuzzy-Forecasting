"""Matched representation and direct-consequent identity checks."""

import unittest

import torch

from src.v4.neural_preopen import construct, predict, train_step


class TimedNeuralTests(unittest.TestCase):
    def test_direct_consequents_preserve_three_dimensional_antecedents(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)
        x, y = torch.randn(24, 20, 8), torch.randn(24)
        model, optimizer = construct("gaussian", "direct_us", 11, x)
        train_step(model, optimizer, x, y)
        prediction, beta, diag = predict(model, x, y, x)
        self.assertEqual(diag["z"].shape, (24, 3))
        self.assertEqual(diag["alpha"].shape, (24, 2))
        self.assertEqual(diag["affine"].shape, (24, 6))
        expert = diag["affine"] @ beta.reshape(2, 6).T
        torch.testing.assert_close(prediction, (diag["alpha"] * expert).sum(-1))
        torch.testing.assert_close(diag["affine"][:, 3:5], x[:, -1, 6:])

    def test_topology_ablation_starts_with_identical_encoder_and_gaussians(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)
        x = torch.randn(24, 20, 8)
        a, _ = construct("gaussian", "latent", 11, x)
        b, _ = construct("gaussian", "direct_us", 11, x)
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key], atol=0, rtol=0)

    def test_all_matching_control_heads_have_finite_training_paths(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)
        x, y = torch.randn(24, 20, 8), torch.randn(24)
        for topology in ("latent", "direct_us"):
            for head in ("linear", "gaussian", "gate"):
                model, optimizer = construct(head, topology, 11, x)
                self.assertTrue(torch.isfinite(torch.tensor(train_step(model, optimizer, x, y))))
                prediction, _, _ = predict(model, x, y, x)
                self.assertTrue(torch.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
