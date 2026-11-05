"""Known-history regime and trainable gate checks for V8."""

import unittest

import numpy as np
import torch

from src.models import build_model
from src.project_eval import Dataset
from src.v8.regime_supervision import train_epoch_state, volatility_regime


class RegimeSupervisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.data = Dataset()

    def test_regime_threshold_is_fit_only_and_uses_observed_feature(self):
        fold = self.data.prepare("validation_1", "all", inner_stop=True)
        threshold, train, stop = volatility_regime(self.data, fold.train_ids, fold.eval_ids)
        self.assertEqual(threshold, float(np.median(self.data.x[fold.train_ids, -1, 2])))
        np.testing.assert_array_equal(train.numpy(),
            (self.data.x[fold.train_ids, -1, 2] >= threshold).astype(np.float32))
        np.testing.assert_array_equal(stop.numpy(),
            (self.data.x[fold.eval_ids, -1, 2] >= threshold).astype(np.float32))
        original = self.data.x[fold.eval_ids, -1, 2].copy()
        self.data.x[fold.eval_ids, -1, 2] += 100
        try:
            changed, _, _ = volatility_regime(self.data, fold.train_ids, fold.eval_ids)
            self.assertEqual(threshold, changed)
        finally:
            self.data.x[fold.eval_ids, -1, 2] = original

    def test_auxiliary_loss_updates_existing_gate_without_new_inference_head(self):
        generator = torch.Generator().manual_seed(8)
        x = torch.randn(12, 20, 6, generator=generator)
        y = torch.randn(12, generator=generator)
        state = torch.tensor([0.0, 1.0] * 6)
        for model_id in ("Fuzzy2", "Gate2"):
            torch.manual_seed(7)
            model = build_model(model_id, 6)
            before = {name: value.detach().clone() for name, value in model.state_dict().items()}
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            losses = train_epoch_state(model, optimizer, x, y, state, seed=7, epoch=1)
            self.assertEqual(len(losses), 3)
            self.assertTrue(all(np.isfinite(value) for value in losses))
            with torch.no_grad():
                pred, diag = model(x, return_diagnostics=True)
            self.assertEqual(tuple(pred.shape), (12,))
            self.assertEqual(tuple(diag["alpha"].shape), (12, 2))
            self.assertEqual(before.keys(), model.state_dict().keys())
            parameter = "centers" if model_id == "Fuzzy2" else "gate.0.weight"
            self.assertFalse(torch.equal(before[parameter], model.state_dict()[parameter]))


if __name__ == "__main__":
    unittest.main()
