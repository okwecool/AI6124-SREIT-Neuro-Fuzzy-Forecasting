"""One-batch checks of the model contract and the raw export identity."""

import unittest

import numpy as np
import torch

from src.export_contract import fuzzy_batch_tables
from src.models import build_model, parameter_count
from src.project_eval import Dataset


class ModelContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.data = Dataset()
        cls.split = cls.data.prepare("validation_1", "all", inner_stop=True)

    def test_shapes_finite_gradients_and_weighted_experts(self):
        torch.manual_seed(11)
        x = torch.from_numpy(self.split.x_train[:16])
        y = torch.from_numpy(self.split.y_train[:16])
        for model_id in ("LSTM32", "LSTM3", "Fuzzy1", "Fuzzy2", "Fuzzy3"):
            with self.subTest(model_id=model_id):
                model = build_model(model_id, 6)
                self.assertGreater(parameter_count(model), 0)
                if model_id.startswith("Fuzzy"):
                    prediction, diag = model(x, return_diagnostics=True)
                    k = int(model_id[-1])
                    self.assertEqual(diag["z"].shape, (16, 3))
                    self.assertEqual(diag["alpha"].shape, (16, k))
                    self.assertTrue(torch.allclose(diag["alpha"].sum(1), torch.ones(16), atol=1e-6))
                    self.assertTrue(torch.allclose(prediction, (diag["alpha"] * diag["expert_predictions"]).sum(1), atol=1e-6))
                    self.assertTrue(torch.allclose(diag["log_membership"], -diag["distance2"] / 2))
                else:
                    prediction = model(x)
                self.assertEqual(prediction.shape, (16,))
                loss = torch.mean((prediction - y) ** 2)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_fuzzy_raw_csv_round_trip(self):
        torch.manual_seed(11)
        model = build_model("Fuzzy2", 6)
        outer = self.data.prepare("validation_1", "all")
        x = torch.from_numpy(outer.x_eval[:4])
        with torch.no_grad():
            prediction, diag = model(x, return_diagnostics=True)
        predictions, cases, rules = fuzzy_batch_tables(
            model_id="Fuzzy2", source_group="all", fold="validation_1", seed=11,
            window_ids=outer.eval_ids[:4], prediction=prediction,
            diagnostics=diag, scaler=outer.scaler,
        )
        self.data.validate_fuzzy_diagnostics(predictions, cases, rules, complete=False)
        np.testing.assert_allclose(
            predictions.y_pred.to_numpy(),
            rules.groupby("window_id").apply(lambda g: np.sum(g.alpha * g.expert_raw), include_groups=False).to_numpy(),
            atol=1e-6,
        )

    def test_source_group_shapes_and_fuzzy_initialization(self):
        for input_size in (3, 5, 6):
            with self.subTest(input_size=input_size):
                model = build_model("Fuzzy2", input_size)
                x = torch.zeros(2, 20, input_size)
                prediction, diag = model(x, return_diagnostics=True)
                self.assertEqual(prediction.shape, (2,))
                self.assertEqual(diag["alpha"].shape, (2, 2))
                np.testing.assert_allclose(model.centers.detach().numpy(),
                                           [[-.25, 0, 0], [.25, 0, 0]])
                scales = .1 + 1.9 * torch.sigmoid(model.scale_logits)
                np.testing.assert_allclose(scales.detach().numpy(), np.ones((2, 3)), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
