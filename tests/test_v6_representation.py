"""Boundary checks for the training-only SRT-history auxiliary task."""

import unittest

import numpy as np
import torch

from src.project_eval import Dataset
from src.v6.representation_pretrain import (make_model, masked_history,
                                            pretrain_representation)


class RepresentationPretrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = Dataset()

    def test_mask_hides_only_historical_srt_channels_and_preserves_input(self) -> None:
        inner = self.data.prepare("validation_1", "all", inner_stop=True)
        original = torch.from_numpy(inner.x_train[:3]).clone()
        masked, target = masked_history(original)
        self.assertTrue(torch.equal(original[:, -5:, 0], target))
        self.assertTrue(torch.equal(masked[:, :-5], original[:, :-5]))
        self.assertTrue(torch.equal(masked[:, -5:, 3:], original[:, -5:, 3:]))
        self.assertTrue(torch.equal(masked[:, -5:, :3], torch.zeros_like(masked[:, -5:, :3])))
        self.assertTrue(torch.equal(original, torch.from_numpy(inner.x_train[:3])))

    def test_auxiliary_updates_only_original_encoder_and_projection(self) -> None:
        inner = self.data.prepare("validation_1", "all", inner_stop=True)
        scratch = make_model("Fuzzy2", 11, None, False)
        state, info = pretrain_representation(inner.x_train[:16], 11)
        pretrained = make_model("Fuzzy2", 11, state, False)
        self.assertEqual(info["train_rows"], 16)
        self.assertEqual(info["epochs"], 20)
        self.assertTrue(np.isfinite(info["loss_trace"]).all())
        self.assertTrue(any(not torch.equal(scratch.encoder.state_dict()[key], value)
                            for key, value in state["encoder"].items()))
        self.assertTrue(torch.equal(scratch.experts.weight, pretrained.experts.weight))
        self.assertTrue(torch.equal(scratch.centers, pretrained.centers))
        self.assertTrue(torch.equal(scratch.scale_logits, pretrained.scale_logits))

    def test_freeze_changes_gradients_without_changing_fuzzy_topology(self) -> None:
        inner = self.data.prepare("validation_1", "all", inner_stop=True)
        state, _ = pretrain_representation(inner.x_train[:8], 11)
        model = make_model("Fuzzy2", 11, state, True)
        self.assertEqual(model.rule_count, 2)
        self.assertTrue(all(not p.requires_grad for p in model.encoder.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.projection.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.experts.parameters()))
        self.assertEqual(tuple(model(torch.from_numpy(inner.x_train[:8])).shape), (8,))


if __name__ == "__main__":
    unittest.main()
