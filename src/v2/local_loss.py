"""The one predeclared local-expert loss for the V2-H3 development experiment."""

from __future__ import annotations

import torch
from torch import Tensor


def local_expert_loss(alpha: Tensor, experts: Tensor, target: Tensor,
                      *, epsilon: float = 1e-8) -> Tensor:
    if alpha.ndim != 2 or experts.shape != alpha.shape or target.shape != (len(alpha),):
        raise ValueError("Expected alpha/experts=(batch,rules), target=(batch,)")
    weights = alpha.detach()
    squared_error = (experts - target[:, None]).square()
    per_rule = (weights * squared_error).sum(dim=0) / (weights.sum(dim=0) + epsilon)
    return per_rule.mean()


def loss_components(prediction: Tensor, experts: Tensor, alpha: Tensor,
                    target: Tensor, *, local_weight: float) -> tuple[Tensor, Tensor, Tensor]:
    if local_weight not in (0.0, 0.1):
        raise ValueError("Only lambda=0 engineering control or frozen lambda=0.1 is allowed")
    mixed = torch.nn.functional.mse_loss(prediction, target)
    if local_weight == 0:
        local = mixed.new_zeros(())
    else:
        local = local_expert_loss(alpha, experts, target)
    return mixed + local_weight * local, mixed, local
