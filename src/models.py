"""Protocol v1.2 forecasting heads; outputs are standardized next-day returns."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class LSTMEncoder(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        if input_size not in (3, 5, 6):
            raise ValueError("The frozen source groups have 3, 5, or 6 features")
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=32, num_layers=1, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[1:] != (20, self.input_size):
            raise ValueError(f"Expected (batch,20,{self.input_size})")
        _, (hidden, _) = self.lstm(x)
        return hidden[-1]


class LSTM32(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.encoder = LSTMEncoder(input_size)
        self.head = nn.Linear(32, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.encoder(x)).squeeze(-1)


class LSTM3(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.encoder = LSTMEncoder(input_size)
        self.projection = nn.Linear(32, 3)
        self.head = nn.Linear(3, 1)

    def forward(self, x: Tensor) -> Tensor:
        z = torch.tanh(self.projection(self.encoder(x)))
        return self.head(z).squeeze(-1)


class FuzzyK(nn.Module):
    """Three-dimensional Gaussian TSK gate with K affine local experts."""

    def __init__(self, input_size: int, rule_count: int = 2):
        super().__init__()
        if rule_count not in (1, 2, 3):
            raise ValueError("The protocol specifies K=1, 2, or 3")
        self.rule_count = rule_count
        self.encoder = LSTMEncoder(input_size)
        self.projection = nn.Linear(32, 3)
        initial = torch.zeros(rule_count, 3)
        if rule_count > 1:
            initial[:, 0] = torch.linspace(-0.25, 0.25, rule_count)
        self.centers = nn.Parameter(initial)
        scale_fraction = (1.0 - 0.1) / 1.9
        initial_rho = math.log(scale_fraction / (1.0 - scale_fraction))
        self.scale_logits = nn.Parameter(torch.full((rule_count, 3), initial_rho))
        self.experts = nn.Linear(3, rule_count)

    def forward(self, x: Tensor, *, return_diagnostics: bool = False):
        z = torch.tanh(self.projection(self.encoder(x)))
        scales = 0.1 + 1.9 * torch.sigmoid(self.scale_logits)
        distance2 = (((z[:, None, :] - self.centers[None, :, :]) / scales[None, :, :]) ** 2).sum(-1)
        log_membership = -0.5 * distance2
        alpha = torch.softmax(log_membership, dim=-1)
        expert_predictions = self.experts(z)
        prediction = (alpha * expert_predictions).sum(-1)
        if not return_diagnostics:
            return prediction
        return prediction, {
            "z": z,
            "alpha": alpha,
            "expert_predictions": expert_predictions,
            "log_membership": log_membership,
            "distance2": distance2,
        }


class Gate2(nn.Module):
    """Small non-Gaussian gate with the same latent space and two experts."""

    def __init__(self, input_size: int):
        super().__init__()
        self.encoder = LSTMEncoder(input_size)
        self.projection = nn.Linear(32, 3)
        self.gate = nn.Sequential(nn.Linear(3, 2), nn.Tanh(), nn.Linear(2, 2))
        self.experts = nn.Linear(3, 2)

    def forward(self, x: Tensor, *, return_diagnostics: bool = False):
        z = torch.tanh(self.projection(self.encoder(x)))
        alpha = torch.softmax(self.gate(z), dim=-1)
        expert_predictions = self.experts(z)
        prediction = (alpha * expert_predictions).sum(-1)
        if not return_diagnostics:
            return prediction
        return prediction, {"z": z, "alpha": alpha,
                            "expert_predictions": expert_predictions}


def build_model(model_id: str, input_size: int) -> nn.Module:
    if model_id == "LSTM32":
        return LSTM32(input_size)
    if model_id == "LSTM3":
        return LSTM3(input_size)
    if model_id.startswith("Fuzzy") and model_id[5:] in ("1", "2", "3"):
        return FuzzyK(input_size, int(model_id[5:]))
    if model_id == "Gate2":
        return Gate2(input_size)
    raise ValueError(f"Unknown model_id: {model_id}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
