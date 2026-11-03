"""Create the three version-2 CSV tables from one fuzzy model batch."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.project_eval import FUZZY_CASE_COLUMNS, FUZZY_RULE_COLUMNS, PREDICTION_COLUMNS, Standardizer


def fuzzy_batch_tables(
    *, model_id: str, source_group: str, fold: str, seed: int,
    window_ids: np.ndarray, prediction: torch.Tensor, diagnostics: dict,
    scaler: Standardizer,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Inverse-transform experts individually so their raw weighted sum matches y_pred."""
    if not model_id.startswith("Fuzzy") or model_id[5:] not in ("1", "2", "3"):
        raise ValueError("Expected Fuzzy1, Fuzzy2, or Fuzzy3")
    k = int(model_id[5:])
    ids = np.asarray(window_ids, dtype=np.int64)
    values = prediction.detach().cpu().numpy().reshape(-1)
    arrays = {name: tensor.detach().cpu().numpy() for name, tensor in diagnostics.items()}
    if len(ids) != len(values) or arrays["z"].shape != (len(ids), 3):
        raise ValueError("Batch dimensions do not align")
    for name in ("alpha", "expert_predictions", "log_membership", "distance2"):
        if arrays[name].shape != (len(ids), k):
            raise ValueError(f"Wrong {name} shape")
    raw_prediction = scaler.y_inverse(values)
    raw_experts = scaler.y_inverse(arrays["expert_predictions"])
    prediction_rows, case_rows, rule_rows = [], [], []
    for i, window_id in enumerate(ids):
        key = (model_id, source_group, k, fold, int(seed), int(window_id))
        prediction_rows.append((*key, float(raw_prediction[i])))
        case_rows.append((*key, *map(float, arrays["z"][i]), float(raw_prediction[i])))
        for j in range(k):
            rule_rows.append((*key, j + 1, float(arrays["alpha"][i, j]),
                              float(raw_experts[i, j]), float(arrays["log_membership"][i, j]),
                              float(arrays["distance2"][i, j])))
    return (
        pd.DataFrame(prediction_rows, columns=PREDICTION_COLUMNS),
        pd.DataFrame(case_rows, columns=FUZZY_CASE_COLUMNS),
        pd.DataFrame(rule_rows, columns=FUZZY_RULE_COLUMNS),
    )
