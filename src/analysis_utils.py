"""Shared integrity checks and paired-loss summaries for saved experiments.

This module is deliberately separate from frozen training implementations.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact_path(root: Path, relative: str) -> Path:
    """Read Windows-recorded relative paths on any platform, within root."""
    base = Path(root).resolve()
    path = (base / relative.replace("\\", "/")).resolve()
    if not path.is_relative_to(base):
        raise ValueError(f"Artifact path escapes its root: {relative}")
    return path


def verify_hashes(root: Path, expected: Mapping[str, str]) -> int:
    for relative, digest in expected.items():
        path = artifact_path(root, relative)
        if sha256_file(path) != digest:
            raise ValueError(f"Artifact hash mismatch: {path}")
    return len(expected)


def circular_block_interval(
    chunks: Sequence[np.ndarray],
    block_length: int,
    *,
    resamples: int = 5000,
    seed: int = 20261105,
) -> tuple[float, float, float]:
    """Resample paired loss differences within folds, weighting by date count.

    Each array is one chronological fold. Shared-date seed losses must already
    be averaged; neither folds nor model seeds are additional market dates.
    """
    if block_length < 1 or resamples < 1 or not chunks:
        raise ValueError("Positive block length/resamples and nonempty folds required")
    folds = [np.asarray(chunk, dtype=float) for chunk in chunks]
    if any(chunk.ndim != 1 or not len(chunk) or not np.isfinite(chunk).all() for chunk in folds):
        raise ValueError("Each fold must be a nonempty finite vector")
    rng = np.random.default_rng(seed)
    total = np.zeros(resamples)
    for chunk in folds:
        n = len(chunk)
        starts = rng.integers(0, n, size=(resamples, int(np.ceil(n / block_length))))
        ids = (starts[:, :, None] + np.arange(block_length)) % n
        total += chunk[ids.reshape(resamples, -1)[:, :n]].sum(1)
    lower, upper = np.quantile(total / sum(map(len, folds)), [0.025, 0.975])
    return float(np.concatenate(folds).mean()), float(lower), float(upper)
