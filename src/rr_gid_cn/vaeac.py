"""Lightweight arbitrary-conditioning generator interface for P6 smoke."""

from __future__ import annotations

import numpy as np


class VAEACGenerator:
    """Empirical VAEAC-compatible interface; checkpoint-free smoke backbone."""

    def __init__(self, reference: np.ndarray):
        self.reference = np.asarray(reference, dtype=float)
        if self.reference.ndim != 2:
            raise ValueError("reference must be a 2D array")

    @property
    def dimension(self) -> int:
        return self.reference.shape[1]

    def sample_full(self, n: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return self.reference[rng.integers(0, len(self.reference), size=n)]

    def sample_conditional(self, observed: np.ndarray, panel: tuple[int, ...], n: int, seed: int = 0) -> np.ndarray:
        observed = np.asarray(observed)
        panel = tuple(panel)
        if observed.shape[-1] != len(panel):
            raise ValueError("observed values and panel size disagree")
        rng = np.random.default_rng(seed)
        distances = np.sum((self.reference[:, list(panel)] - observed) ** 2, axis=1)
        weights = np.exp(-distances / max(np.median(distances), 1e-8))
        weights /= weights.sum()
        draws = self.reference[rng.choice(len(self.reference), size=n, p=weights)].copy()
        draws[:, list(panel)] = observed
        return draws

    def tilted_sample(self, beta: np.ndarray, feature_fn, n: int, seed: int = 0) -> tuple[np.ndarray, float, float]:
        rng = np.random.default_rng(seed)
        pool = self.sample_full(max(10 * n, 1000), int(rng.integers(2**31 - 1)))
        logits = feature_fn(pool) @ np.asarray(beta)
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        indices = rng.choice(len(pool), size=n, replace=True, p=weights)
        ess = float(1.0 / np.sum(weights**2) / len(weights))
        return pool[indices], 1.0, ess

