"""Discriminative Score OED baseline interface."""

from __future__ import annotations

import numpy as np


def masked_input(x: np.ndarray, panel: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x)
    mask = np.zeros(x.shape[-1], dtype=float)
    mask[list(panel)] = 1.0
    masked = x * mask
    return np.concatenate([masked, np.broadcast_to(mask, x.shape[:-1] + mask.shape)], axis=-1), mask


class LinearScoreNetwork:
    """Small deterministic CPU baseline implementing masked score prediction."""

    def __init__(self, input_dim: int, output_dim: int):
        self.weights = np.zeros((input_dim, output_dim))

    def fit(self, x: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> None:
        gram = x.T @ x + ridge * np.eye(x.shape[1])
        self.weights = np.linalg.solve(gram, x.T @ y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x) @ self.weights


def score_information(model: LinearScoreNetwork, validation: np.ndarray, panels: tuple[tuple[int, ...], ...], weights: np.ndarray) -> np.ndarray:
    infos = []
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    for panel in panels:
        inputs, _ = masked_input(validation, panel)
        scores = model.predict(inputs)
        mean = weights @ scores
        centered = scores - mean
        infos.append((centered * weights[:, None]).T @ centered)
    return np.asarray(infos)

