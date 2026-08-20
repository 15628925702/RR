"""Gas Sensor preprocessing schema from the frozen specification."""

from __future__ import annotations

from itertools import combinations

import numpy as np


def panel_library(n_sensors: int = 16):
    return tuple(combinations(range(n_sensors), 2))


def fit_scaler_pca(reference_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(reference_train, dtype=float)
    if x.ndim != 2 or x.shape[1] != 128:
        raise ValueError("Gas input must have shape (n, 128)")
    mean, std = x.mean(0), x.std(0, ddof=1)
    z = (x - mean) / np.maximum(std, 1e-12)
    pcs = []
    for sensor in range(16):
        block = z[:, sensor * 8 : (sensor + 1) * 8]
        _, _, vh = np.linalg.svd(block, full_matrices=False)
        vector = vh[0]
        if vector[np.argmax(np.abs(vector))] < 0:
            vector = -vector
        pcs.append(vector)
    return mean, np.asarray(pcs)


def transform_features(x: np.ndarray, mean: np.ndarray, std: np.ndarray, pcs: np.ndarray) -> np.ndarray:
    z = (np.asarray(x) - mean) / np.maximum(std, 1e-12)
    scores = np.column_stack([z[:, i * 8 : (i + 1) * 8] @ pcs[i] for i in range(16)])
    return np.concatenate([np.tanh(scores[:, :8]), np.tanh(scores[:, :8] * scores[:, 8:])], axis=1)

