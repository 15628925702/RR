"""Formal balanced-pilot/design/update RR-GID toy implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .policies import frank_wolfe, round_cost_share
from .synthetic_oracle import feature_map, sample_conditional, sample_full


@dataclass(frozen=True)
class Theta:
    lower: np.ndarray
    upper: np.ndarray

    def project(self, beta: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(beta), self.lower, self.upper)


def balanced_counts(budget: int, probabilities: np.ndarray, costs: np.ndarray) -> np.ndarray:
    return round_cost_share(probabilities, budget, costs).counts


def ht_moment(observations: list[tuple[tuple[int, ...], np.ndarray]], supports: tuple[tuple[int, ...], ...], coverage: np.ndarray, feature_fn) -> np.ndarray:
    out = np.zeros(len(supports))
    for a, support in enumerate(supports):
        if coverage[a] <= 0:
            raise ValueError("HT requires positive feature coverage")
        vals = [feature_fn(y, panel)[a] for panel, y in observations if set(support).issubset(panel)]
        out[a] = np.mean(vals) / coverage[a] if vals else 0.0
    return out


def psd_project(matrix: np.ndarray, cutoff: float = 1e-10) -> np.ndarray:
    matrix = (np.asarray(matrix) + np.asarray(matrix).T) / 2
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.maximum(values, cutoff)) @ vectors.T


def cross_completion_information(mixture, beta: np.ndarray, panel: tuple[int, ...], tilted: np.ndarray, scale: np.ndarray, n_completion: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    phi = feature_map(tilted, scale)
    mu = phi.mean(0)
    batch1, batch2 = [], []
    for row in tilted:
        batch1.append(feature_map(sample_conditional(mixture, row[list(panel)], panel, n_completion, int(rng.integers(2**31 - 1))), scale).mean(0) - mu)
        batch2.append(feature_map(sample_conditional(mixture, row[list(panel)], panel, n_completion, int(rng.integers(2**31 - 1))), scale).mean(0) - mu)
    a, b = np.asarray(batch1), np.asarray(batch2)
    ac, bc = a - a.mean(0), b - b.mean(0)
    return psd_project((ac.T @ bc + bc.T @ ac) / (2 * (len(a) - 1)))


def fisher_update(beta: np.ndarray, information: np.ndarray, estimating_equation: np.ndarray, theta: Theta) -> np.ndarray:
    return theta.project(np.asarray(beta) + np.linalg.pinv(psd_project(information)) @ np.asarray(estimating_equation))

