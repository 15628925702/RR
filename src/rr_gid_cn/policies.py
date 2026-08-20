"""P2 allocation policies and cost-aware Frank-Wolfe solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Allocation:
    probabilities: np.ndarray
    counts: np.ndarray
    costs: np.ndarray
    budget: int

    @property
    def spent(self) -> int:
        return int(self.counts @ self.costs)


def uniform_probabilities(n_panels: int) -> np.ndarray:
    if n_panels <= 0:
        raise ValueError("number of panels must be positive")
    return np.full(n_panels, 1.0 / n_panels)


def round_cost_share(probabilities: np.ndarray, budget: int, costs: np.ndarray) -> Allocation:
    p = np.asarray(probabilities, dtype=float)
    c = np.asarray(costs, dtype=int)
    if p.ndim != 1 or c.shape != p.shape or budget < 0 or np.any(c <= 0):
        raise ValueError("invalid allocation inputs")
    if np.any(p < 0) or not np.isclose(p.sum(), 1.0, atol=1e-10):
        raise ValueError("probabilities must be nonnegative and sum to one")
    counts = np.floor(budget * p / c).astype(int)
    remaining = budget - int(counts @ c)
    remainders = budget * p / c - counts
    order = np.argsort(-remainders)
    while remaining >= int(c[order[0]]):
        added = False
        for idx in order:
            if remaining >= c[idx]:
                counts[idx] += 1
                remaining -= int(c[idx])
                added = True
                break
        if not added:
            break
    return Allocation(p, counts, c, budget)


def _safe_inverse(matrix: np.ndarray, jitter: float = 1e-10) -> np.ndarray:
    matrix = (matrix + matrix.T) / 2
    try:
        np.linalg.cholesky(matrix)
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(matrix)
        floor = max(jitter, -float(eigvals.min()) + jitter)
        return (eigvecs * (1.0 / np.maximum(eigvals, floor))) @ eigvecs.T


def objective(fisher: np.ndarray, informations: np.ndarray, probabilities: np.ndarray, costs: np.ndarray) -> float:
    M = np.tensordot(probabilities / costs, informations, axes=(0, 0))
    return float(np.trace(fisher @ _safe_inverse(M)))


def frank_wolfe(fisher: np.ndarray, informations: np.ndarray, costs: np.ndarray, safe_probabilities: np.ndarray, tolerance: float = 1e-8, max_iter: int = 500) -> tuple[np.ndarray, float, int]:
    F = (np.asarray(fisher) + np.asarray(fisher).T) / 2
    I = np.asarray(informations)
    c = np.asarray(costs, dtype=float)
    p = np.asarray(safe_probabilities, dtype=float).copy()
    if len(p) != len(I) or not np.isclose(p.sum(), 1.0, atol=1e-10):
        raise ValueError("safe allocation must be a probability vector")
    for iteration in range(max_iter):
        M = np.tensordot(p / c, I, axes=(0, 0))
        Minv = _safe_inverse(M)
        phi = float(np.trace(F @ Minv))
        sensitivities = np.asarray([np.trace(Minv @ F @ Minv @ info) / cost for info, cost in zip(I, c)])
        idx = int(np.argmax(sensitivities))
        gap = float(sensitivities[idx] - phi)
        if gap <= tolerance:
            return p, gap, iteration
        direction = np.zeros_like(p)
        direction[idx] = 1.0
        lo, hi = 0.0, 1.0
        golden = (np.sqrt(5.0) - 1.0) / 2.0
        x1, x2 = hi - golden * (hi - lo), lo + golden * (hi - lo)
        f1 = objective(F, I, (1 - x1) * p + x1 * direction, c)
        f2 = objective(F, I, (1 - x2) * p + x2 * direction, c)
        for _ in range(40):
            if f1 > f2:
                lo, x1, f1 = x1, x2, f2
                x2 = lo + golden * (hi - lo)
                f2 = objective(F, I, (1 - x2) * p + x2 * direction, c)
            else:
                hi, x2, f2 = x2, x1, f1
                x1 = hi - golden * (hi - lo)
                f1 = objective(F, I, (1 - x1) * p + x1 * direction, c)
        eta = float((lo + hi) / 2)
        p = (1 - eta) * p + eta * direction
    return p, gap, max_iter


def covariance_information(reference_samples: np.ndarray, panels: tuple[tuple[int, int], ...]) -> np.ndarray:
    x = np.asarray(reference_samples)
    return np.asarray([np.cov(x[:, list(panel)], rowvar=False) for panel in panels])


def conditional_information(mixture, beta: np.ndarray, panels, reference_samples: np.ndarray, scale: np.ndarray, n_tilted: int = 1000, n_conditional: int = 64, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Monte Carlo exact-oracle panel information using independent conditional completions."""
    from .synthetic_oracle import feature_map, sample_conditional, sample_full, tilted_moments
    rng = np.random.default_rng(seed)
    tilted = sample_full(mixture, n_tilted, seed=int(rng.integers(2**31 - 1)))
    phi = feature_map(tilted, scale)
    mu = phi.mean(0)
    F = np.cov(phi, rowvar=False)
    infos = []
    for panel in panels:
        projected = []
        for row in tilted:
            completions = sample_conditional(mixture, row[list(panel)], panel, n_conditional, seed=int(rng.integers(2**31 - 1)))
            projected.append(feature_map(completions, scale).mean(0) - mu)
        projected = np.asarray(projected)
        infos.append(np.cov(projected, rowvar=False))
    return F, np.asarray(infos)
