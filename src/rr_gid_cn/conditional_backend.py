"""Cached Q0 conditional bases for bulk scoring and information.

High-precision / rejection samplers stay in ``synthetic_oracle`` for
calibration. Bulk experiments reweight a frozen Q0 feature basis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .rrgid import psd_project
from .synthetic_oracle import (
    _conditional_parameters,
    _nested_qmc_normals,
    _record_workload,
    conditional_component_posterior_batch,
    feature_map,
    inverse_warp,
    log_partition,
    tilted_conditional_mean_qmc,
    warp,
)

FEATURE_DIM = 12
_CACHE_COUNTERS = {
    "feature_nodes_built": 0,
    "feature_nodes_reweighted": 0,
    "component_nodes": 0,
    "cache_hit": 0,
    "cache_miss": 0,
    "bytes_cached": 0,
}


def reset_cache_counters() -> None:
    for key in _CACHE_COUNTERS:
        _CACHE_COUNTERS[key] = 0


def cache_counters() -> dict[str, int]:
    return {key: int(value) for key, value in _CACHE_COUNTERS.items()}


def _bump(**values: int) -> None:
    payload = {key: int(value) for key, value in values.items()}
    for key, value in payload.items():
        if key in _CACHE_COUNTERS:
            _CACHE_COUNTERS[key] += value
    _record_workload(**payload)


def _stable_logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    shift = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(shift, axis=axis) + np.log(np.exp(values - shift).sum(axis=axis))


def _hot_dtype(name: str) -> np.dtype:
    if name == "float32":
        return np.float32
    if name == "float64":
        return np.float64
    raise ValueError(f"unsupported hot dtype: {name}")


@dataclass
class ReferenceMomentCache:
    phi_reference: np.ndarray
    phi_reference_eval: np.ndarray | None = None
    A_beta_true: float | None = None
    mu_beta_true: np.ndarray | None = None
    F_beta_true: np.ndarray | None = None

    @classmethod
    def build(
        cls,
        reference: np.ndarray,
        scale: np.ndarray,
        *,
        beta_true: np.ndarray | None = None,
        feature_fn=None,
        dtype: str = "float64",
    ) -> "ReferenceMomentCache":
        fn = feature_fn or (lambda x: feature_map(x, scale))
        phi = np.asarray(fn(np.asarray(reference)), dtype=np.float64)
        cache = cls(phi_reference=phi, phi_reference_eval=phi)
        if beta_true is not None:
            cache.mu_beta_true, cache.F_beta_true = cache.moments(np.asarray(beta_true, dtype=float))
            logits = phi @ np.asarray(beta_true, dtype=float)
            cache.A_beta_true = float(_stable_logsumexp(logits, axis=0) - np.log(len(logits)))
        _bump(bytes_cached=int(phi.nbytes), cache_miss=1)
        return cache

    def moments(self, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        beta = np.asarray(beta, dtype=float)
        logits = self.phi_reference @ beta
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        mean = weights @ self.phi_reference
        centered = self.phi_reference - mean
        fisher = (centered * weights[:, None]).T @ centered
        return mean, fisher

    def log_partition(self, beta: np.ndarray) -> float:
        logits = self.phi_reference @ np.asarray(beta, dtype=float)
        return float(_stable_logsumexp(logits, axis=0) - np.log(len(logits)))

    def kl(self, beta_true: np.ndarray, beta_hat: np.ndarray) -> float:
        if self.A_beta_true is None or self.mu_beta_true is None:
            mu_true, _ = self.moments(beta_true)
            a_true = self.log_partition(beta_true)
        else:
            mu_true = self.mu_beta_true
            a_true = self.A_beta_true
        # KL(Q_{β*} || Q_{β̂}) = A(β̂) - A(β*) - μ* · (β̂ - β*)
        return float(
            self.log_partition(beta_hat) - a_true
            - mu_true @ (np.asarray(beta_hat) - np.asarray(beta_true))
        )


@dataclass
class GroupedPanelData:
    panel_ids: tuple[tuple[int, ...], ...]
    offsets: np.ndarray
    observed_values: np.ndarray
    counts: np.ndarray

    @classmethod
    def from_observations(
        cls,
        observations: list[tuple[tuple[int, ...], np.ndarray]],
        *,
        width: int | None = None,
    ) -> "GroupedPanelData":
        grouped: dict[tuple[int, ...], list[np.ndarray]] = {}
        for panel, row in observations:
            grouped.setdefault(tuple(panel), []).append(np.asarray(row, dtype=float))
        panel_ids = tuple(grouped)
        blocks = [np.atleast_2d(np.asarray(grouped[panel], dtype=float)) for panel in panel_ids]
        if not blocks:
            return cls((), np.zeros(1, dtype=int), np.zeros((0, int(width or 0))), np.zeros(0, dtype=int))
        counts = np.asarray([len(block) for block in blocks], dtype=int)
        offsets = np.concatenate(([0], np.cumsum(counts)))
        return cls(panel_ids, offsets, np.concatenate(blocks, axis=0), counts)

    def rows(self, panel: tuple[int, ...]) -> np.ndarray:
        index = self.panel_ids.index(tuple(panel))
        return self.observed_values[self.offsets[index] : self.offsets[index + 1]]

    def as_groups(self) -> dict[tuple[int, ...], np.ndarray]:
        return {panel: self.rows(panel) for panel in self.panel_ids}


@dataclass
class ConditionalFeatureBasis:
    panel: tuple[int, ...]
    observed_rows: np.ndarray
    component_log_posterior: np.ndarray
    phi_nodes: np.ndarray
    qmc_order: int
    seed: int
    dtype: str
    n_nodes: int = 0

    def __post_init__(self) -> None:
        self.n_nodes = int(self.phi_nodes.shape[2])

    def conditional_mean(self, beta: np.ndarray) -> np.ndarray:
        mean, _cov = self._reduce(beta, want_cov=False)
        return mean

    def conditional_mean_and_cov(self, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._reduce(beta, want_cov=True)

    def _reduce(self, beta: np.ndarray, *, want_cov: bool) -> tuple[np.ndarray, np.ndarray | None]:
        beta = np.asarray(beta, dtype=np.float64)
        phi = self.phi_nodes.astype(np.float64, copy=False)
        n_rows, n_comp, n_nodes, rank = phi.shape
        logits = np.einsum("kcnr,r->kcn", phi, beta)
        logw = logits + self.component_log_posterior[:, :, None]
        logw = logw.reshape(n_rows, n_comp * n_nodes)
        shift = logw.max(axis=1, keepdims=True)
        weights = np.exp(logw - shift)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-300)
        flat = phi.reshape(n_rows, n_comp * n_nodes, rank)
        mean = np.einsum("kn,knr->kr", weights, flat)
        _bump(feature_nodes_reweighted=n_rows * n_comp * n_nodes)
        if not want_cov:
            return mean.astype(np.float64, copy=False), None
        centered = flat - mean[:, None, :]
        cov = np.einsum("kn,knr,kns->krs", weights, centered, centered)
        return mean.astype(np.float64, copy=False), cov.astype(np.float64, copy=False)


def build_conditional_feature_basis(
    mixture,
    rows: np.ndarray,
    panel: tuple[int, ...],
    *,
    order: int,
    seed: int,
    scale: np.ndarray,
    feature_fn=None,
    dtype: str = "float32",
    feature_coords: int = FEATURE_DIM,
) -> ConditionalFeatureBasis:
    """Build one Q0 conditional feature basis. Does not call Gaussian sampling later."""
    if order < 4 or order > 18:
        raise ValueError("QMC order must be in [4, 18]")
    panel = tuple(int(i) for i in panel)
    rows = np.atleast_2d(np.asarray(rows, dtype=np.float64))
    n_nodes = 1 << int(order)
    complement = tuple(i for i in range(mixture.dimension) if i not in panel)
    z_s = inverse_warp(rows, mixture.alpha)
    log_post = _component_log_posterior(mixture, rows, panel)
    posterior = np.exp(log_post - log_post.max(axis=1, keepdims=True))
    posterior /= posterior.sum(axis=1, keepdims=True)
    parameters = _conditional_parameters(mixture, panel)[1]
    normals = _nested_qmc_normals(n_nodes, len(complement), seed=int(seed))
    hot = _hot_dtype(dtype)
    n_comp = len(parameters)
    fn = feature_fn
    built = []
    for k, (mean_c, mean_s, _inv_ss, gain, _cond_cov, chol, _logdet) in enumerate(parameters):
        x = np.zeros((len(rows), n_nodes, mixture.dimension), dtype=np.float64)
        x[:, :, list(panel)] = warp(z_s[:, None, :], mixture.alpha)
        cond_mean = mean_c[None, :] + (z_s - mean_s) @ gain.T
        z_missing = cond_mean[:, None, :] + normals[None, :, :] @ chol.T
        x[:, :, list(complement)] = warp(z_missing, mixture.alpha)
        if fn is None:
            phi = feature_map(x.reshape(-1, mixture.dimension), scale).reshape(len(rows), n_nodes, -1)
        else:
            phi = np.asarray(fn(x.reshape(-1, mixture.dimension)), dtype=np.float64).reshape(len(rows), n_nodes, -1)
        built.append(phi.astype(hot, copy=False))
    phi_nodes = np.stack(built, axis=1)
    _bump(
        feature_nodes_built=int(phi_nodes.size // max(phi_nodes.shape[-1], 1)),
        component_nodes=int(n_comp * n_nodes),
        cache_miss=1,
        bytes_cached=int(phi_nodes.nbytes + log_post.nbytes),
    )
    return ConditionalFeatureBasis(
        panel=panel,
        observed_rows=rows.astype(np.float64, copy=False),
        component_log_posterior=np.log(np.maximum(posterior, 1e-300)),
        phi_nodes=phi_nodes,
        qmc_order=int(order),
        seed=int(seed),
        dtype=str(dtype),
    )


def evaluate_conditional_basis(beta: np.ndarray, basis: ConditionalFeatureBasis) -> np.ndarray:
    return basis.conditional_mean(beta)


def _component_log_posterior(mixture, rows: np.ndarray, panel: tuple[int, ...]) -> np.ndarray:
    posterior = conditional_component_posterior_batch(mixture, rows, panel)
    return np.log(np.maximum(posterior, 1e-300))


@dataclass
class PanelInformationBasis:
    outer_phi: np.ndarray
    outer_panel_values: dict[tuple[int, ...], np.ndarray]
    basis_a: dict[tuple[int, ...], ConditionalFeatureBasis]
    basis_b: dict[tuple[int, ...], ConditionalFeatureBasis]
    outer_rows: int
    qmc_order: int
    seed: int

    def information(self, beta: np.ndarray, panels: tuple[tuple[int, ...], ...] | None = None) -> dict[tuple[int, ...], np.ndarray]:
        beta = np.asarray(beta, dtype=np.float64)
        logits = self.outer_phi @ beta
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        mu = weights @ self.outer_phi
        active = panels if panels is not None else tuple(self.basis_a)
        out: dict[tuple[int, ...], np.ndarray] = {}
        for panel in active:
            a = self.basis_a[panel].conditional_mean(beta) - mu
            b = self.basis_b[panel].conditional_mean(beta) - mu
            a = a - a.mean(0)
            b = b - b.mean(0)
            info = (a.T @ b + b.T @ a) / max(2 * (len(a) - 1), 1)
            out[panel] = psd_project(info)
        return out


def build_panel_information_basis(
    mixture,
    reference: np.ndarray,
    panels: tuple[tuple[int, ...], ...],
    *,
    scale: np.ndarray,
    outer_rows: int,
    order: int,
    seed: int,
    feature_fn=None,
    dtype: str = "float32",
    scrambles: int = 2,
) -> PanelInformationBasis:
    if int(scrambles) < 2:
        raise ValueError("cross-completion requires at least two independent bases")
    fn = feature_fn or (lambda x: feature_map(x, scale))
    rng = np.random.default_rng(int(seed))
    index = rng.choice(len(reference), size=int(outer_rows), replace=False)
    outer = np.asarray(reference)[index]
    outer_phi = np.asarray(fn(outer), dtype=np.float64)
    outer_panel_values = {tuple(panel): outer[:, list(panel)] for panel in panels}
    basis_a: dict[tuple[int, ...], ConditionalFeatureBasis] = {}
    basis_b: dict[tuple[int, ...], ConditionalFeatureBasis] = {}
    for panel in panels:
        observed = outer_panel_values[tuple(panel)]
        basis_a[tuple(panel)] = build_conditional_feature_basis(
            mixture, observed, tuple(panel), order=order, seed=int(seed) + 1,
            scale=scale, feature_fn=feature_fn, dtype=dtype,
        )
        basis_b[tuple(panel)] = build_conditional_feature_basis(
            mixture, observed, tuple(panel), order=order, seed=int(seed) + 2,
            scale=scale, feature_fn=feature_fn, dtype=dtype,
        )
    return PanelInformationBasis(
        outer_phi=outer_phi,
        outer_panel_values=outer_panel_values,
        basis_a=basis_a,
        basis_b=basis_b,
        outer_rows=int(outer_rows),
        qmc_order=int(order),
        seed=int(seed),
    )


def cholesky_solve(matrix: np.ndarray, rhs: np.ndarray, *, ridge: float = 1e-8) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve H x = u with Cholesky. Auditable ridge if the minimum eigenvalue is too small."""
    matrix = 0.5 * (np.asarray(matrix, dtype=np.float64) + np.asarray(matrix, dtype=np.float64).T)
    rhs = np.asarray(rhs, dtype=np.float64)
    values = np.linalg.eigvalsh(matrix)
    lambda_min = float(values.min())
    used_ridge = 0.0
    work = matrix
    if lambda_min < float(ridge):
        used_ridge = float(ridge) - min(lambda_min, 0.0)
        work = matrix + used_ridge * np.eye(len(matrix))
    try:
        step = np.linalg.solve(work, rhs)
        factor = "cholesky_via_solve"
        np.linalg.cholesky(work)
        factor = "cholesky"
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError("H is not usable after auditable ridge") from exc
    diagnostics = {
        "lambda_min_H": lambda_min,
        "lambda_max_H": float(values.max()),
        "ridge": used_ridge,
        "linear_algebra": factor,
    }
    return step, diagnostics


def uncached_qmc_mean(mixture, beta, rows, panel, order, seed, scale, feature_fn=None) -> np.ndarray:
    """Reference path used by T1: existing QMC integrand, no cache."""
    return tilted_conditional_mean_qmc(
        mixture, beta, rows, panel, int(order), seed=int(seed), scale=scale, feature_fn=feature_fn,
    )
