"""Exact synthetic reference oracle from the frozen RR-GID_CN specification."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FrozenMixture:
    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    alpha: float

    @property
    def dimension(self) -> int:
        return int(self.means.shape[1])


def make_frozen_mixture(seed: int = 2026, dimension: int = 16, components: int = 4, alpha: float = 1.0) -> FrozenMixture:
    rng = np.random.default_rng(seed)
    means = rng.normal(0.0, 1.5, size=(components, dimension))
    covariances = np.empty((components, dimension, dimension))
    for k in range(components):
        L = rng.normal(0.0, 0.25, size=(dimension, 4))
        D = np.diag(rng.uniform(0.4, 0.8, size=dimension))
        covariances[k] = D + L @ L.T
    return FrozenMixture(np.full(components, 1.0 / components), means, covariances, float(alpha))


def warp(z: np.ndarray, alpha: float) -> np.ndarray:
    z = np.asarray(z)
    return np.sinh(alpha * z) / alpha if alpha > 0 else z.copy()


def inverse_warp(x: np.ndarray, alpha: float) -> np.ndarray:
    x = np.asarray(x)
    return np.arcsinh(alpha * x) / alpha if alpha > 0 else x.copy()


def all_pairs(dimension: int = 16) -> tuple[tuple[int, int], ...]:
    return tuple(combinations(range(dimension), 2))


def sample_full(mixture: FrozenMixture, n: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    components = rng.choice(len(mixture.weights), size=n, p=mixture.weights)
    z = np.empty((n, mixture.dimension))
    for k in range(len(mixture.weights)):
        idx = np.flatnonzero(components == k)
        if idx.size:
            z[idx] = rng.multivariate_normal(mixture.means[k], mixture.covariances[k], size=idx.size)
    return warp(z, mixture.alpha)


def conditional_component_posterior(mixture: FrozenMixture, x_s: np.ndarray, panel: tuple[int, ...]) -> np.ndarray:
    """Posterior component probabilities given warped observations on ``panel``."""
    panel = tuple(panel)
    z_s = inverse_warp(np.asarray(x_s), mixture.alpha)
    logs = []
    for weight, mean, cov in zip(mixture.weights, mixture.means, mixture.covariances):
        mu = mean[list(panel)]
        sigma = cov[np.ix_(panel, panel)]
        sign, logdet = np.linalg.slogdet(sigma)
        if sign <= 0:
            raise ValueError("mixture covariance is not positive definite")
        delta = z_s - mu
        logs.append(np.log(weight) - 0.5 * (len(panel) * np.log(2 * np.pi) + logdet + delta @ np.linalg.solve(sigma, delta)))
    logs = np.asarray(logs)
    probs = np.exp(logs - logs.max())
    return probs / probs.sum()


def sample_conditional(mixture: FrozenMixture, x_s: np.ndarray, panel: tuple[int, ...], n: int, seed: int | None = None) -> np.ndarray:
    panel = tuple(panel)
    if len(panel) == mixture.dimension:
        return np.repeat(np.asarray(x_s)[None, :], n, axis=0)
    rng = np.random.default_rng(seed)
    complement = tuple(i for i in range(mixture.dimension) if i not in panel)
    z_s = inverse_warp(np.asarray(x_s), mixture.alpha)
    posterior = conditional_component_posterior(mixture, x_s, panel)
    components = rng.choice(len(posterior), size=n, p=posterior)
    z = np.empty((n, mixture.dimension))
    z[:, list(panel)] = z_s
    for k in range(len(posterior)):
        idx = np.flatnonzero(components == k)
        if not idx.size:
            continue
        cov = mixture.covariances[k]
        ss = np.ix_(panel, panel)
        cc = np.ix_(complement, complement)
        cs = np.ix_(complement, panel)
        cond_mean = mixture.means[k][list(complement)] + cov[cs] @ np.linalg.solve(cov[ss], z_s - mixture.means[k][list(panel)])
        cond_cov = cov[cc] - cov[cs] @ np.linalg.solve(cov[ss], cov[np.ix_(panel, complement)])
        z[np.ix_(idx, complement)] = rng.multivariate_normal(cond_mean, cond_cov, size=idx.size)
    return warp(z, mixture.alpha)


def conditional_moments(mixture: FrozenMixture, x_s: np.ndarray, panel: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Analytic conditional mean/covariance in warped coordinates' latent space.

    The exact Gaussian-mixture conditional is used as the sampling oracle check;
    moments are returned for the latent coordinates because the nonlinear warp
    has no closed-form Gaussian moments.
    """
    panel = tuple(panel)
    complement = tuple(i for i in range(mixture.dimension) if i not in panel)
    z_s = inverse_warp(np.asarray(x_s), mixture.alpha)
    posterior = conditional_component_posterior(mixture, x_s, panel)
    means = []
    covs = []
    for k in range(len(posterior)):
        cov = mixture.covariances[k]
        ss = np.ix_(panel, panel)
        cs = np.ix_(complement, panel)
        cond_mean = mixture.means[k][list(complement)] + cov[cs] @ np.linalg.solve(cov[ss], z_s - mixture.means[k][list(panel)])
        cond_cov = cov[np.ix_(complement, complement)] - cov[cs] @ np.linalg.solve(cov[ss], cov[np.ix_(panel, complement)])
        means.append(cond_mean)
        covs.append(cond_cov)
    means = np.asarray(means)
    mean = posterior @ means
    covariance = sum(p * (cov + np.outer(mu - mean, mu - mean)) for p, mu, cov in zip(posterior, means, covs))
    return mean, covariance


def feature_map(x: np.ndarray, scale: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(x)
    if x.shape[-1] != 16:
        raise ValueError("synthetic feature map expects 16 coordinates")
    scale = np.ones(16) if scale is None else np.asarray(scale)
    if scale.shape != (16,) or np.any(scale <= 0):
        raise ValueError("scale must be a positive length-16 vector")
    x_tilde = x / scale
    unary = np.tanh(x_tilde[..., :6])
    pair = np.tanh(x_tilde[..., :6] * x_tilde[..., 6:12])
    return np.concatenate([unary, pair], axis=-1)


def reference_scale(mixture: FrozenMixture, n: int = 6000, seed: int = 2026) -> np.ndarray:
    """Estimate the frozen per-coordinate Q0 reference scale from an independent pool."""
    if n <= 1:
        raise ValueError("reference scale pool size must exceed one")
    samples = sample_full(mixture, n, seed=seed)
    scale = samples.std(axis=0, ddof=1)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("reference scale must be finite and positive")
    return scale


def log_partition(beta: np.ndarray, reference_samples: np.ndarray, scale: np.ndarray | None = None) -> float:
    from numpy import logaddexp
    beta = np.asarray(beta)
    values = feature_map(reference_samples, scale) @ beta
    return float(logaddexp.reduce(values) - np.log(values.size))


def tilted_moments(beta: np.ndarray, reference_samples: np.ndarray, scale: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    features = feature_map(reference_samples, scale)
    logits = features @ np.asarray(beta)
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    mean = weights @ features
    centered = features - mean
    fisher = (centered * weights[:, None]).T @ centered
    return mean, fisher


def full_target_kl(beta_true: np.ndarray, beta_est: np.ndarray, reference_samples: np.ndarray, scale: np.ndarray | None = None) -> float:
    mean_true, _ = tilted_moments(beta_true, reference_samples, scale)
    return float((np.asarray(beta_true) - np.asarray(beta_est)) @ mean_true - log_partition(beta_true, reference_samples, scale) + log_partition(beta_est, reference_samples, scale))
