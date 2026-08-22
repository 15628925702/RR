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


def sample_conditional_batch(
    mixture: FrozenMixture, x_s: np.ndarray, panel: tuple[int, ...], n: int, seed: int | None = None
) -> np.ndarray:
    """Independent exact GMM conditional completions for a batch of panels.

    The returned array has shape ``(n_rows, n, dimension)``.  This is a
    vectorized implementation of :func:`sample_conditional`, used only to make
    the formal-budget S1 estimator computationally feasible.
    """
    panel = tuple(panel)
    observed = np.atleast_2d(np.asarray(x_s, dtype=float))
    if observed.shape[1] != len(panel):
        raise ValueError("observed batch has incompatible panel width")
    rng = np.random.default_rng(seed)
    rows = observed.shape[0]
    complement = tuple(i for i in range(mixture.dimension) if i not in panel)
    z_s = inverse_warp(observed, mixture.alpha)
    log_probs = np.empty((rows, len(mixture.weights)))
    parameters = []
    for k, (weight, mean, cov) in enumerate(zip(mixture.weights, mixture.means, mixture.covariances)):
        sigma_ss = cov[np.ix_(panel, panel)]
        inv_ss = np.linalg.inv(sigma_ss)
        delta = z_s - mean[list(panel)]
        sign, logdet = np.linalg.slogdet(sigma_ss)
        if sign <= 0:
            raise ValueError("mixture covariance is not positive definite")
        log_probs[:, k] = np.log(weight) - 0.5 * (
            len(panel) * np.log(2 * np.pi) + logdet + np.einsum("ni,ij,nj->n", delta, inv_ss, delta)
        )
        gain = cov[np.ix_(complement, panel)] @ inv_ss
        cond_cov = cov[np.ix_(complement, complement)] - gain @ cov[np.ix_(panel, complement)]
        parameters.append((mean[list(complement)], mean[list(panel)], gain, cond_cov))
    probs = np.exp(log_probs - log_probs.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    uniforms = rng.random((rows, n))
    components = (uniforms[:, :, None] > np.cumsum(probs, axis=1)[:, None, :]).sum(axis=2)
    z = np.empty((rows, n, mixture.dimension))
    z[:, :, list(panel)] = z_s[:, None, :]
    for k, (mean_c, mean_s, gain, cond_cov) in enumerate(parameters):
        row_idx, sample_idx = np.nonzero(components == k)
        if row_idx.size:
            conditional_mean = mean_c + (z_s[row_idx] - mean_s) @ gain.T
            draws = rng.multivariate_normal(np.zeros(len(complement)), cond_cov, size=row_idx.size) + conditional_mean
            for j, coordinate in enumerate(complement):
                z[row_idx, sample_idx, coordinate] = draws[:, j]
    return warp(z, mixture.alpha)


def tilted_conditional_sample(
    mixture: FrozenMixture,
    beta: np.ndarray,
    x_s: np.ndarray,
    panel: tuple[int, ...],
    n: int,
    seed: int | None = None,
    scale: np.ndarray | None = None,
    feature_fn=None,
) -> np.ndarray:
    """Exact rejection samples from the relative tilt conditional law.

    Since every feature coordinate is bounded by one, ``sum(abs(beta))`` is a
    valid log-tilt envelope. Rejection from the exact GMM conditional therefore
    has no self-normalized importance bias.
    """
    rng = np.random.default_rng(seed)
    fn = feature_map if feature_fn is None else feature_fn
    beta = np.asarray(beta, dtype=float)
    panel_set = set(panel)
    fixed = np.zeros(beta.shape[0], dtype=bool)
    fixed[:6] = [i in panel_set for i in range(6)]
    fixed[6:] = [i in panel_set and i + 6 in panel_set for i in range(6)]
    observed_full = np.zeros(mixture.dimension)
    observed_full[list(panel)] = np.asarray(x_s)
    observed_features = fn(observed_full[None, :])[0]
    envelope = float(np.dot(beta[fixed], observed_features[fixed]) + np.sum(np.abs(beta[~fixed])))
    accepted = []
    while len(accepted) < n:
        proposal = sample_conditional(mixture, x_s, panel, max(32, 2 * (n - len(accepted))), int(rng.integers(2**31 - 1)))
        logits = fn(proposal) @ beta
        keep = rng.random(len(proposal)) < np.exp(logits - envelope)
        accepted.extend(proposal[keep])
    return np.asarray(accepted[:n])


def tilted_conditional_batch(
    mixture: FrozenMixture,
    beta: np.ndarray,
    x_s: np.ndarray,
    panel: tuple[int, ...],
    n: int,
    seed: int | None = None,
    scale: np.ndarray | None = None,
    feature_fn=None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fn = feature_map if feature_fn is None else feature_fn
    rows = np.atleast_2d(np.asarray(x_s))
    out = np.empty((len(rows), n, mixture.dimension))
    panel_set = set(panel)
    fixed = np.zeros(beta.shape[0], dtype=bool)
    fixed[:6] = [i in panel_set for i in range(6)]
    fixed[6:] = [i in panel_set and i + 6 in panel_set for i in range(6)]
    full = np.zeros((len(rows), mixture.dimension)); full[:, list(panel)] = rows
    fixed_features = fn(full)[:, fixed] @ beta[fixed]
    envelope = fixed_features + np.sum(np.abs(beta[~fixed]))
    filled = np.zeros(len(rows), dtype=int)
    while np.any(filled < n):
        need = np.maximum(n - filled, 0)
        proposal_n = int(max(64, min(2048, 4 * int(need.max()))))
        observed_rep = np.repeat(rows, proposal_n, axis=0)
        proposals = sample_conditional_batch(mixture, observed_rep, panel, 1, int(rng.integers(2**31 - 1)))[:, 0]
        feats = fn(proposals)
        logits = feats @ beta
        row_ids = np.repeat(np.arange(len(rows)), proposal_n)
        keep = rng.random(len(proposals)) < np.exp(logits - envelope[row_ids])
        for row_id in np.flatnonzero(need):
            selected = proposals[(row_ids == row_id) & keep]
            take = min(len(selected), n - filled[row_id])
            if take:
                out[row_id, filled[row_id]:filled[row_id] + take] = selected[:take]
                filled[row_id] += take
    return out


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


# ---------------------------------------------------------------------------
# P7 S2 reuse: bounded unary/pairwise feature dictionary.
#
# PDF 7.3 freezes the single-task feature map ``phi`` above but requires that a
# generator trained once can be *reused* across campaigns that redraw their own
# 12-feature map from a pre-defined bounded dictionary, redraw ``beta*``, and
# redraw a subset of candidate pair panels.  ``feature_fn_from_dictionary``
# materializes any such draw as an opaque ``x -> R^r`` callable so the whole
# downstream tilt / information machinery can accept ``feature_fn`` uniformly.
# ---------------------------------------------------------------------------

UNARY_FEATURE = "unary"
PAIR_FEATURE = "pair"


def make_feature_dictionary(dimension: int = 16) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Bounded unary/pairwise dictionary (PDF 7.3) over all coordinate pairs."""
    dim = int(dimension)
    unary = [(UNARY_FEATURE, (j,)) for j in range(dim)]
    pairs = [(PAIR_FEATURE, (i, j)) for i in range(dim) for j in range(i + 1, dim)]
    return tuple(unary + pairs)


def sample_feature_draw(dictionary, seed: int, r: int = 12):
    """Draw ``r`` features without replacement from the dictionary."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dictionary), size=r, replace=False)
    return [dictionary[i] for i in idx]


def feature_fn_from_dictionary(features, scale: np.ndarray | None = None):
    """Build an ``x -> R^r`` callable for a list of (kind, coords) features."""
    scale = np.ones(16) if scale is None else np.asarray(scale, dtype=float)

    def fn(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x_tilde = x / scale
        cols = []
        for kind, coords in features:
            if kind == UNARY_FEATURE:
                cols.append(np.tanh(x_tilde[..., coords[0]]))
            else:  # PAIR_FEATURE
                i, j = coords
                cols.append(np.tanh(x_tilde[..., i] * x_tilde[..., j]))
        return np.stack(cols, axis=-1)

    return fn


def reference_scale(mixture: FrozenMixture, n: int = 6000, seed: int = 2026) -> np.ndarray:
    """Estimate the frozen per-coordinate Q0 reference scale from an independent pool."""
    if n <= 1:
        raise ValueError("reference scale pool size must exceed one")
    samples = sample_full(mixture, n, seed=seed)
    scale = samples.std(axis=0, ddof=1)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("reference scale must be finite and positive")
    return scale


def log_partition(beta: np.ndarray, reference_samples: np.ndarray, scale: np.ndarray | None = None,
                  feature_fn=None) -> float:
    from numpy import logaddexp
    beta = np.asarray(beta)
    fn = feature_map if feature_fn is None else feature_fn
    values = fn(reference_samples) @ beta
    return float(logaddexp.reduce(values) - np.log(values.size))


def tilted_moments(beta: np.ndarray, reference_samples: np.ndarray, scale: np.ndarray | None = None,
                   feature_fn=None) -> tuple[np.ndarray, np.ndarray]:
    fn = feature_map if feature_fn is None else feature_fn
    features = fn(reference_samples)
    logits = features @ np.asarray(beta)
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    mean = weights @ features
    centered = features - mean
    fisher = (centered * weights[:, None]).T @ centered
    return mean, fisher


def tilted_sample_from_reference(beta: np.ndarray, reference_samples: np.ndarray, n: int, seed: int = 0,
                                 scale: np.ndarray | None = None, feature_fn=None) -> np.ndarray:
    """Sample Q_beta by exact importance resampling from an independent Q0 pool."""
    rng = np.random.default_rng(seed)
    fn = feature_map if feature_fn is None else feature_fn
    features = fn(reference_samples)
    logits = features @ np.asarray(beta)
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    return np.asarray(reference_samples)[rng.choice(len(reference_samples), size=n, replace=True, p=weights)]


def tilted_full_sample(mixture: FrozenMixture, beta: np.ndarray, n: int, seed: int = 0,
                       scale: np.ndarray | None = None, feature_fn=None) -> np.ndarray:
    """Exact accept-reject samples from Q_beta relative to Q0."""
    rng = np.random.default_rng(seed)
    fn = feature_map if feature_fn is None else feature_fn
    envelope = float(np.sum(np.abs(np.asarray(beta))))
    accepted = []
    while len(accepted) < n:
        proposal = sample_full(mixture, max(256, min(4096, 4 * (n - len(accepted)))), int(rng.integers(2**31 - 1)))
        logits = fn(proposal) @ beta
        keep = rng.random(len(proposal)) < np.exp(logits - envelope)
        accepted.extend(proposal[keep])
    return np.asarray(accepted[:n])


def beta_direction_and_scale(reference_samples: np.ndarray, seed: int = 2026, target_ess_fraction: float = 0.5,
                             scale: np.ndarray | None = None, feature_fn=None) -> np.ndarray:
    """Freeze a nonzero direction and choose its magnitude by ESS/N bisection."""
    rng = np.random.default_rng(seed)
    fn = feature_map if feature_fn is None else feature_fn
    phi = fn(reference_samples)
    r = phi.shape[-1]
    direction = rng.normal(size=r)
    direction /= np.linalg.norm(direction)
    target = target_ess_fraction * len(phi)
    lo, hi = 0.0, 8.0
    for _ in range(60):
        mag = (lo + hi) / 2
        logits = phi @ (mag * direction)
        weights = np.exp(logits - logits.max())
        ess = weights.sum() ** 2 / np.sum(weights ** 2)
        if ess > target:
            lo = mag
        else:
            hi = mag
    return ((lo + hi) / 2) * direction


def full_target_kl(beta_true: np.ndarray, beta_est: np.ndarray, reference_samples: np.ndarray,
                   scale: np.ndarray | None = None, feature_fn=None) -> float:
    mean_true, _ = tilted_moments(beta_true, reference_samples, scale, feature_fn=feature_fn)
    return float((np.asarray(beta_true) - np.asarray(beta_est)) @ mean_true
                 - log_partition(beta_true, reference_samples, scale, feature_fn=feature_fn)
                 + log_partition(beta_est, reference_samples, scale, feature_fn=feature_fn))
