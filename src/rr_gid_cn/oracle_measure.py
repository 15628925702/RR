"""Single-true-Q0 numerical measure used by the P4 Phase-2 gold gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

import numpy as np
from scipy.special import ndtri
from scipy.optimize import minimize
from scipy.stats import qmc

from .synthetic_oracle import (
    FrozenMixture,
    conditional_component_posterior_batch,
    feature_map,
    inverse_warp,
    tilted_conditional_mean_exact,
    tilted_conditional_sample,
    tilted_full_sample,
    warp,
    _conditional_parameters,
)


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FullLawQMC:
    order: int = 14
    scrambles: int = 8
    seed: int = 26082026

    def validate(self) -> None:
        if self.order < 6 or self.order > 20 or self.scrambles < 2:
            raise ValueError("invalid full-law QMC configuration")


@dataclass(frozen=True)
class ConditionalQMC:
    start_order: int = 7
    max_order: int = 11
    scrambles: int = 4
    atol: float = 2e-4
    rtol: float = 2e-4
    scramble_se_atol: float = 3e-4
    scramble_se_rtol: float = 3e-4
    seed: int = 26083026

    def validate(self) -> None:
        if self.start_order < 4 or self.max_order < self.start_order:
            raise ValueError("invalid conditional QMC orders")
        if self.scrambles < 2:
            raise ValueError("conditional QMC requires independent scrambles")


@dataclass(frozen=True)
class InformationQMC:
    outer_order: int = 8
    outer_scrambles: int = 4
    seed: int = 26084026
    psd_floor: float = 0.0

    def validate(self) -> None:
        if self.outer_order < 5 or self.outer_order > 16 or self.outer_scrambles < 2:
            raise ValueError("invalid information QMC configuration")
        if self.psd_floor < 0:
            raise ValueError("PSD floor must be nonnegative")


class OracleMeasure:
    """All P4 oracle quantities evaluated against one analytic GMM/warp Q0."""

    schema_version = "p4-oracle-measure-v1"

    def __init__(
        self,
        mixture: FrozenMixture,
        scale: np.ndarray,
        full_config: FullLawQMC,
        conditional_config: ConditionalQMC,
        information_config: InformationQMC,
        feature_fn=None,
    ):
        full_config.validate()
        conditional_config.validate()
        information_config.validate()
        self.mixture = mixture
        self.scale = np.asarray(scale, dtype=float)
        if self.scale.shape != (mixture.dimension,) or np.any(self.scale <= 0):
            raise ValueError("scale must be a positive vector matching mixture dimension")
        self.full_config = full_config
        self.conditional_config = conditional_config
        self.information_config = information_config
        self.feature_fn = feature_fn
        self._moment_cache: dict[bytes, dict[str, Any]] = {}

    def _features(self, x: np.ndarray) -> np.ndarray:
        return feature_map(x, self.scale) if self.feature_fn is None else self.feature_fn(x)

    def _component_qmc(self, order: int, seed: int) -> np.ndarray:
        n = 1 << int(order)
        chunks = []
        for component, (mean, covariance) in enumerate(
            zip(self.mixture.means, self.mixture.covariances)
        ):
            engine = qmc.Sobol(
                d=self.mixture.dimension,
                scramble=True,
                seed=int(seed) + 104729 * component,
            )
            normal = ndtri(np.clip(engine.random_base2(int(order)), 1e-12, 1.0 - 1e-12))
            latent = normal @ np.linalg.cholesky(covariance).T + mean
            chunks.append(warp(latent, self.mixture.alpha))
        return np.stack(chunks, axis=0).reshape(len(chunks) * n, self.mixture.dimension)

    def _integral_replicates(self, beta: np.ndarray) -> list[dict[str, Any]]:
        beta = np.asarray(beta, dtype=float)
        values = []
        for scramble in range(self.full_config.scrambles):
            seed = self.full_config.seed + 1_000_003 * scramble
            x = self._component_qmc(self.full_config.order, seed)
            phi = self._features(x)
            logits = phi @ beta
            shift = float(np.max(logits))
            unnormalized = np.exp(logits - shift)
            z = float(np.mean(unnormalized))
            weights = unnormalized / np.sum(unnormalized)
            mu = weights @ phi
            centered = phi - mu
            fisher = (centered * weights[:, None]).T @ centered
            values.append(
                {
                    "A": float(shift + np.log(z)),
                    "mu": mu,
                    "F": fisher,
                    "x": x,
                    "phi": phi,
                    "weights": weights,
                    "seed": int(seed),
                }
            )
        return values

    def moments(self, beta: np.ndarray) -> dict[str, Any]:
        beta = np.asarray(beta, dtype=float)
        key = np.ascontiguousarray(beta).tobytes()
        if key not in self._moment_cache:
            replicates = self._integral_replicates(beta)
            A_values = np.asarray([item["A"] for item in replicates])
            mu_values = np.stack([item["mu"] for item in replicates])
            F_values = np.stack([item["F"] for item in replicates])
            raw_F = F_values.mean(axis=0)
            sym_F = (raw_F + raw_F.T) / 2
            eigvals, eigvecs = np.linalg.eigh(sym_F)
            projected_F = (eigvecs * np.maximum(eigvals, 0.0)) @ eigvecs.T
            self._moment_cache[key] = {
                "A": float(A_values.mean()),
                "mu": mu_values.mean(axis=0),
                "F_raw": raw_F,
                "F_sym": sym_F,
                "F_projected": projected_F,
                "F_psd_correction_norm": float(np.linalg.norm(projected_F - sym_F, ord="fro")),
                "A_scramble_se": float(A_values.std(ddof=1) / np.sqrt(len(A_values))),
                "mu_scramble_se_max": float(
                    np.max(mu_values.std(axis=0, ddof=1) / np.sqrt(len(mu_values)))
                ),
                "F_scramble_se_max": float(
                    np.max(F_values.std(axis=0, ddof=1) / np.sqrt(len(F_values)))
                ),
                "replicate_A": A_values.tolist(),
                "sample_size_per_component": 1 << self.full_config.order,
                "total_sample_size_per_scramble": len(self.mixture.weights)
                * (1 << self.full_config.order),
                "scramble_seeds": [item["seed"] for item in replicates],
            }
        return self._moment_cache[key]

    def A(self, beta: np.ndarray) -> float:
        return float(self.moments(beta)["A"])

    def mu(self, beta: np.ndarray) -> np.ndarray:
        return np.asarray(self.moments(beta)["mu"])

    def F(self, beta: np.ndarray) -> np.ndarray:
        return np.asarray(self.moments(beta)["F_projected"])

    def tilt_full(self, beta: np.ndarray, n: int, seed: int) -> np.ndarray:
        return tilted_full_sample(
            self.mixture, beta, n, seed=seed, scale=self.scale, feature_fn=self.feature_fn
        )

    def tilt_cond(
        self, beta: np.ndarray, panel: tuple[int, ...], x_s: np.ndarray, n: int, seed: int
    ) -> np.ndarray:
        return tilted_conditional_sample(
            self.mixture, beta, x_s, panel, n, seed=seed,
            scale=self.scale, feature_fn=self.feature_fn,
        )

    def conditional_mean(
        self,
        beta: np.ndarray,
        panel: tuple[int, ...],
        x_s: np.ndarray,
        *,
        seed_offset: int = 0,
        return_diagnostics: bool = False,
    ):
        cfg = self.conditional_config
        return tilted_conditional_mean_exact(
            self.mixture,
            beta,
            x_s,
            panel,
            seed=cfg.seed + int(seed_offset),
            scale=self.scale,
            feature_fn=self.feature_fn,
            start_order=cfg.start_order,
            max_order=cfg.max_order,
            atol=cfg.atol,
            rtol=cfg.rtol,
            scrambles=cfg.scrambles,
            scramble_se_atol=cfg.scramble_se_atol,
            scramble_se_rtol=cfg.scramble_se_rtol,
            return_diagnostics=return_diagnostics,
        )

    def _outer_qmc(self, order: int, seed: int, beta: np.ndarray):
        x = self._component_qmc(order, seed)
        phi = self._features(x)
        logits = phi @ np.asarray(beta)
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        return x, phi, weights

    def panel_information(
        self, beta: np.ndarray, panel: tuple[int, ...]
    ) -> dict[str, Any]:
        cfg = self.information_config
        mu = self.mu(beta)
        raw_replicates = []
        score_mean_replicates = []
        conditional_diagnostics = []
        for scramble in range(cfg.outer_scrambles):
            seed = cfg.seed + 1_000_003 * scramble
            x, _phi, weights = self._outer_qmc(cfg.outer_order, seed, beta)
            conditional, diag = self.conditional_mean(
                beta,
                panel,
                x[:, list(panel)],
                seed_offset=10_000_019 * scramble + 8191 * sum(panel),
                return_diagnostics=True,
            )
            score = conditional - mu
            raw_replicates.append((score * weights[:, None]).T @ score)
            score_mean_replicates.append(weights @ score)
            conditional_diagnostics.append(diag)
        raw_values = np.stack(raw_replicates)
        score_mean_values = np.stack(score_mean_replicates)
        raw = raw_values.mean(axis=0)
        sym = (raw + raw.T) / 2
        eigvals, eigvecs = np.linalg.eigh(sym)
        projected = (eigvecs * np.maximum(eigvals, cfg.psd_floor)) @ eigvecs.T
        return {
            "raw": raw,
            "sym": sym,
            "projected": projected,
            "psd_correction_norm": float(np.linalg.norm(projected - sym, ord="fro")),
            "lambda_min_raw_sym": float(eigvals.min()),
            "outer_scramble_se_max": float(
                np.max(raw_values.std(axis=0, ddof=1) / np.sqrt(len(raw_values)))
            ),
            "score_mean": score_mean_values.mean(axis=0),
            "score_mean_scramble_se": score_mean_values.std(axis=0, ddof=1)
            / np.sqrt(len(score_mean_values)),
            "outer_order": int(cfg.outer_order),
            "outer_sample_size_per_scramble": int(
                len(self.mixture.weights) * (1 << cfg.outer_order)
            ),
            "outer_scramble_seeds": [
                int(cfg.seed + 1_000_003 * scramble)
                for scramble in range(cfg.outer_scrambles)
            ],
            "conditional_diagnostics": conditional_diagnostics,
        }

    def kl(self, beta_true: np.ndarray, beta_hat: np.ndarray) -> dict[str, float]:
        raw = float(
            (np.asarray(beta_true) - np.asarray(beta_hat)) @ self.mu(beta_true)
            - self.A(beta_true)
            + self.A(beta_hat)
        )
        tolerance = float(
            4.0
            * (
                self.moments(beta_true)["A_scramble_se"]
                + self.moments(beta_hat)["A_scramble_se"]
            )
        )
        return {"raw": raw, "numerical_tolerance": tolerance}

    def calibrate_beta(self, seed: int = 2026, target_ess_fraction: float = 0.5) -> np.ndarray:
        rng = np.random.default_rng(seed)
        dimension = self._features(np.zeros((1, self.mixture.dimension))).shape[-1]
        direction = rng.normal(size=dimension)
        direction /= np.linalg.norm(direction)
        x = self._component_qmc(self.full_config.order, self.full_config.seed + 97)
        phi = self._features(x)
        target = float(target_ess_fraction) * len(phi)
        lo, hi = 0.0, 8.0
        for _ in range(60):
            magnitude = (lo + hi) / 2
            logits = phi @ (magnitude * direction)
            weights = np.exp(logits - logits.max())
            ess = weights.sum() ** 2 / np.square(weights).sum()
            if ess > target:
                lo = magnitude
            else:
                hi = magnitude
        return ((lo + hi) / 2) * direction

    def definition(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "integration_method": "stratified_multi_scramble_sobol_true_q0",
            "full": asdict(self.full_config),
            "conditional": asdict(self.conditional_config),
            "information": asdict(self.information_config),
            "scale_sha256": array_sha256(self.scale),
            "mixture_parameter_sha256": mixture_sha256(self.mixture),
        }


def mixture_sha256(mixture: FrozenMixture) -> str:
    digest = hashlib.sha256()
    for value in (mixture.weights, mixture.means, mixture.covariances):
        digest.update(np.ascontiguousarray(value).tobytes())
    digest.update(np.float64(mixture.alpha).tobytes())
    return digest.hexdigest()


def cholesky_objective(
    fisher: np.ndarray, information: np.ndarray, probabilities: np.ndarray
) -> tuple[float, dict[str, float]]:
    M = np.tensordot(np.asarray(probabilities), np.asarray(information), axes=(0, 0))
    M = (M + M.T) / 2
    eigvals = np.linalg.eigvalsh(M)
    if eigvals.min() <= 0:
        raise np.linalg.LinAlgError("gold M(p) is not positive definite")
    chol = np.linalg.cholesky(M)
    solved = np.linalg.solve(chol.T, np.linalg.solve(chol, np.asarray(fisher)))
    return float(np.trace(solved)), {
        "lambda_min": float(eigvals.min()),
        "lambda_max": float(eigvals.max()),
        "condition_number": float(eigvals.max() / eigvals.min()),
    }


def frank_wolfe_gold(
    fisher: np.ndarray,
    information: np.ndarray,
    safe_probabilities: np.ndarray,
    tolerance: float,
    max_iter: int = 500,
) -> tuple[np.ndarray, float, int, float, dict[str, float]]:
    F = (np.asarray(fisher) + np.asarray(fisher).T) / 2
    I = np.asarray(information)
    p = np.asarray(safe_probabilities, dtype=float).copy()

    def value_and_gradient(probabilities):
        M = np.tensordot(probabilities, I, axes=(0, 0))
        chol = np.linalg.cholesky((M + M.T) / 2)
        Minv = np.linalg.solve(chol.T, np.linalg.solve(chol, np.eye(M.shape[0])))
        value = float(np.trace(F @ Minv))
        sensitivity = np.asarray(
            [np.trace(Minv @ F @ Minv @ item) for item in I]
        )
        return value, -sensitivity

    # Fully-corrective optimization is used to reach a sharp certificate on
    # the 120-panel simplex.  Acceptance still uses the original
    # Frank-Wolfe sensitivity gap, not the optimizer's success flag.
    result = minimize(
        lambda probabilities: value_and_gradient(probabilities),
        p,
        method="SLSQP",
        jac=True,
        bounds=[(0.0, 1.0)] * len(p),
        constraints={
            "type": "eq",
            "fun": lambda probabilities: float(np.sum(probabilities) - 1.0),
            "jac": lambda probabilities: np.ones_like(probabilities),
        },
        options={
            "ftol": min(float(tolerance) * 1e-4, 1e-12),
            "maxiter": int(max_iter),
            "disp": False,
        },
    )
    p = np.maximum(np.asarray(result.x, dtype=float), 0.0)
    p /= p.sum()
    for correction in range(50):
        M = np.tensordot(p, I, axes=(0, 0))
        chol = np.linalg.cholesky((M + M.T) / 2)
        Minv = np.linalg.solve(chol.T, np.linalg.solve(chol, np.eye(M.shape[0])))
        phi = float(np.trace(F @ Minv))
        sensitivity = np.asarray([np.trace(Minv @ F @ Minv @ item) for item in I])
        target = int(np.argmax(sensitivity))
        gap = float(sensitivity[target] - phi)
        if gap <= tolerance:
            objective, linear = cholesky_objective(F, I, p)
            return p, gap, int(result.nit) + correction, objective, linear
        vertex = np.zeros_like(p)
        vertex[target] = 1.0
        lo, hi = 0.0, 1.0
        for _ in range(40):
            left = lo + (hi - lo) / 3
            right = hi - (hi - lo) / 3
            f_left = cholesky_objective(F, I, (1 - left) * p + left * vertex)[0]
            f_right = cholesky_objective(F, I, (1 - right) * p + right * vertex)[0]
            if f_left <= f_right:
                hi = right
            else:
                lo = left
        eta = (lo + hi) / 2
        p = (1 - eta) * p + eta * vertex
        # Re-optimize all active weights after adding the FW vertex.
        result = minimize(
            lambda probabilities: value_and_gradient(probabilities),
            p,
            method="SLSQP",
            jac=True,
            bounds=[(0.0, 1.0)] * len(p),
            constraints={
                "type": "eq",
                "fun": lambda probabilities: float(np.sum(probabilities) - 1.0),
                "jac": lambda probabilities: np.ones_like(probabilities),
            },
            options={"ftol": min(float(tolerance) * 1e-4, 1e-12), "maxiter": 100},
        )
        p = np.maximum(np.asarray(result.x, dtype=float), 0.0)
        p /= p.sum()
    raise RuntimeError(
        "gold fully-corrective design did not meet the frozen FW certificate; "
        f"final gap={gap}, optimizer_message={result.message}"
    )
