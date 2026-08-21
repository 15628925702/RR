"""Synthetic S1 oracle-gate evaluation with panel-specific exact conditional scores."""

from __future__ import annotations

import numpy as np

from .policies import frank_wolfe, uniform_probabilities
from .synthetic_oracle import beta_direction_and_scale, feature_map, full_target_kl, log_partition, sample_conditional, sample_conditional_batch, sample_full, tilted_conditional_sample, tilted_conditional_batch, tilted_full_sample, tilted_moments, tilted_sample_from_reference


def exact_panel_information(mixture, beta, panels, reference_pool, scale, n_tilted=256, n_conditional=64, seed=0):
    rng = np.random.default_rng(seed)
    tilted = tilted_full_sample(mixture, beta, n_tilted, int(rng.integers(2**31 - 1)), scale)
    phi = feature_map(tilted, scale)
    mu = phi.mean(0)
    fisher = np.cov(phi, rowvar=False)
    infos = []
    for panel in panels:
        observed = tilted[:, list(panel)]
        projected_a = feature_map(tilted_conditional_batch(mixture, beta, observed, panel, n_conditional, int(rng.integers(2**31 - 1)), scale), scale).mean(axis=1) - mu
        projected_b = feature_map(tilted_conditional_batch(mixture, beta, observed, panel, n_conditional, int(rng.integers(2**31 - 1)), scale), scale).mean(axis=1) - mu
        a = np.asarray(projected_a); b = np.asarray(projected_b)
        a = a - a.mean(0); b = b - b.mean(0)
        info = (a.T @ b + b.T @ a) / max(2 * (len(a) - 1), 1)
        vals, vecs = np.linalg.eigh((info + info.T) / 2)
        infos.append((vecs * np.maximum(vals, 1e-10)) @ vecs.T)
    return fisher, np.asarray(infos)


def a_optimal_information(reference: np.ndarray, panels: tuple[tuple[int, int], ...], dimension: int = 16) -> np.ndarray:
    """Panel Fisher information used by the A-optimal split-questionnaire baseline.

    For a multivariate-normal mean parameter, observing ``X_S`` carries the
    ``(S, S)`` block of the full precision matrix (complete-case information).
    Each panel contributes a ``dimension x dimension`` matrix that is zero outside
    its ``(S, S)`` block; the A-optimal solver then minimizes ``tr(M_A^-1)`` over
    the shared cost-aware ``M_A(p) = sum_S p_S/c(S) I_S``.  This is the PDF's
    "complete reference covariance" formulation, with a ridge on the block
    precision for finite-sample stability.
    """
    x = np.asarray(reference, dtype=float)
    full_cov = np.cov(x, rowvar=False)
    inv_full = np.linalg.inv(full_cov + 1e-6 * np.eye(dimension))
    infos = []
    for panel in panels:
        idx = list(panel)
        info = np.zeros((dimension, dimension))
        info[np.ix_(idx, idx)] = inv_full[np.ix_(idx, idx)]
        infos.append(info)
    return np.asarray(infos)


def policy_designs(reference, panels, fisher, oracle_information):
    costs = np.ones(len(panels))
    uniform = uniform_probabilities(len(panels))
    a_info = a_optimal_information(reference, panels)
    a_p, _, _ = frank_wolfe(np.eye(16), a_info, costs, uniform, tolerance=1e-4, max_iter=500)
    rr_p, _, _ = frank_wolfe(fisher, oracle_information, costs, uniform, tolerance=0.2, max_iter=300)
    return {"Uniform SQD": uniform, "A-OSQD": a_p, "oracle RR-GID": rr_p}


def balanced_pilot_counts(panels: tuple[tuple[int, int], ...], budget: int) -> np.ndarray:
    """PDF balanced pilot: equal mass on the six direct feature-support pairs."""
    counts = np.zeros(len(panels), dtype=int)
    supports = [(i, i + 6) for i in range(6)]
    indices = [panels.index(s) for s in supports if s in panels]
    if not indices:
        return np.floor(budget * uniform_probabilities(len(panels))).astype(int)
    base, rem = divmod(budget, len(indices))
    counts[indices] = base
    for idx in indices[:rem]:
        counts[idx] += 1
    return counts


def pilot_ht_moment(observations, pilot_counts, panels, scale, reference):
    n0 = int(pilot_counts.sum())
    if n0 <= 0:
        return np.zeros(12), np.zeros(12)
    supports = [(i,) for i in range(6)] + [(i, i + 6) for i in range(6)]
    values = np.zeros(12)
    for a, support in enumerate(supports):
        rho = sum(c for c, panel in zip(pilot_counts, panels) if set(support).issubset(panel)) / n0
        if rho <= 0:
            continue
        vals = []
        for panel, observed in observations:
            if set(support).issubset(panel):
                full = np.zeros(16); full[list(panel)] = observed
                vals.append(feature_map(full[None, :], scale)[0, a])
        values[a] = np.sum(vals) / (n0 * rho) if vals else 0.0
    return values, np.asarray([sum(c for c, panel in zip(pilot_counts, panels) if set(s).issubset(panel)) / n0 for s in supports])


def solve_pilot_beta(mu_pil, reference, scale, theta_bound=4.0, steps=20, norm_cap=None):
    beta = np.zeros(12)
    features = feature_map(reference, scale)
    target = np.asarray(mu_pil)
    def objective(x):
        z = features @ x
        return float(np.logaddexp.reduce(z) - np.log(len(z)) - x @ target)
    for _ in range(max(steps, 100)):
        mu, fisher = tilted_moments(beta, reference, scale)
        direction = np.linalg.solve(fisher + 1e-3 * np.eye(12), target - mu)
        current = objective(beta)
        step = 1.0
        while step > 1e-5:
            candidate = np.clip(beta + step * direction, -theta_bound, theta_bound)
            if norm_cap is not None:
                nrm = np.linalg.norm(candidate)
                if nrm > norm_cap:
                    candidate = candidate * (norm_cap / nrm)
            if objective(candidate) <= current + 1e-9:
                break
            step *= 0.5
        if step <= 1e-5 or np.linalg.norm(candidate - beta) < 1e-7:
            break
        beta = candidate
    return beta


def prepare_s1_oracle(mixture, scale, panels, seed=2026, reference_size=50000, information_samples=256, conditional_samples=32, large_reference_size=200000):
    reference = sample_full(mixture, reference_size, seed)
    reference_large = sample_full(mixture, large_reference_size, seed + 12345)
    beta_true = beta_direction_and_scale(reference, 2026, 0.5, scale)
    fisher, oracle_information = exact_panel_information(mixture, beta_true, panels, reference, scale, information_samples, conditional_samples, seed + 1)
    designs = policy_designs(reference, panels, fisher, oracle_information)
    return {"reference": reference, "reference_large": reference_large, "beta_true": beta_true, "fisher": fisher, "information": oracle_information, "designs": designs}


def imp_conditional_mean(mixture, beta, batch, panel, n, seed, scale):
    """E_{Q_beta}[phi(X)|X_S] by importance weighting on exact Q0 conditional samples.

    Vectorized: returns (n_rows, r).  This replaces accept-reject tilting of the
    conditional law for the S1 exact-conditional-oracle gate (PDF Sec. 4.1 allows
    importance proposals from Q0; cross-completion removes the self-normalized bias).
    """
    rng = np.random.default_rng(seed)
    completions = sample_conditional_batch(mixture, batch, panel, n, int(rng.integers(2**31 - 1)))
    phi = feature_map(completions, scale)
    w = np.exp(phi @ np.asarray(beta))
    return np.einsum("onr,on->or", phi, w) / w.sum(axis=1, keepdims=True)


def panel_information_cross(mixture, beta, panels, reference, scale, n_tilted, n_cond, seed):
    """Cross-completion panel information estimator (PDF Eq. 9) with PSD projection.

    Uses importance-weighted conditional completions (not accept-reject), so it is
    cheap enough to re-estimate ``I_S(beta^(j))`` at every scoring step as PDF
    Algorithm 2 step 6 requires.  ``reference`` must be a large Q0 pool so the
    tilted ``mu`` estimate has negligible variance.
    """
    rng = np.random.default_rng(seed)
    features = feature_map(reference, scale)
    logits = features @ np.asarray(beta)
    w = np.exp(logits - logits.max())
    w /= w.sum()
    idx = rng.choice(len(reference), size=n_tilted, p=w)
    tilted = reference[idx]
    mu = features[idx].mean(0)
    infos = []
    for panel in panels:
        observed = tilted[:, list(panel)]
        a = imp_conditional_mean(mixture, beta, observed, panel, n_cond, seed + 1, scale) - mu
        b = imp_conditional_mean(mixture, beta, observed, panel, n_cond, seed + 2, scale) - mu
        a = a - a.mean(0)
        b = b - b.mean(0)
        info_hat = (a.T @ b + b.T @ a) / max(2 * (len(a) - 1), 1)
        vals, vecs = np.linalg.eigh((info_hat + info_hat.T) / 2)
        infos.append((vecs * np.maximum(vals, 1e-10)) @ vecs.T)
    return np.asarray(infos)


def final_rr_estimator(mixture, beta_start, observations, panels, reference, scale, lu, seed,
                       theta_bound=4.0, h_tilted=128, h_cond=32, step_size=1.0, norm_cap=None):
    """One Fisher-scoring update following PDF Algorithm 2 step 6.

    Re-estimates ``I_S(beta^(j))`` by cross-completion at the current beta, builds
    ``U`` from exact-conditional-oracle observed scores (importance-weighted), and
    takes the projected full step ``beta + H^{-1} U``, projected back onto Theta
    (the PDF leaves Theta's numeric boundary open; the norm cap keeps tilt overlap
    bounded for extreme baseline allocations).
    """
    rng = np.random.default_rng(seed)
    grouped = {}
    for panel, obs in observations:
        grouped.setdefault(panel, []).append(obs)
    infos = panel_information_cross(mixture, beta_start, panels, reference, scale, h_tilted, h_cond, seed + 7)
    H = np.zeros((12, 12))
    projected = []
    for panel, rows in grouped.items():
        H += len(rows) * infos[panels.index(panel)]
        projected.append(imp_conditional_mean(mixture, beta_start, np.asarray(rows), panel, lu, int(rng.integers(2**31 - 1)), scale))
    projected = np.concatenate(projected, axis=0)
    mu_beta, _ = tilted_moments(beta_start, reference, scale)
    U = np.sum(projected - mu_beta, axis=0)
    H = (H + H.T) / 2
    step = np.linalg.solve(H + 1e-2 * np.eye(12), U)
    updated = np.asarray(beta_start) + step_size * step
    updated = np.clip(updated, -theta_bound, theta_bound)
    if norm_cap is not None:
        nrm = np.linalg.norm(updated)
        if nrm > norm_cap:
            updated = updated * (norm_cap / nrm)
    return updated


def run_replication(mixture, scale, panels, budget, seed, prepared=None,
                    lu=128, h_tilted=128, h_cond=32, pilot_norm_cap=2.0, kl_samples=20000,
                    scoring_steps=2):
    prepared = prepared or prepare_s1_oracle(mixture, scale, panels, seed)
    reference = prepared["reference"]
    ref_large = prepared["reference_large"]
    beta_true = prepared["beta_true"]
    fisher = prepared["fisher"]
    oracle_information = prepared["information"]
    designs = prepared["designs"]
    target_reference = sample_full(mixture, max(4000, budget * 2), seed + 2)
    target_full = tilted_full_sample(mixture, beta_true, budget, seed + 3, scale)
    rows = []
    pilot_budget = int(np.ceil(10.0 * budget ** (1.0 / 3.0)))
    pilot_budget = min(pilot_budget, budget)
    remaining_budget = budget - pilot_budget
    rr_phi = float(np.trace(fisher @ np.linalg.pinv(np.tensordot(designs["oracle RR-GID"], oracle_information, axes=(0, 0)))))
    mu_bt = feature_map(tilted_full_sample(mixture, beta_true, kl_samples, seed + 90, scale), scale).mean(0)
    for name, probabilities in designs.items():
        expected = remaining_budget * probabilities
        counts = np.floor(expected).astype(int)
        # Largest-remainder apportionment exactly honors the frozen acquisition
        # budget without changing the continuous policy design.
        remainder = remaining_budget - int(counts.sum())
        if remainder:
            counts[np.argsort(expected - counts)[-remainder:]] += 1
        if int(counts.sum()) != remaining_budget:
            raise RuntimeError("main panel allocation does not exhaust the remaining acquisition budget")
        pilot_counts = balanced_pilot_counts(panels, pilot_budget)
        pilot_observations = []
        pilot_cursor = 0
        for panel, count in zip(panels, pilot_counts):
            for row in target_full[pilot_cursor : pilot_cursor + count]:
                pilot_observations.append((panel, row[list(panel)]))
            pilot_cursor += count
        main_observations = []
        main_cursor = pilot_budget
        for panel, count in zip(panels, counts):
            for row in target_full[main_cursor : main_cursor + count]:
                main_observations.append((panel, row[list(panel)]))
            main_cursor += count
        pilot_mu, pilot_rho = pilot_ht_moment(pilot_observations, pilot_counts, panels, scale, reference)
        beta_hat = solve_pilot_beta(pilot_mu, reference, scale, norm_cap=pilot_norm_cap)
        observations = pilot_observations + main_observations
        update_diagnostics = [{"step": "pilot", "pilot_budget": int(pilot_counts.sum()), "beta_norm": float(np.linalg.norm(beta_hat)), "rho_min": float(np.min(pilot_rho[pilot_rho > 0])) if np.any(pilot_rho > 0) else 0.0}]
        for update in range(scoring_steps):
            beta_next = final_rr_estimator(mixture, beta_hat, observations, panels, ref_large, scale, lu, seed + 4 + update, h_tilted=h_tilted, h_cond=h_cond, step_size=1.0, norm_cap=pilot_norm_cap * 1.25)
            update_diagnostics.append({"step": update, "step_norm": float(np.linalg.norm(beta_next - beta_hat)), "projected": bool(np.any(np.abs(beta_next) >= 4.0)), "pilot_budget": int(pilot_counts.sum())})
            beta_hat = beta_next
        kl = max(0.0, float((beta_true - beta_hat) @ mu_bt - log_partition(beta_true, target_reference, scale) + log_partition(beta_hat, target_reference, scale)))
        rows.append({"policy": name, "budget": budget, "allocated_observations": int(counts.sum() + pilot_counts.sum()), "pilot_budget": int(pilot_counts.sum()), "seed": seed, "beta_true_norm": float(np.linalg.norm(beta_true)), "beta_hat_norm": float(np.linalg.norm(beta_hat)), "kl": kl, "B_kl": budget * kl, "design_ratio": float(kl / max(rr_phi / (2 * budget), 1e-12)), "target_draw_seed": seed + 3, "update_diagnostics": update_diagnostics})
    return rows
