"""Synthetic S1 oracle-gate evaluation with panel-specific exact conditional scores."""

from __future__ import annotations

import numpy as np

from .policies import covariance_information, frank_wolfe, uniform_probabilities
from .synthetic_oracle import beta_direction_and_scale, feature_map, full_target_kl, sample_conditional, sample_conditional_batch, sample_full, tilted_conditional_sample, tilted_conditional_batch, tilted_full_sample, tilted_moments, tilted_sample_from_reference


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


def policy_designs(reference, panels, fisher, oracle_information):
    costs = np.ones(len(panels))
    uniform = uniform_probabilities(len(panels))
    a_info = covariance_information(reference, panels)
    a_p, _, _ = frank_wolfe(np.eye(2), a_info, costs, uniform, tolerance=0.2, max_iter=300)
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


def solve_pilot_beta(mu_pil, reference, scale, theta_bound=4.0, steps=20):
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
            if objective(candidate) <= current + 1e-9:
                break
            step *= 0.5
        if step <= 1e-5 or np.linalg.norm(candidate - beta) < 1e-7:
            break
        beta = candidate
    return beta


def prepare_s1_oracle(mixture, scale, panels, seed=2026, reference_size=50000, information_samples=256, conditional_samples=32):
    reference = sample_full(mixture, reference_size, seed)
    beta_true = beta_direction_and_scale(reference, 2026, 0.5, scale)
    fisher, oracle_information = exact_panel_information(mixture, beta_true, panels, reference, scale, information_samples, conditional_samples, seed + 1)
    designs = policy_designs(reference, panels, fisher, oracle_information)
    return {"reference": reference, "beta_true": beta_true, "fisher": fisher, "information": oracle_information, "designs": designs}


def final_rr_estimator(mixture, beta_start, observations, panel_information, scale, reference_mu, n_conditional=64, seed=0, theta_bound=4.0, step_size=1.0):
    rng = np.random.default_rng(seed)
    projected = []
    H = np.zeros((12, 12))
    grouped = {}
    for panel, observed in observations:
        grouped.setdefault(panel, []).append(observed)
    for panel, observed_rows in grouped.items():
      batch = np.asarray(observed_rows)
      if n_conditional <= 4:
          completions = sample_conditional_batch(mixture, batch, panel, n_conditional * 4, int(rng.integers(2**31 - 1)))
          features = feature_map(completions, scale)
          logits = features @ beta_start
          weights = np.exp(logits - logits.max(axis=1, keepdims=True)); weights /= weights.sum(axis=1, keepdims=True)
          uniforms = rng.random((len(batch), n_conditional))
          choices = (uniforms[:, :, None] > np.cumsum(weights, axis=1)[:, None, :]).sum(axis=2)
          selected = np.take_along_axis(features, choices[:, :, None], axis=1)
          projected.extend(selected.mean(axis=1))
      else:
          completions = tilted_conditional_batch(mixture, beta_start, batch, panel, n_conditional, int(rng.integers(2**31 - 1)), scale)
          projected.extend(feature_map(completions, scale).mean(axis=1))
      H += len(batch) * panel_information[panel]
    if not projected:
        return np.asarray(beta_start).copy()
    U = np.sum(np.asarray(projected) - reference_mu, axis=0)
    # Finite conditional Monte Carlo can leave weak directions nearly singular;
    # the ridge is recorded in the run manifest and is not used to define Phi.
    H = (H + H.T) / 2
    step = np.linalg.solve(H + 1e-2 * np.eye(12), U)
    updated = np.asarray(beta_start) + step_size * step
    return np.clip(updated, -theta_bound, theta_bound)


def run_replication(mixture, scale, panels, budget, seed, reference_size=4000, information_samples=256, conditional_samples=32, prepared=None):
    prepared = prepared or prepare_s1_oracle(mixture, scale, panels, seed, reference_size, information_samples, conditional_samples)
    reference = prepared["reference"]
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
        observations = []
        pilot_counts = balanced_pilot_counts(panels, pilot_budget)
        panel_info_map = {panel: info for panel, info in zip(panels, oracle_information)}
        pilot_cursor = 0
        pilot_observations = []
        pilot_cursor = 0
        for panel, count in zip(panels, pilot_counts):
            for row in target_full[pilot_cursor : pilot_cursor + count]:
                pilot_observations.append((panel, row[list(panel)]))
            pilot_cursor += count
        main_cursor = pilot_budget
        for panel, count in zip(panels, counts):
            for row in target_full[main_cursor : main_cursor + count]:
                observations.append((panel, row[list(panel)]))
            main_cursor += count
        pilot_mu, pilot_rho = pilot_ht_moment(pilot_observations, pilot_counts, panels, scale, reference)
        beta_hat = solve_pilot_beta(pilot_mu, reference, scale)
        observations = pilot_observations + observations
        update_diagnostics = [{"step": "pilot", "pilot_budget": pilot_budget, "beta_norm": float(np.linalg.norm(beta_hat)), "rho_min": float(np.min(pilot_rho[pilot_rho > 0])) if np.any(pilot_rho > 0) else 0.0}]
        observed_H = sum((panel_info_map[panel] for panel, _ in observations), start=np.zeros((12, 12)))
        lambda_min_H = float(np.linalg.eigvalsh((observed_H + observed_H.T) / 2).min())
        for update in range(2):
            mu_beta, _ = tilted_moments(beta_hat, reference, scale)
            beta_next = final_rr_estimator(mixture, beta_hat, observations, panel_info_map, scale, mu_beta, conditional_samples, seed + 4 + update, step_size=1.0)
            update_diagnostics.append({"step": update, "lambda_min_H": lambda_min_H, "step_norm": float(np.linalg.norm(beta_next - beta_hat)), "projected": bool(np.any(np.abs(beta_next) >= 4.0)), "pilot_budget": pilot_budget})
            beta_hat = beta_next
        kl = max(0.0, full_target_kl(beta_true, beta_hat, target_reference, scale))
        rows.append({"policy": name, "budget": budget, "allocated_observations": int(counts.sum() + pilot_counts.sum()), "pilot_budget": int(pilot_counts.sum()), "seed": seed, "beta_true_norm": float(np.linalg.norm(beta_true)), "beta_hat_norm": float(np.linalg.norm(beta_hat)), "kl": kl, "B_kl": budget * kl, "design_ratio": float(kl / max(rr_phi / (2 * budget), 1e-12)), "target_draw_seed": seed + 3, "update_diagnostics": update_diagnostics})
    return rows
