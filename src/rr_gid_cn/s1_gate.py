"""Synthetic S1 oracle-gate evaluation with panel-specific exact conditional scores."""

from __future__ import annotations

import numpy as np

from .policies import covariance_information, frank_wolfe, uniform_probabilities
from .synthetic_oracle import beta_direction_and_scale, feature_map, full_target_kl, sample_conditional, sample_conditional_batch, sample_full, tilted_moments, tilted_sample_from_reference


def exact_panel_information(mixture, beta, panels, reference_pool, scale, n_tilted=256, n_conditional=64, seed=0):
    rng = np.random.default_rng(seed)
    tilted = tilted_sample_from_reference(beta, reference_pool, n_tilted, int(rng.integers(2**31 - 1)), scale)
    phi = feature_map(tilted, scale)
    mu = phi.mean(0)
    fisher = np.cov(phi, rowvar=False)
    infos = []
    for panel in panels:
        projected = []
        for row in tilted:
            completed = sample_conditional(mixture, row[list(panel)], panel, n_conditional * 4, int(rng.integers(2**31 - 1)))
            logits = feature_map(completed, scale) @ beta
            weights = np.exp(logits - logits.max()); weights /= weights.sum()
            chosen = completed[rng.choice(len(completed), size=n_conditional, replace=True, p=weights)]
            projected.append(feature_map(chosen, scale).mean(0) - mu)
        infos.append(np.cov(np.asarray(projected), rowvar=False))
    return fisher, np.asarray(infos)


def policy_designs(reference, panels, fisher, oracle_information):
    costs = np.ones(len(panels))
    uniform = uniform_probabilities(len(panels))
    a_info = covariance_information(reference, panels)
    a_p, _, _ = frank_wolfe(np.eye(2), a_info, costs, uniform, tolerance=0.2, max_iter=300)
    rr_p, _, _ = frank_wolfe(fisher, oracle_information, costs, uniform, tolerance=0.2, max_iter=300)
    return {"Uniform SQD": uniform, "A-OSQD": a_p, "oracle RR-GID": rr_p}


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
      completions = sample_conditional_batch(mixture, batch, panel, n_conditional * 4, int(rng.integers(2**31 - 1)))
      features = feature_map(completions, scale)
      logits = features @ beta_start
      weights = np.exp(logits - logits.max(axis=1, keepdims=True))
      weights /= weights.sum(axis=1, keepdims=True)
      uniforms = rng.random((len(batch), n_conditional))
      choices = (uniforms[:, :, None] > np.cumsum(weights, axis=1)[:, None, :]).sum(axis=2)
      selected = np.take_along_axis(features, choices[:, :, None], axis=1)
      projected.extend(selected.mean(axis=1))
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
    target_full = tilted_sample_from_reference(beta_true, target_reference, budget, seed + 3, scale)
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
        pilot_counts = np.zeros(len(panels), dtype=int)
        pilot_expected = pilot_budget * uniform_probabilities(len(panels))
        pilot_counts[:] = np.floor(pilot_expected).astype(int)
        pilot_remainder = pilot_budget - int(pilot_counts.sum())
        if pilot_remainder:
            pilot_counts[np.argsort(pilot_expected - pilot_counts)[-pilot_remainder:]] += 1
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
        # Pilot moment initialization uses the global reference Fisher rather
        # than sparse panel-specific curvature, matching the HT moment step.
        pilot_info_map = {panel: fisher for panel in panels}
        beta_hat = np.zeros_like(beta_true)
        pilot_mu, _ = tilted_moments(beta_hat, reference, scale)
        for pilot_step in range(2):
            pilot_mu, _ = tilted_moments(beta_hat, reference, scale)
            beta_hat = final_rr_estimator(mixture, beta_hat, pilot_observations, pilot_info_map, scale, pilot_mu, conditional_samples, seed + 3 + pilot_step, step_size=1.0)
        observations = pilot_observations + observations
        update_diagnostics = [{"step": "pilot", "pilot_budget": pilot_budget, "beta_norm": float(np.linalg.norm(beta_hat))}]
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
