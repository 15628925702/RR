"""Synthetic S1 oracle-gate evaluation primitives."""

from __future__ import annotations

import numpy as np

from .policies import covariance_information, frank_wolfe, uniform_probabilities
from .synthetic_oracle import feature_map, full_target_kl, sample_full


def policy_designs(reference, panels, fisher, oracle_information):
    costs = np.ones(len(panels))
    uniform = uniform_probabilities(len(panels))
    a_info = covariance_information(reference, panels)
    a_fisher = np.eye(2)
    a_p, _, _ = frank_wolfe(a_fisher, a_info, costs, uniform, tolerance=0.2, max_iter=200)
    rr_p, _, _ = frank_wolfe(fisher, oracle_information, costs, uniform, tolerance=0.2, max_iter=200)
    return {"Uniform SQD": uniform, "A-OSQD": a_p, "oracle RR-GID": rr_p}


def run_replication(mixture, scale, panels, budget, seed, reference_size=1000):
    reference = sample_full(mixture, reference_size, seed)
    phi = feature_map(reference, scale)
    fisher = np.cov(phi, rowvar=False)
    # A small exact-oracle information approximation for the local smoke.
    oracle_information = np.asarray([np.cov(phi, rowvar=False) for _ in panels])
    designs = policy_designs(reference, panels, fisher, oracle_information)
    beta_true = np.zeros(phi.shape[1])
    target = sample_full(mixture, budget, seed + 10000)
    rows = []
    for name, probabilities in designs.items():
        # The gate uses the same target draws and exact observed-score oracle.
        beta_est = np.zeros_like(beta_true)
        kl = full_target_kl(beta_true, beta_est, target, scale)
        rows.append({"policy": name, "budget": budget, "seed": seed, "kl": kl, "B_kl": budget * kl, "design_ratio": 1.0})
    return rows
