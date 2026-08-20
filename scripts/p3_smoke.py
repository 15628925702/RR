"""Deterministic P3 formal algorithm smoke."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from rr_gid_cn.policies import frank_wolfe, uniform_probabilities
from rr_gid_cn.rrgid import Theta, cross_completion_information, fisher_update
from rr_gid_cn.synthetic_oracle import feature_map, make_frozen_mixture, reference_scale, sample_full


def run(seed: int = 2026) -> dict:
    mix = make_frozen_mixture(seed)
    scale = reference_scale(mix, 1000, seed)
    panels = tuple((i, i + 1) for i in range(6))
    costs = np.ones(len(panels))
    pilot = sample_full(mix, 40, seed + 1)
    infos = np.asarray([cross_completion_information(mix, np.zeros(12), panel, pilot, scale, 4, seed + 10 + i) for i, panel in enumerate(panels)])
    F = np.cov(feature_map(pilot, scale), rowvar=False)
    p, gap, _ = frank_wolfe(F, infos, costs, uniform_probabilities(len(panels)), tolerance=0.2)
    beta = np.zeros(12)
    theta = Theta(np.full(12, -2.0), np.full(12, 2.0))
    trajectory = []
    for j in range(2):
        main = sample_full(mix, 40, seed + 100 + j)
        H = np.tensordot(p, infos, axes=(0, 0))
        U = feature_map(main, scale).mean(0) * 0.01
        beta = fisher_update(beta, H, U, theta)
        trajectory.append(beta.tolist())
    summary = {"stage": "P3", "budget": 200, "pilot_budget": 60, "spent_upper_bound": 200, "fw_gap": float(gap), "beta_trajectory": trajectory, "allocation_sum": float(p.sum()), "information_min_eigenvalue": float(min(np.linalg.eigvalsh(x).min() for x in infos))}
    payload = json.dumps(summary, sort_keys=True, indent=2) + "\n"
    Path("results").mkdir(exist_ok=True)
    Path("results/p3_rrgid_summary.json").write_text(payload, encoding="utf-8")
    summary["summary_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    Path("results/p3_rrgid_summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))

