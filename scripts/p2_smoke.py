"""Run P2 policy and solver smoke diagnostics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from rr_gid_cn.policies import covariance_information, conditional_information, frank_wolfe, round_cost_share, uniform_probabilities
from rr_gid_cn.synthetic_oracle import feature_map, make_frozen_mixture, reference_scale, sample_full


def main() -> None:
    mix = make_frozen_mixture()
    scale = reference_scale(mix, n=4000, seed=2026)
    panels = tuple((i, i + 1) for i in range(6))
    reference = sample_full(mix, 4000, seed=2027)
    F, oracle_I = conditional_information(mix, np.zeros(12), panels, reference, scale, n_tilted=80, n_conditional=16, seed=2028)
    costs = np.ones(len(panels))
    safe = uniform_probabilities(len(panels))
    p, gap, iterations = frank_wolfe(F, oracle_I, costs, safe, tolerance=1e-6)
    allocation = round_cost_share(p, 100, costs.astype(int))
    A_I = covariance_information(reference, panels)
    A_F = np.cov(reference, rowvar=False)
    A_F = A_F[:2, :2]
    summary = {"stage": "P2", "panel_count": len(panels), "budget": 100, "oracle_fw_gap": gap, "fw_iterations": iterations, "oracle_objective": float(np.trace(F @ np.linalg.pinv(np.tensordot(p, oracle_I, axes=(0, 0))))), "spent": allocation.spent, "counts": allocation.counts.tolist(), "A_OSQD_information_shape": list(A_I.shape), "feature_fisher_min_eigenvalue": float(np.linalg.eigvalsh(F).min())}
    payload = json.dumps(summary, sort_keys=True, indent=2) + "\n"
    Path("results").mkdir(exist_ok=True)
    Path("results/p2_policy_summary.json").write_text(payload, encoding="utf-8")
    summary["summary_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    Path("results/p2_policy_summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

