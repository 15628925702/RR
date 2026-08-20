"""Run deterministic P1 oracle diagnostics and write a machine-readable summary."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from rr_gid_cn.synthetic_oracle import conditional_moments, feature_map, make_frozen_mixture, reference_scale, sample_conditional, sample_full


def main() -> None:
    mix = make_frozen_mixture()
    scale = reference_scale(mix, n=6000, seed=2026)
    full = sample_full(mix, 6000, seed=2028)
    panel = (0, 1)
    observed = full[0, list(panel)]
    latent_mean, latent_cov = conditional_moments(mix, observed, panel)
    conditional = sample_conditional(mix, observed, panel, 20000, seed=2027)
    latent_draws = np.arcsinh(mix.alpha * conditional[:, [i for i in range(16) if i not in panel]]) / mix.alpha
    mean_error = float(np.linalg.norm(latent_draws.mean(0) - latent_mean))
    sample_cov = np.cov(latent_draws, rowvar=False)
    cov_error = float(np.linalg.norm(sample_cov - latent_cov, ord="fro"))
    cov_relative_error = cov_error / float(np.linalg.norm(latent_cov, ord="fro"))
    mean_se = np.sqrt(np.diag(sample_cov) / len(latent_draws))
    mean_zscore = float(np.max(np.abs((latent_draws.mean(0) - latent_mean) / np.maximum(mean_se, 1e-12))))
    phi = feature_map(full, scale)
    centered = phi - phi.mean(0)
    fisher = centered.T @ centered / (len(phi) - 1)
    summary = {
        "stage": "P1",
        "mixture_seed": 2026,
        "dimension": 16,
        "components": 4,
        "panel_count": 120,
        "conditional_panel": [0, 1],
        "conditional_latent_mean_l2_error": mean_error,
        "conditional_latent_cov_fro_error": cov_error,
        "conditional_latent_cov_relative_error": cov_relative_error,
        "conditional_latent_mean_max_zscore": mean_zscore,
        "fisher_min_eigenvalue": float(np.linalg.eigvalsh(fisher).min()),
        "reference_scale_sha256": hashlib.sha256(scale.tobytes()).hexdigest(),
        "full_sample_count": len(full),
        "conditional_sample_count": len(conditional),
    }
    out = Path(os.environ.get("RR_GID_RESULTS", "results"))
    out.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(summary, sort_keys=True, indent=2) + "\n"
    (out / "p1_oracle_summary.json").write_text(payload, encoding="utf-8")
    summary["summary_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    (out / "p1_oracle_summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
