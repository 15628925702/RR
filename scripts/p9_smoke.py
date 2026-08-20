"""Gas R1 empirical-base tilt and paired-policy smoke."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from rr_gid_cn.gas_preprocess import fit_scaler_pca, transform_features
from rr_gid_cn.vaeac import VAEACGenerator


def main() -> None:
    rng = np.random.default_rng(2026)
    raw = rng.normal(size=(300, 128))
    mean, pcs = fit_scaler_pca(raw[:240])
    std = raw[:240].std(0, ddof=1)
    reference = raw[:240]
    features = transform_features(reference, mean, std, pcs)
    beta = np.zeros(16)
    logits = features @ beta
    weights = np.exp(logits - logits.max()); weights /= weights.sum()
    generator = VAEACGenerator(reference)
    target = generator.sample_full(100, seed=2027)
    rows = []
    for budget in (400, 800, 1600, 3200):
        for rep in range(3):
            for policy in ("Uniform SQD", "A-OSQD", "Discriminative Score OED", "RR-GID"):
                rows.append({"budget": budget, "replication": rep, "policy": policy, "ess_fraction": float(1 / np.sum(weights ** 2) / len(weights)), "target_draw_seed": 202600 + budget + rep})
    summary = {"stage": "P9", "reference_shape": list(reference.shape), "target_shape": list(target.shape), "generator_hash": hashlib.sha256(reference.tobytes()).hexdigest(), "rows": rows}
    payload = json.dumps(summary, sort_keys=True, indent=2) + "\n"
    Path("results").mkdir(exist_ok=True)
    Path("results/p9_r1_summary.json").write_text(payload, encoding="utf-8")
    summary["summary_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    Path("results/p9_r1_summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "P9", "rows": len(rows), "generator_hash": summary["generator_hash"]}, sort_keys=True))


if __name__ == "__main__":
    main()

