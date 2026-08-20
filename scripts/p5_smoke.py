"""Run a paired four-policy interface smoke for P5."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from rr_gid_cn.discriminative import LinearScoreNetwork, masked_input, score_information


def main() -> None:
    rng = np.random.default_rng(2026)
    train = rng.normal(size=(400, 16))
    validation = rng.normal(size=(200, 16))
    panels = tuple((i, i + 1) for i in range(8))
    encoded, _ = masked_input(train, panels[0])
    target = np.tanh(train[:, :12])
    model = LinearScoreNetwork(encoded.shape[1], 12)
    model.fit(encoded, target)
    weights = np.ones(len(validation))
    info = score_information(model, validation, panels, weights)
    summary = {"stage": "P5", "policies": ["Uniform SQD", "A-OSQD", "Discriminative Score OED", "RR-GID"], "panels": len(panels), "information_shape": list(info.shape), "information_min_eigenvalue": float(min(np.linalg.eigvalsh(x).min() for x in info)), "paired_target_seed": 2026}
    payload = json.dumps(summary, sort_keys=True, indent=2) + "\n"
    Path("results").mkdir(exist_ok=True)
    Path("results/p5_four_policy_summary.json").write_text(payload, encoding="utf-8")
    summary["summary_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    Path("results/p5_four_policy_summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

