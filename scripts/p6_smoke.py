"""Run P6 arbitrary-conditioning and generator diagnostics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from rr_gid_cn.synthetic_oracle import feature_map, make_frozen_mixture, sample_full
from rr_gid_cn.vaeac import VAEACGenerator


def main() -> None:
    mix = make_frozen_mixture()
    reference = sample_full(mix, 500, seed=2026)
    gen = VAEACGenerator(reference)
    observed = reference[0, [0, 1]]
    conditional = gen.sample_conditional(observed, (0, 1), 100, seed=2027)
    _, acceptance, ess = gen.tilted_sample(np.zeros(12), lambda x: feature_map(x), 100, seed=2028)
    summary = {"stage": "P6", "reference_shape": list(reference.shape), "conditional_shape": list(conditional.shape), "observed_coordinates_preserved": bool(np.all(conditional[:, [0, 1]] == observed)), "acceptance": acceptance, "ess_fraction": ess}
    payload = json.dumps(summary, sort_keys=True, indent=2) + "\n"
    Path("results").mkdir(exist_ok=True)
    Path("results/p6_vaeac_summary.json").write_text(payload, encoding="utf-8")
    summary["summary_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    Path("results/p6_vaeac_summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

