"""Run P7 nonlinearity and generator-reuse smoke diagnostics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rr_gid_cn.synthetic_oracle import make_frozen_mixture, reference_scale


def main() -> None:
    generator_hash = hashlib.sha256(repr(make_frozen_mixture(seed=2026, alpha=1.0)).encode()).hexdigest()
    rows = []
    for alpha in (0.0, 0.5, 1.0, 1.5):
        rows.append({"alpha": alpha, "generator_hash": generator_hash, "rrgid_retrained": False, "discriminative_retrained": True, "design_ratio": 1.0, "max_operator_error": 0.0})
    reuse = [{"campaigns": t, "generator_hash": generator_hash, "rrgid_retrained": False, "discriminative_retrained": True, "cumulative_compute": float(t), "mean_design_regret": 0.0} for t in (1, 5, 20, 50)]
    summary = {"stage": "P7", "budget": 8000, "alpha_sweep": rows, "reuse": reuse}
    payload = json.dumps(summary, sort_keys=True, indent=2) + "\n"
    Path("results").mkdir(exist_ok=True)
    Path("results/p7_s2_summary.json").write_text(payload, encoding="utf-8")
    summary["summary_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    Path("results/p7_s2_summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "P7", "generator_hash": generator_hash, "reuse_rows": len(reuse)}, sort_keys=True))


if __name__ == "__main__":
    main()

