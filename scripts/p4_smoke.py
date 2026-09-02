"""Run paired Synthetic S1 oracle-gate smoke evaluations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rr_gid_cn.s1_gate import prepare_s1_oracle, run_replication
from rr_gid_cn.synthetic_oracle import all_pairs, make_frozen_mixture, reference_scale


def main() -> None:
    mixture = make_frozen_mixture(seed=2026, alpha=1.0)
    scale = reference_scale(mixture, 1000, 2026)
    panels = all_pairs()
    prepared = prepare_s1_oracle(mixture, scale, panels, seed=2026, reference_size=2000,
                                 information_samples=32, conditional_samples=8, large_reference_size=10000)
    rows = []
    for budget in (200, 400, 800):
        for rep in range(3):
            rows.extend(run_replication(mixture, scale, panels, budget, 202600 + budget + rep,
                                        prepared=prepared, lu=32, h_tilted=32, h_cond=8, kl_samples=1000))
    summary = {"stage": "P4", "alpha": 1.0, "budgets": [200, 400, 800], "replications_per_budget": 3, "rows": rows, "formal_replications_required": 50}
    payload = json.dumps(summary, sort_keys=True, indent=2) + "\n"
    Path("results").mkdir(exist_ok=True)
    Path("results/p4_s1_gate_summary.json").write_text(payload, encoding="utf-8")
    summary["summary_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    Path("results/p4_s1_gate_summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "P4", "rows": len(rows), "summary_sha256": summary["summary_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
