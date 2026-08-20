"""Gas R2 natural-drift robustness result schema smoke."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def main() -> None:
    rows = []
    for campaign in ("batch7", "batches8_9", "batch10"):
        for budget in (400, 800, 1600, 3200):
            for rep in range(3):
                for policy in ("Uniform SQD", "A-OSQD", "Discriminative Score OED", "RR-GID"):
                    rows.append({"campaign": campaign, "budget": budget, "replication": rep, "policy": policy, "pool_role": "campaign_pool", "full_test_exposed_to_acquisition": False, "projection_loss": 0.0, "heldout_moment_rmse": 0.0, "c2st_auc": 0.5, "ess": 1.0, "conditional_acceptance": 1.0, "fw_gap": 0.0, "lambda_min_M": 1e-10})
    summary = {"stage": "P10", "campaigns": ["batch7", "batches8_9", "batch10"], "rows": rows, "formal_replications_required_per_budget_campaign": 50}
    payload = json.dumps(summary, sort_keys=True, indent=2) + "\n"
    Path("results").mkdir(exist_ok=True)
    Path("results/p10_r2_summary.json").write_text(payload, encoding="utf-8")
    summary["summary_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    Path("results/p10_r2_summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "P10", "rows": len(rows), "campaigns": len(summary["campaigns"])}, sort_keys=True))


if __name__ == "__main__":
    main()

