"""Create the immutable P4 paired target-draw seed manifest.

Replication count is the authorized formal cap: 50, not 200.
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    budgets = [2000, 4000, 8000, 16000, 32000]
    rows = []
    for budget in budgets:
        for replication in range(50):
            root = 202600000 + budget * 1000 + replication
            rows.append({
                "budget": budget,
                "replication": replication,
                "replication_seed": root,
                "target_draw_seed": root + 3,
                "pilot_or_design_seed": root + 11,
                "score_seed_root": root + 1000,
                "information_seed_root": root + 7000,
            })
    path = Path("experiments/p4_target_draw_manifest.json")
    path.write_text(json.dumps({"stage": "P4", "rows": rows}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} paired target-draw entries to {path}")


if __name__ == "__main__":
    main()
