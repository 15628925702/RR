"""Create the immutable P4 paired target-draw seed manifest."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    budgets = [2000, 4000, 8000, 16000, 32000]
    rows = []
    for budget in budgets:
        for replication in range(200):
            rows.append({"budget": budget, "replication": replication, "target_draw_seed": 202600000 + budget * 1000 + replication})
    path = Path("experiments/p4_target_draw_manifest.json")
    path.write_text(json.dumps({"stage": "P4", "rows": rows}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} paired target-draw entries to {path}")


if __name__ == "__main__":
    main()
