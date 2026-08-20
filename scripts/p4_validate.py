"""Validate frozen P4 formal rows without modifying source artifacts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--required-replications", type=int, default=200)
    args = parser.parse_args()
    rows = [json.loads(line) for path in args.paths for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["budget"], row["replication"])].append(row)
    failures = []
    by_budget = defaultdict(int)
    for (budget, replication), group in grouped.items():
        by_budget[budget] += 1
        if len(group) != 3 or len({r["policy"] for r in group}) != 3:
            failures.append(f"{budget}/{replication}: incomplete policies")
        if len({r["target_draw_seed"] for r in group}) != 1:
            failures.append(f"{budget}/{replication}: unpaired target draw")
        for row in group:
            if row.get("allocated_observations") != budget:
                failures.append(f"{budget}/{replication}/{row['policy']}: budget mismatch")
            if not np.isfinite(row["kl"]) or row["kl"] < 0:
                failures.append(f"{budget}/{replication}/{row['policy']}: invalid KL")
    missing = {budget: args.required_replications - count for budget, count in by_budget.items() if count < args.required_replications}
    report = {"rows": len(rows), "replications_by_budget": dict(sorted(by_budget.items())), "missing": missing, "failures": failures}
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures or missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
