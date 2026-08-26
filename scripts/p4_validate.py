"""Validate frozen P4 formal rows without modifying source artifacts."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def _pilot_budget(budget: int) -> int:
    return min(int(math.ceil(10.0 * float(budget) ** (1.0 / 3.0))), int(budget))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--required-replications", type=int, default=200)
    parser.add_argument("--formal", action="store_true", help="PDF S1 formal gates; not a diagnostic check")
    parser.add_argument("--require-method", default=None)
    parser.add_argument("--oracle-half-phi", type=float, default=33.52962983788052)
    parser.add_argument("--oracle-ratio-max-largest", type=float, default=1.25)
    args = parser.parse_args()
    rows = [json.loads(line) for path in args.paths for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    by_policy: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["budget"], row["replication"])].append(row)
        by_policy[(row["budget"], row["policy"])].append(float(row["B_kl_raw"]))
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
            if not np.isfinite(row.get("kl_raw", row["kl"])):
                failures.append(f"{budget}/{replication}/{row['policy']}: nonfinite kl_raw")
            if args.formal or args.require_method:
                method = args.require_method or "rejection"
                actual = row.get("conditional_method")
                if actual != method:
                    failures.append(f"{budget}/{replication}/{row['policy']}: method {actual} != {method}")
            if args.formal:
                expected_pilot = _pilot_budget(budget)
                if int(row.get("pilot_budget", -1)) != expected_pilot:
                    failures.append(f"{budget}/{replication}/{row['policy']}: pilot {row.get('pilot_budget')} != {expected_pilot}")
                for diag in row.get("update_diagnostics") or []:
                    if "lambda_min_H" in diag and not (np.isfinite(diag["lambda_min_H"]) and float(diag["lambda_min_H"]) > 0):
                        failures.append(f"{budget}/{replication}/{row['policy']}: nonpositive lambda_min_H")
    missing = {budget: args.required_replications - count for budget, count in by_budget.items() if count < args.required_replications}
    summary = {}
    for (budget, policy), values in sorted(by_policy.items()):
        arr = np.asarray(values, dtype=float)
        mean = float(arr.mean())
        se = float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else float("nan")
        summary[f"{budget}/{policy}"] = {
            "n": len(arr),
            "B_kl_mean": mean,
            "B_kl_se": se,
            "B_kl_ci95": [mean - 1.96 * se, mean + 1.96 * se] if len(arr) > 1 else None,
            "design_ratio_mean": mean / args.oracle_half_phi if policy == "oracle RR-GID" else None,
        }
    if args.formal and by_budget:
        largest = max(by_budget)
        oracle = by_policy.get((largest, "oracle RR-GID"), [])
        if len(oracle) >= args.required_replications:
            ratio = float(np.mean(oracle)) / args.oracle_half_phi
            if not (0.5 <= ratio <= args.oracle_ratio_max_largest):
                failures.append(
                    f"B={largest} oracle RR-GID mean design ratio {ratio:.3f} outside [0.5, {args.oracle_ratio_max_largest}]"
                )
        j_groups = defaultdict(list)
        for row in rows:
            if row.get("budget") == 8000 and row.get("policy") == "oracle RR-GID":
                j_groups[int(row.get("scoring_steps", 2))].append(float(row["B_kl_raw"]))
        if 0 in j_groups and 2 in j_groups:
            if float(np.mean(j_groups[2])) > float(np.mean(j_groups[0])) * 1.05:
                failures.append("B=8000 J=2 oracle B_kl mean exceeds J=0")
    report = {
        "rows": len(rows),
        "replications_by_budget": dict(sorted(by_budget.items())),
        "missing": missing,
        "summary": summary,
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures or missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
