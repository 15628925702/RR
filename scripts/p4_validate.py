"""Validate frozen P4 formal rows without modifying source artifacts."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from rr_gid_cn.p4_integrity import (
    FORMAL_POLICIES,
    load_seed_manifest,
    sha256_file,
    validate_expected_grid,
    validate_experiment_mode,
)


def validate_rows(
    rows,
    *,
    budgets,
    replications,
    policies=FORMAL_POLICIES,
    conditional_method,
    experiment_mode,
    integration_tolerance,
    design_ratio_tolerance,
    fw_tolerance,
    artifact_sha256,
    config_sha256,
    source_sha256,
    manifest=None,
):
    failures = []
    try:
        validate_expected_grid(rows, budgets, replications, policies)
    except ValueError as exc:
        failures.append(str(exc))
    for row in rows:
        key = (int(row["budget"]), int(row["replication"]))
        label = f"{key[0]}/{key[1]}/{row['policy']}"
        if int(row.get("allocated_observations", -1)) != key[0]:
            failures.append(f"{label}: budget mismatch")
        raw = float(row.get("kl_raw", np.nan))
        if not np.isfinite(raw):
            failures.append(f"{label}: nonfinite kl_raw")
        elif raw < -float(integration_tolerance):
            failures.append(
                f"{label}: kl_raw {raw} below integration tolerance {-float(integration_tolerance)}"
            )
        if row.get("conditional_method") != conditional_method:
            failures.append(
                f"{label}: method {row.get('conditional_method')} != {conditional_method}"
            )
        if row.get("experiment_mode") != experiment_mode:
            failures.append(
                f"{label}: experiment_mode {row.get('experiment_mode')} != {experiment_mode}"
            )
        if row.get("artifact_sha256") != artifact_sha256:
            failures.append(f"{label}: artifact hash mismatch")
        if row.get("config_sha256") != config_sha256:
            failures.append(f"{label}: config hash mismatch")
        if row.get("source_sha256") != source_sha256:
            failures.append(f"{label}: code hash mismatch")
        if manifest is not None:
            entry = manifest.get(key)
            if entry is None:
                failures.append(f"{label}: missing manifest entry")
            elif int(row.get("target_draw_seed", -1)) != int(entry["target_draw_seed"]):
                failures.append(f"{label}: target seed does not match manifest")
        if row["policy"] == "oracle RR-GID":
            ratio = float(row.get("design_ratio_main", np.nan))
            if not np.isfinite(ratio) or abs(ratio - 1.0) > float(design_ratio_tolerance):
                failures.append(f"{label}: oracle design_ratio_main {ratio} != 1")
        gap = float(row.get("fw_gap", np.nan))
        if not np.isfinite(gap) or gap > float(fw_tolerance):
            failures.append(f"{label}: FW gap {gap} exceeds {fw_tolerance}")
        if "design_ratio" in row:
            failures.append(f"{label}: unlabelled legacy design_ratio field is forbidden")
        if not isinstance(row.get("pilot_schedule"), dict):
            failures.append(f"{label}: missing structured pilot_schedule")
        if sum(row.get("pilot_counts", [])) != int(row.get("pilot_budget", -1)):
            failures.append(f"{label}: pilot count mismatch")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--required-replications", type=int, default=None)
    parser.add_argument("--formal", action="store_true", help="PDF S1 formal gates; not a diagnostic check")
    parser.add_argument("--require-method", default=None)
    args = parser.parse_args()
    rows = [json.loads(line) for path in args.paths for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.config is None or args.artifact is None:
        raise SystemExit("--config and --artifact are required; oracle constants are never hard-coded")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))["p4"]
    validate_experiment_mode(cfg, formal=args.formal)
    with args.artifact.open("rb") as stream:
        artifact = pickle.load(stream)
    oracle_constant = artifact.get("oracle_constant")
    if not oracle_constant:
        raise SystemExit("artifact is missing its oracle constant/FW certificate")
    config_sha256 = sha256_file(args.config)
    artifact_sha256 = sha256_file(args.artifact)
    source_paths = (
        Path("src/rr_gid_cn/s1_gate.py"),
        Path("src/rr_gid_cn/p4_integrity.py"),
        Path("scripts/p4_formal_run.py"),
        Path("scripts/p4_validate.py"),
    )
    source_sha256 = {path.as_posix(): sha256_file(path) for path in source_paths}
    manifest_path = args.manifest or Path(cfg["target_manifest"])
    manifest, _ = load_seed_manifest(manifest_path)
    budgets = tuple(map(int, cfg["budgets"]))
    required_replications = int(args.required_replications or cfg["replications"])
    replications = tuple(range(required_replications))
    method = args.require_method or str(cfg["conditional_method"])
    failures = validate_rows(
        rows,
        budgets=budgets,
        replications=replications,
        policies=tuple(cfg.get("policies", FORMAL_POLICIES)),
        conditional_method=method,
        experiment_mode=str(cfg["experiment_mode"]),
        integration_tolerance=float(cfg.get("integration_tolerance", 0.0)),
        design_ratio_tolerance=float(cfg.get("design_ratio_tolerance", 1e-10)),
        fw_tolerance=float(cfg.get("fw_tolerance", 1e-6)),
        artifact_sha256=artifact_sha256,
        config_sha256=config_sha256,
        source_sha256=source_sha256,
        manifest=manifest,
    )
    by_policy: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        by_policy[(row["budget"], row["policy"])].append(float(row["B_kl_raw"]))
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
            "risk_ratio_raw_mean": mean / float(oracle_constant["half_phi_oracle"]),
            "negative_count": int(np.sum(arr < 0)),
            "minimum_B_kl_raw": float(arr.min()),
            "integration_tolerance": float(cfg.get("integration_tolerance", 0.0)),
        }
    by_budget = {
        budget: len({int(row["replication"]) for row in rows if int(row["budget"]) == budget})
        for budget in budgets
    }
    report = {
        "rows": len(rows),
        "replications_by_budget": by_budget,
        "oracle_constant": oracle_constant,
        "summary": summary,
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
