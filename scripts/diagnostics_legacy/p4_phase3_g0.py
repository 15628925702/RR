"""G0 oracle-start sanity check on the frozen Phase-2 gold artifact."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

from rr_gid_cn.oracle_measure import (
    ConditionalQMC,
    FullLawQMC,
    InformationQMC,
    OracleMeasure,
    canonical_sha256,
    mixture_sha256,
)
from rr_gid_cn.p4_integrity import load_seed_manifest, sha256_file
from rr_gid_cn.s1_gate import (
    design_metrics,
    gold_oracle_start_step,
    largest_remainder_counts,
)
from rr_gid_cn.synthetic_oracle import (
    all_pairs,
    make_frozen_mixture,
    tilted_full_sample,
)
from scripts.p4_phase2_score_centering import load_verified_base_artifact


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def validate_g0_config(payload: dict) -> None:
    if payload.get("phase") != 3 or payload.get("ladder") != "G0":
        raise ValueError("G0 config requires phase=3 and ladder=G0")
    if payload.get("not_formal") is not True:
        raise ValueError("G0 is a ladder diagnostic and must set not_formal=true")
    if payload.get("schema_version") != "p4-phase3-g0-config-v1":
        raise ValueError("unsupported G0 config schema")
    if payload.get("policy") != "oracle RR-GID":
        raise ValueError("G0 uses only oracle RR-GID allocation")
    if int(payload["budget"]) <= 0:
        raise ValueError("G0 budget must be positive")


def g0_gate(rows: list[dict], payload: dict, half_phi: float) -> dict:
    if not rows:
        raise ValueError("G0 gate requires completed rows")
    j0 = [row for row in rows if int(row["scoring_steps"]) == 0]
    j_pos = [row for row in rows if int(row["scoring_steps"]) > 0]
    design_tol = float(payload["design_ratio_tolerance"])
    design_ok = all(abs(row["design_ratio_main"] - 1.0) <= design_tol for row in rows)
    j0_ok = all(
        abs(float(row["kl_raw"])) <= float(row["kl_numerical_tolerance"])
        and float(row["beta_error_norm"]) == 0.0
        for row in j0
    ) if j0 else True
    chol_ok = all(
        step.get("linear_algebra") == "cholesky_solve"
        for row in rows
        for step in row.get("update_diagnostics", [])
        if isinstance(step.get("step"), int)
    )
    kl_floor_ok = all(
        float(row["kl_raw"]) >= -float(row["kl_numerical_tolerance"]) for row in rows
    )
    risks = np.asarray([row["risk_ratio_raw"] for row in j_pos], dtype=float) if j_pos else np.array([])
    mean_risk = float(risks.mean()) if len(risks) else None
    risk_ok = mean_risk is not None and np.isfinite(mean_risk) and mean_risk < float(
        payload["risk_ratio_mean_max"]
    )
    passed = bool(design_ok and j0_ok and chol_ok and kl_floor_ok and risk_ok)
    return {
        "passed": passed,
        "design_ratio_main_is_one": bool(design_ok),
        "j0_stays_at_beta_star": bool(j0_ok),
        "gold_H_cholesky": bool(chol_ok),
        "kl_raw_above_tolerance": bool(kl_floor_ok),
        "j_positive_count": int(len(j_pos)),
        "mean_risk_ratio_raw": mean_risk,
        "risk_ratio_mean_max": float(payload["risk_ratio_mean_max"]),
        "half_phi": float(half_phi),
    }


def _build_observations(target_full, panels, counts):
    observations = []
    cursor = 0
    for panel, count in zip(panels, counts):
        for row in target_full[cursor : cursor + int(count)]:
            observations.append((panel, row[list(panel)]))
        cursor += int(count)
    return observations


def run_g0_replication(
    oracle,
    arrays,
    payload,
    metadata,
    seed_entry,
    scoring_steps: int,
):
    beta_true = np.asarray(arrays["beta_true"], dtype=float)
    p_star = np.asarray(arrays["p_star"], dtype=float)
    information = np.asarray(arrays["information_projected"], dtype=float)
    fisher = np.asarray(arrays["F_projected"], dtype=float)
    panels = all_pairs()
    budget = int(payload["budget"])
    counts = largest_remainder_counts(p_star, budget)
    target_full = tilted_full_sample(
        oracle.mixture,
        beta_true,
        budget,
        int(seed_entry["target_draw_seed"]),
        scale=oracle.scale,
        feature_fn=oracle.feature_fn,
    )
    observations = _build_observations(target_full, panels, counts)
    beta_hat = beta_true.copy()
    update_diagnostics = [{
        "step": "oracle_start",
        "beta_error_norm": 0.0,
        "pilot_budget": 0,
    }]
    started = perf_counter()
    for step in range(int(scoring_steps)):
        beta_hat, step_diagnostics = gold_oracle_start_step(
            oracle,
            beta_hat,
            observations,
            panels,
            information,
            int(seed_entry["score_seed_root"]) + step,
            theta_bound=float(payload["theta_bound"]),
            step_size=float(payload["scoring_step_size"]),
        )
        update_diagnostics.append({
            "step": step,
            "beta_error_norm": float(np.linalg.norm(beta_hat - beta_true)),
            **step_diagnostics,
        })
    kl = oracle.kl(beta_true, beta_hat)
    half_phi = float(metadata["theory_constant_half_phi"])
    metrics = design_metrics(
        fisher, information, p_star, p_star, np.zeros_like(counts), counts
    )
    kl_raw = float(kl["raw"])
    return {
        "schema_version": "p4-phase3-g0-row-v1",
        "phase": 3,
        "ladder": "G0",
        "not_formal": True,
        "policy": "oracle RR-GID",
        "budget": budget,
        "scoring_steps": int(scoring_steps),
        "pilot_budget": 0,
        "allocated_observations": int(counts.sum()),
        "main_counts": counts.tolist(),
        "replication_seed": int(seed_entry["replication_seed"]),
        "target_draw_seed": int(seed_entry["target_draw_seed"]),
        "score_seed_root": int(seed_entry["score_seed_root"]),
        "beta_error_norm": float(np.linalg.norm(beta_hat - beta_true)),
        "kl_raw": kl_raw,
        "kl_numerical_tolerance": float(kl["numerical_tolerance"]),
        "B_kl_raw": float(budget * kl_raw),
        "risk_ratio_raw": float(budget * kl_raw / half_phi),
        **metrics,
        "update_diagnostics": update_diagnostics,
        "elapsed_seconds": float(perf_counter() - started),
        "beta_hat": beta_hat.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/p4_phase3_g0.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-replications", type=int, default=None)
    parser.add_argument("--rep-range", type=int, nargs=2, default=None)
    parser.add_argument("--scoring-steps", type=int, nargs="+", default=None)
    args = parser.parse_args()
    root = REPOSITORY_ROOT
    config_path = _resolve(root, args.config)
    out = _resolve(root, args.out)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))["p4_phase3_g0"]
    validate_g0_config(payload)
    metadata, arrays, verified_paths = load_verified_base_artifact(payload, root)
    half_phi = float(metadata["theory_constant_half_phi"])
    manifest_path = _resolve(root, payload["target_manifest"])
    manifest, manifest_sha256 = load_seed_manifest(manifest_path)
    replications = int(payload["replications"])
    if args.max_replications is not None:
        replications = min(replications, int(args.max_replications))
    start, end = (0, replications) if args.rep_range is None else args.rep_range
    end = min(end, replications)
    steps_list = args.scoring_steps or list(payload["scoring_steps"])
    requested = {
        (int(payload["budget"]), int(rep))
        for rep in range(start, end)
    }
    missing = sorted(requested - set(manifest))
    if missing:
        raise ValueError(f"G0 seed manifest missing entries: {missing[:5]}")
    mixture = make_frozen_mixture(
        seed=int(metadata["mixture_seed"]), alpha=float(metadata["alpha"])
    )
    if mixture_sha256(mixture) != metadata["mixture_parameter_hash"]:
        raise ValueError("rebuilt mixture does not match gold artifact")
    integration = metadata["integration"]
    oracle = OracleMeasure(
        mixture,
        arrays["scale"],
        FullLawQMC(**integration["full"]),
        ConditionalQMC(**integration["conditional"]),
        InformationQMC(**integration["information"]),
    )
    if out.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {out} without --resume")
    out.mkdir(parents=True, exist_ok=True)
    jsonl_path = out / "rows.jsonl"
    done = set()
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((int(row["replication"]), int(row["scoring_steps"])))
    print(
        json.dumps({
            "event": "g0_start",
            "out": str(out),
            "replications": [start, end],
            "scoring_steps": steps_list,
            "already_done": len(done),
        }, sort_keys=True),
        flush=True,
    )
    with jsonl_path.open("a", encoding="utf-8") as stream:
        for replication in range(start, end):
            seed_entry = manifest[(int(payload["budget"]), int(replication))]
            for scoring_steps in steps_list:
                key = (int(replication), int(scoring_steps))
                if key in done:
                    print(json.dumps({"event": "skip", "replication": replication, "J": scoring_steps}), flush=True)
                    continue
                row = run_g0_replication(
                    oracle, arrays, payload, metadata, seed_entry, int(scoring_steps)
                )
                row["replication"] = int(replication)
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                done.add(key)
                print(json.dumps({
                    "event": "row",
                    "replication": replication,
                    "J": scoring_steps,
                    "risk_ratio_raw": row["risk_ratio_raw"],
                    "kl_raw": row["kl_raw"],
                    "design_ratio_main": row["design_ratio_main"],
                    "elapsed_seconds": row["elapsed_seconds"],
                }, sort_keys=True), flush=True)
    rows = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gate = g0_gate(rows, payload, half_phi)
    diagnostics = {
        "schema_version": "p4-phase3-g0-diagnostics-v1",
        "phase": 3,
        "ladder": "G0",
        "not_formal": True,
        "budget": int(payload["budget"]),
        "replications_completed": sorted({int(row["replication"]) for row in rows}),
        "scoring_steps": sorted({int(row["scoring_steps"]) for row in rows}),
        "gate": gate,
        "g0_passed": bool(gate["passed"]),
        "provenance": {
            "config_sha256": sha256_file(config_path),
            "config_canonical_sha256": canonical_sha256(payload),
            "manifest_sha256": manifest_sha256,
            "base_paths": {
                key: str(value.relative_to(root)).replace("\\", "/")
                for key, value in verified_paths.items()
            },
            "code_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "written_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    (out / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"event": "g0_done", **gate}, sort_keys=True), flush=True)
    if any(int(step) > 0 for step in steps_list) and not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
