"""Phase-3 ladder G1–G4 on the frozen Phase-2 gold artifact."""

from __future__ import annotations

import argparse
import json
import math
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
from rr_gid_cn.p4_integrity import compute_pilot_budget, load_seed_manifest, sha256_file
from rr_gid_cn.policies import frank_wolfe, uniform_probabilities
from rr_gid_cn.s1_gate import (
    active_panel_information_cross,
    balanced_pilot_counts,
    design_metrics,
    gold_oracle_start_step,
    largest_remainder_counts,
    panel_information_cross,
    pilot_ht_moment,
    solve_pilot_beta,
)
from rr_gid_cn.synthetic_oracle import (
    all_pairs,
    make_frozen_mixture,
    sample_full,
    tilted_full_sample,
    tilted_moments,
)
from scripts.p4_phase2_score_centering import load_verified_base_artifact


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _ladder_lu(payload: dict, budget: int) -> int:
    if payload.get("lu_schedule"):
        spec = payload["lu_schedule"]
        return max(1, int(math.ceil(float(spec["scale"]) * math.log2(float(budget) + float(spec["offset"])))))
    return int(payload.get("lu", 128))


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _build_observations(target_full, panels, counts):
    observations = []
    cursor = 0
    for panel, count in zip(panels, counts):
        for row in target_full[cursor : cursor + int(count)]:
            observations.append((panel, row[list(panel)]))
        cursor += int(count)
    return observations


def validate_ladder_config(payload: dict) -> None:
    if payload.get("phase") != 3:
        raise ValueError("ladder config requires phase=3")
    if payload.get("not_formal") is not True:
        raise ValueError("ladder runs are diagnostics and must set not_formal=true")
    if payload.get("schema_version") != "p4-phase3-ladder-config-v1":
        raise ValueError("unsupported ladder config schema")
    if payload.get("ladder") not in {"G1", "G2", "G3", "G4"}:
        raise ValueError("ladder must be G1, G2, G3, or G4")
    if payload["h_mode"] not in {"gold_current", "estimated"}:
        raise ValueError("unsupported h_mode")
    if payload["score_mode"] not in {"gold", "qmc", "rejection"}:
        raise ValueError("unsupported score_mode")
    if payload["allocation"] not in {"oracle", "plugin"}:
        raise ValueError("unsupported allocation")


def ladder_gate(rows: list[dict], payload: dict, half_phi: float) -> dict:
    if not rows:
        raise ValueError("ladder gate requires completed rows")
    max_j = int(payload["max_scoring_steps"])
    design_tol = float(payload["design_ratio_tolerance"])
    terminal = [row for row in rows if int(row["scoring_steps"]) == max_j]
    j0 = [row for row in rows if int(row["scoring_steps"]) == 0]
    if payload["allocation"] == "oracle":
        design_ok = all(abs(row["design_ratio_main"] - 1.0) <= design_tol for row in rows)
    else:
        design_ok = all(np.isfinite(row["design_ratio_main"]) for row in terminal)
    chol_ok = all(
        step.get("linear_algebra") == "cholesky_solve"
        for row in rows
        for step in row.get("update_diagnostics", [])
        if isinstance(step.get("step"), int)
    )
    kl_floor_ok = all(
        float(row["kl_raw"]) >= -float(row["kl_numerical_tolerance"]) for row in rows
    )
    risks = np.asarray([row["risk_ratio_raw"] for row in terminal], dtype=float)
    mean_risk = float(risks.mean()) if len(risks) else None
    risk_ok = mean_risk is not None and np.isfinite(mean_risk) and mean_risk < float(
        payload["risk_ratio_mean_max"]
    )
    j0_risk = float(np.mean([row["risk_ratio_raw"] for row in j0])) if j0 else None
    passed = bool(design_ok and chol_ok and kl_floor_ok and risk_ok)
    return {
        "passed": passed,
        "design_ok": bool(design_ok),
        "gold_H_cholesky": bool(chol_ok),
        "kl_raw_above_tolerance": bool(kl_floor_ok),
        "terminal_j": max_j,
        "terminal_count": int(len(terminal)),
        "mean_terminal_risk_ratio_raw": mean_risk,
        "mean_j0_risk_ratio_raw": j0_risk,
        "risk_ratio_mean_max": float(payload["risk_ratio_mean_max"]),
        "half_phi": float(half_phi),
        "mean_terminal_design_ratio_main": None if not terminal else float(
            np.mean([row["design_ratio_main"] for row in terminal])
        ),
    }


def _row_payload(
    *,
    payload,
    budget,
    scoring_steps,
    beta_hat,
    beta_true,
    kl,
    half_phi,
    metrics,
    counts,
    pilot_counts,
    update_diagnostics,
    seed_entry,
    probabilities,
    elapsed,
    fisher,
):
    kl_raw = float(kl["raw"])
    error = np.asarray(beta_hat) - np.asarray(beta_true)
    chol_f = np.linalg.cholesky((fisher + fisher.T) / 2)
    whitened = np.linalg.solve(chol_f, error)
    return {
        "schema_version": "p4-phase3-ladder-row-v1",
        "phase": 3,
        "ladder": payload["ladder"],
        "not_formal": True,
        "policy": "oracle RR-GID" if payload["allocation"] == "oracle" else "plugin RR-GID",
        "h_mode": payload["h_mode"],
        "score_mode": payload["score_mode"],
        "allocation": payload["allocation"],
        "budget": int(budget),
        "scoring_steps": int(scoring_steps),
        "pilot_budget": int(pilot_counts.sum()),
        "pilot_counts": pilot_counts.tolist(),
        "main_counts": counts.tolist(),
        "allocated_observations": int(pilot_counts.sum() + counts.sum()),
        "replication_seed": int(seed_entry["replication_seed"]),
        "target_draw_seed": int(seed_entry["target_draw_seed"]),
        "score_seed_root": int(seed_entry["score_seed_root"]),
        "beta_error_norm": float(np.linalg.norm(error)),
        "pilot_fisher_error": float(np.linalg.norm(whitened)),
        "kl_raw": kl_raw,
        "kl_numerical_tolerance": float(kl["numerical_tolerance"]),
        "B_kl_raw": float(budget * kl_raw),
        "risk_ratio_raw": float(budget * kl_raw / half_phi),
        "allocation_probabilities": np.asarray(probabilities, dtype=float).tolist(),
        **metrics,
        "update_diagnostics": update_diagnostics,
        "elapsed_seconds": float(elapsed),
        "beta_hat": np.asarray(beta_hat, dtype=float).tolist(),
    }


def run_ladder_replication(
    oracle,
    arrays,
    payload,
    metadata,
    seed_entry,
    budget: int,
    reference: np.ndarray,
):
    beta_true = np.asarray(arrays["beta_true"], dtype=float)
    p_star = np.asarray(arrays["p_star"], dtype=float)
    information = np.asarray(arrays["information_projected"], dtype=float)
    fisher = np.asarray(arrays["F_projected"], dtype=float)
    panels = all_pairs()
    half_phi = float(metadata["theory_constant_half_phi"])
    max_j = int(payload["max_scoring_steps"])
    snapshots = [int(j) for j in payload["snapshot_js"]]
    pilot_budget = compute_pilot_budget(payload["pilot_schedule"], budget)
    target_full = tilted_full_sample(
        oracle.mixture,
        beta_true,
        budget,
        int(seed_entry["target_draw_seed"]),
        scale=oracle.scale,
        feature_fn=oracle.feature_fn,
    )
    pilot_counts = balanced_pilot_counts(panels, pilot_budget)
    remaining = budget - int(pilot_counts.sum())
    pilot_observations = _build_observations(target_full[: int(pilot_counts.sum())], panels, pilot_counts)
    pilot_mu, _rho = pilot_ht_moment(
        pilot_observations, pilot_counts, panels, oracle.scale, reference
    )
    beta_hat = solve_pilot_beta(
        pilot_mu,
        reference,
        oracle.scale,
        theta_bound=float(payload["theta_bound"]),
    )
    if payload["allocation"] == "plugin":
        infos = panel_information_cross(
            oracle.mixture,
            beta_hat,
            panels,
            reference,
            oracle.scale,
            int(payload["h_tilted"]),
            int(payload["h_cond"]),
            int(seed_entry["information_seed_root"]),
            conditional_method="rejection",
        )
        _, fisher_hat = tilted_moments(beta_hat, reference, oracle.scale)
        probabilities, fw_gap, fw_iterations = frank_wolfe(
            fisher_hat,
            infos,
            np.ones(len(panels)),
            uniform_probabilities(len(panels)),
            tolerance=1e-4,
            max_iter=300,
        )
    else:
        probabilities = p_star
        fw_gap = float(metadata.get("fw_gap", np.nan))
        fw_iterations = int(metadata.get("fw_iterations", -1))
    counts = largest_remainder_counts(probabilities, remaining)
    main_observations = _build_observations(
        target_full[int(pilot_counts.sum()) :], panels, counts
    )
    observations = pilot_observations + main_observations
    metrics = design_metrics(fisher, information, p_star, probabilities, pilot_counts, counts)
    metrics["fw_gap"] = float(fw_gap)
    metrics["fw_iterations"] = int(fw_iterations)
    grouped = {tuple(panel) for panel, _obs in observations}
    update_diagnostics = [{
        "step": "pilot",
        "pilot_budget": int(pilot_counts.sum()),
        "beta_error_norm": float(np.linalg.norm(beta_hat - beta_true)),
    }]
    rows = []
    started = perf_counter()
    for scoring_steps in snapshots:
        if scoring_steps > 0:
            previous = max((prior for prior in snapshots if prior < scoring_steps), default=0)
            n_steps = scoring_steps - previous
            for local in range(n_steps):
                step_index = previous + local
                estimated_h = None
                if payload["h_mode"] == "estimated":
                    estimated_h = active_panel_information_cross(
                        oracle.mixture,
                        beta_hat,
                        tuple(grouped),
                        reference,
                        oracle.scale,
                        int(payload["h_tilted"]),
                        int(payload["h_cond"]),
                        int(seed_entry["information_seed_root"]) + 1000 * step_index,
                        conditional_method="rejection",
                    )
                beta_hat, step_diagnostics = gold_oracle_start_step(
                    oracle,
                    beta_hat,
                    observations,
                    panels,
                    information,
                    int(seed_entry["score_seed_root"]) + step_index,
                    theta_bound=float(payload["theta_bound"]),
                    step_size=float(payload["scoring_step_size"]),
                    max_step_norm=payload.get("scoring_max_step_norm"),
                    h_mode=payload["h_mode"],
                    estimated_h=estimated_h,
                    score_mode=payload["score_mode"],
                    qmc_order=int(payload.get("qmc_order", 10)),
                    lu=_ladder_lu(payload, budget),
                )
                update_diagnostics.append({
                    "step": step_index,
                    "beta_error_norm": float(np.linalg.norm(beta_hat - beta_true)),
                    **step_diagnostics,
                })
        kl = oracle.kl(beta_true, beta_hat)
        rows.append(
            _row_payload(
                payload=payload,
                budget=budget,
                scoring_steps=scoring_steps,
                beta_hat=beta_hat,
                beta_true=beta_true,
                kl=kl,
                half_phi=half_phi,
                metrics=metrics,
                counts=counts,
                pilot_counts=pilot_counts,
                update_diagnostics=list(update_diagnostics),
                seed_entry=seed_entry,
                probabilities=probabilities,
                elapsed=perf_counter() - started,
                fisher=fisher,
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--max-replications", type=int, default=None)
    parser.add_argument("--rep-range", type=int, nargs=2, default=None)
    args = parser.parse_args()
    root = REPOSITORY_ROOT
    config_path = _resolve(root, args.config)
    out = _resolve(root, args.out)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))["p4_phase3_ladder"]
    validate_ladder_config(payload)
    metadata, arrays, verified_paths = load_verified_base_artifact(payload, root)
    half_phi = float(metadata["theory_constant_half_phi"])
    manifest_path = _resolve(root, payload["target_manifest"])
    manifest, manifest_sha256 = load_seed_manifest(manifest_path)
    budgets = [int(args.budget)] if args.budget is not None else [int(b) for b in payload["budgets"]]
    replications = int(payload["replications"])
    if args.max_replications is not None:
        replications = min(replications, int(args.max_replications))
    start, end = (0, replications) if args.rep_range is None else args.rep_range
    end = min(end, replications)
    snapshots = [int(j) for j in payload["snapshot_js"]]
    requested = {(int(budget), int(rep)) for budget in budgets for rep in range(start, end)}
    missing = sorted(requested - set(manifest))
    if missing:
        raise ValueError(f"ladder seed manifest missing entries: {missing[:5]}")
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
    reference = sample_full(mixture, int(payload["pilot_reference_size"]), int(metadata["mixture_seed"]))
    if out.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {out} without --resume")
    out.mkdir(parents=True, exist_ok=True)
    jsonl_path = out / "rows.jsonl"
    done = {}
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                key = (int(row["budget"]), int(row["replication"]))
                done.setdefault(key, set()).add(int(row["scoring_steps"]))
    print(
        json.dumps({
            "event": "ladder_start",
            "ladder": payload["ladder"],
            "out": str(out),
            "budgets": budgets,
            "replications": [start, end],
            "snapshot_js": snapshots,
        }, sort_keys=True),
        flush=True,
    )
    with jsonl_path.open("a", encoding="utf-8") as stream:
        for budget in budgets:
            for replication in range(start, end):
                key = (int(budget), int(replication))
                existing = done.get(key, set())
                if set(snapshots) <= existing:
                    print(json.dumps({"event": "skip", "budget": budget, "replication": replication}), flush=True)
                    continue
                seed_entry = manifest[key]
                rows = run_ladder_replication(
                    oracle, arrays, payload, metadata, seed_entry, int(budget),
                    reference,
                )
                for row in rows:
                    if int(row["scoring_steps"]) in existing:
                        continue
                    row["replication"] = int(replication)
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                    stream.flush()
                    done.setdefault(key, set()).add(int(row["scoring_steps"]))
                    print(json.dumps({
                        "event": "row",
                        "ladder": payload["ladder"],
                        "budget": budget,
                        "replication": replication,
                        "J": row["scoring_steps"],
                        "risk_ratio_raw": row["risk_ratio_raw"],
                        "design_ratio_main": row["design_ratio_main"],
                        "beta_error_norm": row["beta_error_norm"],
                        "elapsed_seconds": row["elapsed_seconds"],
                    }, sort_keys=True), flush=True)
    rows = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gate = ladder_gate(rows, payload, half_phi)
    diagnostics = {
        "schema_version": "p4-phase3-ladder-diagnostics-v1",
        "phase": 3,
        "ladder": payload["ladder"],
        "not_formal": True,
        "budgets": budgets,
        "replications_completed": sorted({int(row["replication"]) for row in rows}),
        "scoring_steps": sorted({int(row["scoring_steps"]) for row in rows}),
        "gate": gate,
        "ladder_passed": bool(gate["passed"]),
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
    print(json.dumps({"event": "ladder_done", **gate}, sort_keys=True), flush=True)
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
