"""Resumable P4 formal runner. Each completed replication is appended atomically."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import argparse
import hashlib
import math
import subprocess

import yaml

from rr_gid_cn.s1_gate import prepare_s1_oracle, run_replication
from rr_gid_cn.p4_integrity import (
    compute_pilot_budget,
    load_seed_manifest,
    sha256_file,
    validate_artifact_metadata,
    validate_experiment_mode,
)
from rr_gid_cn.synthetic_oracle import all_pairs, make_frozen_mixture, reference_scale, sample_full


def _budget_mc_size(base: int, spec, budget: int) -> int:
    """Return a budget-growing MC size for Algorithm 2 information updates."""
    if spec is None:
        return int(base)
    if isinstance(spec, (int, float)):
        return max(int(base), int(math.ceil(float(spec))))
    scale = float(spec.get("scale", 0.0))
    exponent = float(spec.get("exponent", 0.5))
    return max(int(base), int(math.ceil(scale * float(budget) ** exponent)))


def _budget_qmc_order(base: int, spec, budget: int) -> int:
    """Grow conditional-QMC resolution with LU/B for the exact-score route."""
    if spec is None:
        return int(base)
    if isinstance(spec, (int, float)):
        return max(int(base), int(spec))
    anchor = float(spec.get("anchor_budget", 8000.0))
    slope = float(spec.get("log2_slope", 1.0))
    return max(int(base), int(math.ceil(float(base) + slope * math.log2(max(float(budget), 1.0) / anchor))))


def _budget_lu(spec, budget: int) -> int:
    """Finite completion schedule with L_U(B) increasing to infinity.

    The PDF specifies the asymptotic requirement L_U,B -> infinity, but not
    a finite-run formula.  A logarithmic schedule avoids the quadratic cost of
    setting L_U=B while remaining explicit and auditable in every result.
    """
    if spec is None:
        return int(budget)
    if isinstance(spec, (int, float)):
        return max(1, int(math.ceil(float(spec))))
    kind = str(spec.get("kind", "log2"))
    if kind != "log2":
        raise ValueError(f"unsupported lu_schedule kind: {kind}")
    scale = float(spec.get("scale", 32.0))
    offset = float(spec.get("offset", 1.0))
    return max(1, int(math.ceil(scale * math.log2(float(budget) + offset))))


def _source_hashes() -> dict[str, str]:
    paths = (
        Path("src/rr_gid_cn/s1_gate.py"),
        Path("src/rr_gid_cn/p4_integrity.py"),
        Path("scripts/p4_formal_run.py"),
        Path("scripts/p4_validate.py"),
    )
    return {path.as_posix(): sha256_file(path) for path in paths}


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--config", type=Path, default=Path("configs/p4_formal.yaml"))
    ap.add_argument("--max-replications", type=int, default=None, help="limit replications per budget (for smoke runs)")
    ap.add_argument("--scoring-steps", type=int, default=None, help="J Fisher-scoring steps (default from config, 2; J ablation passes 0/1/2 explicitly)")
    ap.add_argument("--rep-range", type=int, nargs=2, default=None, help="[start, end) replication range for sharding a budget across processes")
    ap.add_argument("--prepared", type=Path, default=Path("experiments/p4_prepared_oracle_hp.pkl"))
    ap.add_argument("--out-prefix", default="p4_exact_fix1")
    ap.add_argument("--reference-size", type=int, default=50000)
    ap.add_argument("--large-reference-size", type=int, default=None)
    ap.add_argument("--info-tilted", type=int, default=None)
    ap.add_argument("--info-cond", type=int, default=None)
    ap.add_argument("--scoring-step-size", type=float, default=None)
    ap.add_argument("--lu", type=int, default=None, help="override update-stage conditional completions")
    ap.add_argument("--h-tilted", type=int, default=None, help="override update-stage tilted samples")
    ap.add_argument("--h-cond", type=int, default=None, help="override cross-completion samples")
    ap.add_argument("--conditional-method", choices=("rejection", "qmc", "exact_adaptive"), default=None)
    ap.add_argument("--qmc-order", type=int, default=None)
    ap.add_argument("--kl-samples", type=int, default=None, help="override KL diagnostic samples")
    ap.add_argument("--diagnostic", action="store_true", help="allow fixed MC overrides; output is not formal")
    args = ap.parse_args()
    with args.config.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    p4 = cfg["p4"]
    validate_experiment_mode(p4, formal=not args.diagnostic)
    if not args.diagnostic and args.lu is not None:
        raise ValueError("formal P4 cannot use fixed --lu; use budget-derived L_U(B)")
    budgets = (args.budget,) if args.budget is not None else tuple(p4["budgets"])
    replications = args.max_replications or int(p4["replications"])
    manifest_path = Path(p4["target_manifest"])
    manifest, manifest_sha256 = load_seed_manifest(manifest_path)
    requested_keys = {
        (int(budget), int(replication))
        for budget in budgets for replication in range(replications)
    }
    missing_manifest = sorted(requested_keys - set(manifest))
    if missing_manifest:
        raise ValueError(f"seed manifest is missing requested entries: {missing_manifest[:5]}")
    config_sha256 = sha256_file(args.config)
    source_sha256 = _source_hashes()
    code_commit = _git_commit()
    steps = int(p4["scoring_steps"]) if args.scoring_steps is None else args.scoring_steps
    # Explicit --scoring-steps always writes a J-suffixed file so the J ablation
    # never collides with the default (J=2) main-configuration output.
    suffix = "" if args.scoring_steps is None else f"_J{steps}"
    out = Path("results")
    done: dict[int, set] = {b: set() for b in budgets}
    for b in budgets:
        fp = out / f"{args.out_prefix}_{b}{suffix}.jsonl"
        if fp.exists():
            for line in fp.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    done[b].add((row["replication"], row["policy"]))
    mixture = make_frozen_mixture(seed=2026, alpha=1.0)
    scale = reference_scale(mixture, 6000, 2026)
    panels = all_pairs()
    prepared_path = args.prepared
    if prepared_path.exists():
        with prepared_path.open("rb") as stream:
            prepared = pickle.load(stream)
        # Formal P4 must use the PDF reference train/large pools.  Diagnostic
        # artifacts intentionally use tiny pools and must never enter JSONL.
        ref_n = int(len(prepared.get("reference", [])))
        large_n = int(len(prepared.get("reference_large", [])))
        if ref_n < 50000 or large_n < int(p4["large_reference_size"]):
            raise ValueError(
                f"prepared artifact is diagnostic-sized (reference={ref_n}, "
                f"reference_large={large_n}); formal P4 requires 50000/{p4['large_reference_size']}"
            )
        expected_metadata = {
            "schema_version": "p4-oracle-artifact-v2",
            "mixture_seed": int(p4.get("mixture_seed", 2026)),
            "alpha": float(p4.get("alpha", 1.0)),
            "reference_size": int(p4.get("reference_size", args.reference_size)),
            "large_reference_size": int(p4["large_reference_size"]),
            "information_tilted_samples": int(p4["information_tilted_samples"]),
            "information_conditional_samples": int(p4["information_conditional_samples"]),
            "information_method": str(p4.get("information_method", "cross_completion_rejection")),
            "build_config_sha256": config_sha256,
            "code_commit": code_commit,
            "source_sha256": source_sha256,
        }
        validate_artifact_metadata(prepared.get("artifact_metadata", {}), expected_metadata)
    else:
        prepared = prepare_s1_oracle(
            mixture, scale, panels, seed=2026,
            reference_size=int(p4.get("reference_size", args.reference_size)),
            information_samples=args.info_tilted or p4["information_tilted_samples"],
            conditional_samples=args.info_cond or p4["information_conditional_samples"],
            large_reference_size=args.large_reference_size or p4["large_reference_size"],
        )
        prepared["artifact_metadata"].update({
            "build_config_sha256": config_sha256,
            "code_commit": code_commit,
            "source_sha256": source_sha256,
        })
        with prepared_path.open("wb") as stream:
            pickle.dump(prepared, stream, protocol=5)
    artifact_sha256 = sha256_file(prepared_path)
    # For the formal S1 route L_U grows with B, as required by the PDF.  A
    # fixed LU is permitted only for explicitly diagnostic runs.
    kwargs = dict(lu=(args.lu if args.lu is not None else None),
                  h_tilted=p4["h_tilted"] if args.h_tilted is None else args.h_tilted,
                  h_cond=p4["h_cond"] if args.h_cond is None else args.h_cond,
                  kl_samples=p4["kl_samples"] if args.kl_samples is None else args.kl_samples,
                  pilot_norm_cap=p4["pilot_norm_cap"],
                  scoring_steps=steps,
                  frozen_beta_star_information=bool(p4.get("frozen_beta_star_information", False)),
                  kl_mu_direct=bool(p4.get("kl_mu_direct", False)),
                  theta_norm_cap=p4.get("theta_norm_cap"),
                  theta_l1_cap=p4.get("theta_l1_cap"),
                  scoring_step_size=p4.get("scoring_step_size", 1.0) if args.scoring_step_size is None else args.scoring_step_size,
                  scoring_max_step_norm=p4.get("scoring_max_step_norm"),
                  conditional_method=p4.get("conditional_method", "rejection") if args.conditional_method is None else args.conditional_method,
                  qmc_order=int(p4.get("qmc_order", 10) if args.qmc_order is None else args.qmc_order),
                  qmc_start_order=int(p4.get("qmc_start_order", 8)),
                  qmc_max_order=int(p4.get("qmc_max_order", 16)),
                  qmc_atol=float(p4.get("qmc_atol", 2e-6)),
                  qmc_rtol=float(p4.get("qmc_rtol", 2e-5)))
    for budget in budgets:
        fp = out / f"{args.out_prefix}_{budget}{suffix}.jsonl"
        with fp.open("a", encoding="utf-8") as stream:
            start, end = args.rep_range if args.rep_range else (0, replications)
            for replication in range(start, min(end, replications)):
                if all((replication, pol) in done[budget] for pol in ("Uniform SQD", "A-OSQD", "oracle RR-GID")):
                    continue
                seed_entry = manifest[(int(budget), int(replication))]
                seed = int(seed_entry["replication_seed"])
                run_kwargs = dict(kwargs)
                if run_kwargs["lu"] is None:
                    run_kwargs["lu"] = _budget_lu(p4.get("lu_schedule"), budget)
                if args.h_tilted is None:
                    run_kwargs["h_tilted"] = _budget_mc_size(
                        p4["h_tilted"], p4.get("h_tilted_growth"), budget
                    )
                if args.h_cond is None:
                    run_kwargs["h_cond"] = _budget_mc_size(
                        p4["h_cond"], p4.get("h_cond_growth"), budget
                    )
                if args.qmc_order is None:
                    run_kwargs["qmc_order"] = _budget_qmc_order(
                        p4.get("qmc_order", 10), p4.get("qmc_order_growth"), budget
                    )
                pilot_schedule = dict(p4["pilot_schedule"])
                pilot_budget = compute_pilot_budget(pilot_schedule, budget)
                reproducibility = {
                    "schema_version": "p4-phase1-row-v2",
                    "not_formal": bool(args.diagnostic),
                    "experiment_mode": str(p4["experiment_mode"]),
                    "code_commit": code_commit,
                    "source_sha256": source_sha256,
                    "config_sha256": config_sha256,
                    "artifact_id": prepared_path.name,
                    "artifact_sha256": artifact_sha256,
                    "manifest_id": manifest_path.name,
                    "manifest_sha256": manifest_sha256,
                }
                rows = run_replication(
                    mixture, scale, panels, budget, seed, prepared=prepared,
                    pilot_schedule=pilot_schedule, pilot_budget=pilot_budget,
                    seed_manifest_entry=seed_entry, reproducibility=reproducibility,
                    **run_kwargs,
                )
                for row in rows:
                    row["replication"] = replication
                    row["lu"] = int(run_kwargs["lu"])
                    row["lu_schedule"] = p4.get("lu_schedule")
                    row["qmc_order"] = int(run_kwargs["qmc_order"])
                    row["conditional_method"] = str(run_kwargs["conditional_method"])
                    row["h_tilted"] = int(run_kwargs["h_tilted"])
                    row["h_cond"] = int(run_kwargs["h_cond"])
                    row["scoring_steps"] = int(run_kwargs["scoring_steps"])
                    if (replication, row["policy"]) not in done[budget]:
                        stream.write(json.dumps(row, sort_keys=True) + "\n")
                        done[budget].add((replication, row["policy"]))
                stream.flush()
                print(json.dumps({"budget": budget, "replication": replication, "rows": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
