"""Run the Phase-0, single-seed P4 profiler without claiming formal acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import yaml

from rr_gid_cn.s1_gate import prepare_s1_oracle, run_replication
from rr_gid_cn.synthetic_oracle import all_pairs, make_frozen_mixture, reference_scale


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()


def git_dirty() -> bool:
    return bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, encoding="utf-8"
        ).strip()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/p4_phase0_local.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/p4/phase0_local_20260826"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))["phase0"]
    if cfg["experiment_mode"] != "phase0_profile":
        raise ValueError("Phase-0 profiler refuses non-profile experiment modes")
    if not bool(cfg.get("not_formal")):
        raise ValueError("Phase-0 output must be explicitly marked not_formal")
    if cfg["policies"] != ["oracle RR-GID"]:
        raise ValueError("Phase-0 must profile only the current oracle-allocation policy")
    if int(cfg["scoring_steps"]) != 2:
        raise ValueError("Phase-0 guidance requires J=2")
    if list(map(int, cfg["budgets"])) != [2000, 8000, 32000]:
        raise ValueError("Phase-0 guidance freezes budgets at 2000/8000/32000")
    if int(cfg["replication"]) != 0:
        raise ValueError("Phase-0 runs exactly one fixed-seed replication")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = args.out_dir / "prepared.pkl"
    rows_path = args.out_dir / "rows.jsonl"
    summary_path = args.out_dir / "summary.json"
    log_path = args.out_dir / "run.log"

    mixture = make_frozen_mixture(
        seed=int(cfg["mixture_seed"]), alpha=float(cfg["alpha"])
    )
    scale = reference_scale(
        mixture, int(cfg["scale_pool"]), int(cfg["scale_seed"])
    )
    panels = all_pairs()
    prepare_started = perf_counter()
    if prepared_path.exists():
        with prepared_path.open("rb") as stream:
            prepared = pickle.load(stream)
    else:
        prepared = prepare_s1_oracle(
            mixture,
            scale,
            panels,
            seed=int(cfg["mixture_seed"]),
            reference_size=int(cfg["reference_size"]),
            large_reference_size=int(cfg["large_reference_size"]),
            information_samples=int(cfg["information_tilted_samples"]),
            conditional_samples=int(cfg["information_conditional_samples"]),
        )
        with prepared_path.open("wb") as stream:
            pickle.dump(prepared, stream, protocol=5)
    prepare_seconds = perf_counter() - prepare_started

    config_hash = sha256_file(args.config)
    prepared_hash = sha256_file(prepared_path)
    commit = git_commit()
    source_hashes = {
        path.as_posix(): sha256_file(path)
        for path in (
            Path("src/rr_gid_cn/s1_gate.py"),
            Path("src/rr_gid_cn/synthetic_oracle.py"),
            Path("scripts/p4_phase0_profile.py"),
        )
    }
    dirty = git_dirty()
    completed = set()
    if rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing = json.loads(line)
                if (
                    existing.get("schema_version") == "p4-phase0-profile-v1"
                    and existing.get("experiment_mode") == "phase0_profile"
                    and existing.get("not_formal") is True
                    and existing.get("config_sha256") == config_hash
                    and existing.get("prepared_sha256") == prepared_hash
                    and existing.get("source_sha256") == source_hashes
                    and existing.get("replication_seed")
                    == int(cfg["replication_seed_base"]) + int(cfg["replication"])
                ):
                    completed.add(int(existing["budget"]))
                else:
                    raise RuntimeError(
                        "existing rows do not match the frozen Phase-0 inputs; "
                        "use a new output directory"
                    )

    with rows_path.open("a", encoding="utf-8") as rows_stream, log_path.open(
        "a", encoding="utf-8"
    ) as log_stream:
        for budget in map(int, cfg["budgets"]):
            if budget in completed:
                continue
            rows = run_replication(
                mixture,
                scale,
                panels,
                budget,
                int(cfg["replication_seed_base"]) + int(cfg["replication"]),
                prepared=prepared,
                lu=int(cfg["lu"]),
                h_tilted=int(cfg["h_tilted"]),
                h_cond=int(cfg["h_cond"]),
                pilot_norm_cap=float(cfg["pilot_norm_cap"]),
                kl_samples=int(cfg["kl_samples"]),
                scoring_steps=int(cfg["scoring_steps"]),
                theta_norm_cap=cfg["theta_norm_cap"],
                theta_l1_cap=cfg["theta_l1_cap"],
                policies=cfg["policies"],
                scoring_step_size=float(cfg["scoring_step_size"]),
                scoring_max_step_norm=cfg["scoring_max_step_norm"],
                conditional_method=str(cfg["conditional_method"]),
            )
            if len(rows) != 1:
                raise RuntimeError(f"expected one oracle-allocation row, got {len(rows)}")
            row = rows[0]
            row.update(
                {
                    "schema_version": "p4-phase0-profile-v1",
                    "experiment_mode": cfg["experiment_mode"],
                    "not_formal": True,
                    "code_commit": commit,
                    "code_dirty": dirty,
                    "source_sha256": source_hashes,
                    "config_sha256": config_hash,
                    "prepared_sha256": prepared_hash,
                    "replication": int(cfg["replication"]),
                    "replication_seed": (
                        int(cfg["replication_seed_base"]) + int(cfg["replication"])
                    ),
                    "device": "cpu",
                    "dtype": "float64",
                }
            )
            try:
                import torch

                if torch.cuda.is_available():
                    row["device"] = torch.cuda.get_device_name(0)
                    row["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated(0))
            except ImportError:
                pass
            rows_stream.write(json.dumps(row, sort_keys=True) + "\n")
            rows_stream.flush()
            log_stream.write(json.dumps({"budget": budget, "runtime": row["runtime"]}) + "\n")
            log_stream.flush()

    final_rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = set(map(int, cfg["budgets"]))
    observed = {int(row["budget"]) for row in final_rows}
    if observed != expected or len(final_rows) != len(expected):
        raise RuntimeError(f"incomplete Phase-0 grid: expected={expected}, observed={observed}")
    summary = {
        "schema_version": "p4-phase0-profile-v1",
        "experiment_mode": cfg["experiment_mode"],
        "not_formal": True,
        "code_commit": commit,
        "code_dirty": dirty,
        "source_sha256": source_hashes,
        "config_sha256": config_hash,
        "prepared_sha256": prepared_hash,
        "prepared_seconds": prepare_seconds,
        "rows_sha256": sha256_file(rows_path),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "budgets": sorted(observed),
        "minimum_attributed_fraction": min(
            float(row["runtime"]["attributed_fraction"]) for row in final_rows
        ),
        "all_reproducibility_fields_present": all(
            row.get("target_draw_sha256")
            and row.get("config_sha256") == config_hash
            and row.get("prepared_sha256") == prepared_hash
            for row in final_rows
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
