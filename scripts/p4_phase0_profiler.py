"""Phase-0 profiler: one fixed seed, oracle allocation only, not a formal gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
from pathlib import Path

import yaml

from rr_gid_cn.s1_gate import prepare_s1_oracle, run_replication
from rr_gid_cn.synthetic_oracle import all_pairs, make_frozen_mixture, reference_scale


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_head(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return os.environ.get("RR_GID_CN_BASELINE_COMMIT", "unknown")


def _budget_lu(spec, budget: int) -> int:
    if spec is None:
        return int(budget)
    if isinstance(spec, (int, float)):
        return max(1, int(spec))
    raise ValueError("phase0 lu must be a positive integer")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/p4_phase0_local.yaml"))
    parser.add_argument("--prepared", type=Path, default=Path("experiments/p4_phase0_prepared.pkl"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/p4/phase0"))
    parser.add_argument("--budgets", type=int, nargs="*")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))["phase0"]
    if not bool(cfg.get("not_formal", False)):
        raise ValueError("phase0 profiler refuses a config that is not marked not_formal")
    if str(cfg.get("experiment_mode")) != "phase0_profile":
        raise ValueError("phase0 profiler requires experiment_mode=phase0_profile")
    budgets = tuple(args.budgets or cfg["budgets"])
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = args.prepared
    freeze_path = out_dir / "phase0_freeze.json"
    config_hash = _sha256_file(args.config)
    code_files = [
        Path("src/rr_gid_cn/s1_gate.py"),
        Path("src/rr_gid_cn/synthetic_oracle.py"),
        Path("src/rr_gid_cn/policies.py"),
        Path("scripts/p4_phase0_profiler.py"),
        args.config,
    ]
    file_hashes = {str(path).replace("\\", "/"): _sha256_file(path) for path in code_files}
    baseline_commit = _git_head(root)
    mixture = make_frozen_mixture(seed=int(cfg["mixture_seed"]), alpha=float(cfg["alpha"]))
    scale = reference_scale(mixture, int(cfg["scale_pool"]), int(cfg["scale_seed"]))
    panels = all_pairs()
    if prepared_path.exists():
        with prepared_path.open("rb") as handle:
            prepared = pickle.load(handle)
    else:
        prepared_path.parent.mkdir(parents=True, exist_ok=True)
        prepared = prepare_s1_oracle(
            mixture,
            scale,
            panels,
            seed=int(cfg["mixture_seed"]),
            reference_size=int(cfg["reference_size"]),
            information_samples=int(cfg["information_tilted_samples"]),
            conditional_samples=int(cfg["information_conditional_samples"]),
            large_reference_size=int(cfg["large_reference_size"]),
        )
        with prepared_path.open("wb") as handle:
            pickle.dump(prepared, handle, protocol=5)
    artifact_sha256 = _sha256_file(prepared_path)
    freeze = {
        "experiment_mode": "phase0_profile",
        "not_formal": True,
        "baseline_commit": baseline_commit,
        "config_path": str(args.config).replace("\\", "/"),
        "config_sha256": config_hash,
        "artifact_path": str(prepared_path).replace("\\", "/"),
        "artifact_sha256": artifact_sha256,
        "file_sha256": file_hashes,
        "budgets": list(budgets),
        "policies": list(cfg["policies"]),
        "scoring_steps": int(cfg["scoring_steps"]),
        "conditional_method": str(cfg["conditional_method"]),
        "lu": int(cfg["lu"]),
        "host_note": "phase0 profiler; output is diagnostic and not a formal acceptance batch",
    }
    if freeze_path.exists():
        previous = json.loads(freeze_path.read_text(encoding="utf-8"))
        if previous.get("artifact_sha256") != artifact_sha256 or previous.get("config_sha256") != config_hash:
            raise FileExistsError(f"refusing to overwrite freeze at {freeze_path}")
    else:
        freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for budget in budgets:
        out_path = out_dir / f"phase0_profile_B{budget}_rep{int(cfg['replication'])}.json"
        if out_path.exists():
            raise FileExistsError(f"refusing to overwrite {out_path}")
        seed = int(cfg["replication_seed_base"]) + int(budget) * 1000 + int(cfg["replication"])
        rows = run_replication(
            mixture,
            scale,
            panels,
            int(budget),
            seed,
            prepared=prepared,
            lu=_budget_lu(cfg.get("lu"), budget),
            h_tilted=int(cfg["h_tilted"]),
            h_cond=int(cfg["h_cond"]),
            kl_samples=int(cfg["kl_samples"]),
            pilot_norm_cap=cfg.get("pilot_norm_cap"),
            scoring_steps=int(cfg["scoring_steps"]),
            theta_norm_cap=cfg.get("theta_norm_cap"),
            theta_l1_cap=cfg.get("theta_l1_cap"),
            scoring_step_size=float(cfg.get("scoring_step_size", 1.0)),
            scoring_max_step_norm=cfg.get("scoring_max_step_norm"),
            policies=list(cfg["policies"]),
            conditional_method=str(cfg["conditional_method"]),
        )
        if len(rows) != 1 or rows[0]["policy"] != "oracle RR-GID":
            raise RuntimeError("phase0 profiler must emit exactly one oracle allocation row")
        row = rows[0]
        row["replication"] = int(cfg["replication"])
        row["experiment_mode"] = "phase0_profile"
        row["not_formal"] = True
        row["config_sha256"] = config_hash
        row["artifact_sha256"] = artifact_sha256
        row["baseline_commit"] = baseline_commit
        row["freeze_sha256"] = _sha256_file(freeze_path)
        attribution = float(row["runtime"]["attributed_fraction"])
        payload = {
            "row": row,
            "attribution_gate": float(cfg["attribution_gate"]),
            "attribution_passed": attribution >= float(cfg["attribution_gate"]),
        }
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({
            "budget": budget,
            "out": str(out_path),
            "attributed_fraction": attribution,
            "time_total": row["runtime"]["time_total"],
            "B_kl_raw": row["B_kl_raw"],
            "target_draw_sha256": row["target_draw_sha256"],
        }), flush=True)
        if attribution < float(cfg["attribution_gate"]):
            raise SystemExit(f"phase0 attribution {attribution:.4f} < {cfg['attribution_gate']}")


if __name__ == "__main__":
    main()
