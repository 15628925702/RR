"""Resumable synthetic main runner (paper E1)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from rr_gid_cn.p4_integrity import load_seed_manifest, sha256_file, validate_experiment_mode
from rr_gid_cn.paper_run import PAPER_METHODS, run_paper_replication
from rr_gid_cn.synthetic_oracle import all_pairs, make_frozen_mixture, reference_scale


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _done_keys(path: Path) -> set[tuple[str, int, int]]:
    keys = set()
    if not path.exists():
        return keys
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        keys.add((str(row.get("method") or row.get("policy")), int(row["budget"]), int(row.get("replication", row.get("seed", 0)))))
    return keys


def write_manifest(path: Path, budgets, replications: int) -> None:
    rows = []
    for budget in budgets:
        for rep in range(int(replications)):
            seed = 202609000 + int(budget) * 1000 + int(rep)
            rows.append({
                "budget": int(budget),
                "replication": int(rep),
                "replication_seed": seed,
                "target_draw_seed": seed + 3,
                "pilot_or_design_seed": seed + 11,
                "score_seed_root": seed + 4,
                "information_seed_root": seed + 7000,
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/paper/synthetic_main.yaml"))
    ap.add_argument("--prepared", type=Path, default=Path("experiments/paper/oracle_artifact.pkl"))
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--rep-range", type=int, nargs=2, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--methods", nargs="*", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_experiment_mode(cfg, paper=True, formal=True)
    budgets = [int(args.budget)] if args.budget is not None else list(cfg["budgets"])
    replications = int(cfg["replications"])
    start, end = (0, replications) if args.rep_range is None else (int(args.rep_range[0]), int(args.rep_range[1]))
    methods = tuple(args.methods) if args.methods else tuple(cfg["methods"])
    manifest_path = Path(cfg["seed_manifest"])
    if not manifest_path.exists():
        write_manifest(manifest_path, cfg["budgets"], replications)
    manifest, _ = load_seed_manifest(manifest_path)
    if not args.prepared.exists():
        raise SystemExit(f"missing oracle artifact: {args.prepared}. Run scripts/build_oracle_artifact.py first.")
    prepared = pickle.loads(args.prepared.read_bytes())
    mixture = make_frozen_mixture(seed=2026, alpha=1.0)
    scale = reference_scale(mixture, n=6000, seed=2026)
    panels = all_pairs()
    out_dir = Path(cfg["output_path"])
    out = out_dir / "rows.jsonl"
    done = _done_keys(out) if args.resume else set()
    commit = _git_commit()
    config_sha = hashlib.sha256(args.config.read_bytes()).hexdigest()
    artifact_sha = sha256_file(args.prepared)
    for budget in budgets:
        for rep in range(start, end):
            pending = [m for m in methods if (m, int(budget), int(rep)) not in done]
            if not pending:
                continue
            entry = dict(manifest[(int(budget), int(rep))])
            entry["replication"] = int(rep)
            rows = run_paper_replication(
                mixture, scale, panels, int(budget), int(entry["replication_seed"]), prepared,
                scoring_steps=int(cfg["scoring_steps"]),
                scoring_max_step_norm=float(cfg["scoring_max_step_norm"]),
                score_qmc_order=int(cfg["score_qmc_order"]),
                information_qmc_order=int(cfg["information_qmc_order"]),
                information_outer_rows=int(cfg["information_outer_rows"]),
                information_scrambles=int(cfg.get("information_scrambles", 2)),
                hot_dtype=str(cfg.get("dtype", "float32")),
                policies=tuple(pending),
                pilot_schedule=cfg["pilot_schedule"],
                seed_manifest_entry=entry,
                reproducibility={
                    "code_commit": commit,
                    "config_sha256": config_sha,
                    "artifact_sha256": artifact_sha,
                    "environment": "synthetic",
                },
            )
            _append_jsonl(out, rows)
            if args.profile:
                for row in rows:
                    rt = row["runtime"]
                    print(
                        f"{row['method']} B={budget} rep={rep} "
                        f"risk={row['risk_ratio']:.3f} "
                        f"total={rt.get('time_estimation', 0)+rt.get('time_score_basis', 0)+rt.get('time_info_basis', 0):.1f}s "
                        f"info={rt['time_info_basis']:.1f}s score={rt['time_score_basis']:.1f}s est={rt['time_estimation']:.1f}s",
                        flush=True,
                    )
            done.update((row["method"], int(budget), int(rep)) for row in rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
