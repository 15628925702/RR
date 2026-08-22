"""Parallel P5 formal launcher.

Splits each budget's 200 replications into ``--shards`` contiguous ranges and
runs one worker per (budget, shard) as a background subprocess. All workers
share the same seed scheme so rows pair across the final four policies.

Compute budget: each worker caps OpenBLAS at ``--threads`` cores and requests
at most ``RR_GID_GPU_FRACTION`` of device memory (default 0.5, half the GPU),
leaving the other half for a co-located project. Use ``--dry-run`` to print the
worker commands without launching.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BUDGETS = (2000, 4000, 8000, 16000, 32000)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=2, help="replication shards per budget")
    ap.add_argument("--threads", type=int, default=8, help="OPENBLAS_NUM_THREADS per worker")
    ap.add_argument("--gpu-fraction", type=float, default=0.5)
    ap.add_argument("--replications", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    log_dir = Path("/root/p5_logs")
    commands = []
    for budget in BUDGETS:
        reps = args.replications
        per = (reps + args.shards - 1) // args.shards
        for shard in range(args.shards):
            start = shard * per
            end = min(start + per, reps)
            if start >= reps:
                continue
            cmd = [
                str(root / "scripts" / "p5_formal_run.py"),
                "--budget", str(budget),
                "--rep-range", str(start), str(end),
                "--mlp-steps", "200",
            ]
            env = {
                "PYTHONPATH": str(root / "src"),
                "OPENBLAS_NUM_THREADS": str(args.threads),
                "OMP_NUM_THREADS": str(args.threads),
                "RR_GID_GPU_FRACTION": str(args.gpu_fraction),
            }
            commands.append((budget, shard, cmd, env))

    if args.dry_run:
        for budget, shard, cmd, env in commands:
            print(f"B={budget} shard={shard}: {' '.join(cmd)}")
        print(f"total workers: {len(commands)}")
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for budget, shard, cmd, env in commands:
        log = log_dir / f"B{budget}_s{shard}.log"
        with log.open("wb") as out:
            p = subprocess.Popen(
                [sys.executable] + cmd,
                stdout=out, stderr=subprocess.STDOUT,
                env={**__import__("os").environ, **env},
                cwd=str(root),
            )
            procs.append((budget, shard, p, log))
        print(f"launched B={budget} shard={shard} (pid {procs[-1][2].pid}) -> {log}", flush=True)
    print(f"total workers: {len(procs)}")


if __name__ == "__main__":
    main()
