"""Paired budget-growing-LU diagnostic for P4 S1.

This is deliberately separate from formal JSONL.  It evaluates the same
target-draw seed at several LU factors and reports paired differences and
Monte-Carlo summaries; it never changes or overwrites formal results.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np

from rr_gid_cn.s1_gate import run_replication
from rr_gid_cn.synthetic_oracle import all_pairs, make_frozen_mixture, reference_scale


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", type=Path, default=Path("experiments/p4_prepared_oracle.pkl"))
    ap.add_argument("--out", type=Path, default=Path("results/p4_lu_convergence_paired.json"))
    ap.add_argument("--budgets", type=int, nargs="+", default=[200, 400, 800])
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--factors", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--h-tilted", type=int, default=512)
    ap.add_argument("--h-cond", type=int, default=16)
    ap.add_argument("--kl-samples", type=int, default=2048)
    args = ap.parse_args()
    if not args.prepared.exists():
        raise FileNotFoundError(args.prepared)
    with args.prepared.open("rb") as f:
        prepared = pickle.load(f)
    ref_n, large_n = len(prepared.get("reference", [])), len(prepared.get("reference_large", []))
    if ref_n < 50000 or large_n < 200000:
        raise ValueError(f"prepared artifact too small: reference={ref_n}, reference_large={large_n}")

    mix = make_frozen_mixture(seed=2026, alpha=1.0)
    scale = reference_scale(mix, 6000, 2026)
    panels = all_pairs()
    rows = []
    for budget in args.budgets:
        for rep in range(args.reps):
            seed = 202600000 + budget * 1000 + rep
            for factor in args.factors:
                lu = max(1, int(math.ceil(factor * budget)))
                result = run_replication(
                    mix, scale, panels, budget, seed, prepared=prepared,
                    lu=lu, h_tilted=args.h_tilted, h_cond=args.h_cond,
                    kl_samples=args.kl_samples, pilot_norm_cap=None,
                    scoring_steps=2, theta_norm_cap=4.0, theta_l1_cap=5.0,
                    kl_mu_direct=False, use_oracle_H=False,
                    policies=["oracle RR-GID"],
                )[0]
                rows.append({
                    "budget": budget, "replication": rep, "seed": seed,
                    "factor": factor, "lu": lu,
                    "B_kl": result["B_kl"], "kl": result["kl"],
                    "B_kl_raw": result.get("B_kl_raw"), "kl_raw": result.get("kl_raw"),
                    "beta_hat_norm": result["beta_hat_norm"],
                    "design_ratio": result["design_ratio"],
                    "update_diagnostics": result["update_diagnostics"],
                })
                print(json.dumps(rows[-1], sort_keys=True), flush=True)

    target = 63.73693541665105 / 2.0
    summaries = []
    for b in args.budgets:
        sub = [r for r in rows if r["budget"] == b]
        for f in args.factors:
            vals = np.asarray([r["B_kl"] for r in sub if r["factor"] == f], dtype=float)
            summaries.append({
                "budget": b, "factor": f, "n": int(vals.size),
                "mean_B_kl": float(vals.mean()),
                "sd_B_kl": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                "se_B_kl": float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0,
                "ci95_low": float(vals.mean() - 1.96 * vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else float(vals.mean()),
                "ci95_high": float(vals.mean() + 1.96 * vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else float(vals.mean()),
                "target_half_phi": target,
            })
    paired = []
    for b in args.budgets:
        by_rep = {r: {x["factor"]: x["B_kl"] for x in rows if x["budget"] == b and x["replication"] == r} for r in range(args.reps)}
        base = min(args.factors)
        for f in args.factors:
            if f == base:
                continue
            d = np.asarray([v[f] - v[base] for v in by_rep.values() if f in v and base in v], dtype=float)
            paired.append({"budget": b, "factor": f, "vs_factor": base, "mean_delta": float(d.mean()), "sd_delta": float(d.std(ddof=1)) if d.size > 1 else 0.0, "n": int(d.size)})
    params = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    payload = {"stage": "P4", "kind": "paired budget-growing-LU diagnostic", "parameters": params, "rows": rows, "summaries": summaries, "paired_deltas": paired}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
