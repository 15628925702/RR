"""P11 Fig.3: Gas R1 + R2 budget curves (PDF Fig.3).

Left: R1 semi-synthetic B*KL vs budget (four policies, 200 reps).  Right: R2
natural-drift family-internal loss vs budget, averaged over the three campaigns
(50 reps each).  Four policies share the same final RR estimator so the gap
isolates allocation quality.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BUDGETS = (400, 800, 1600, 3200)
POLICIES = ("Uniform SQD", "A-OSQD", "Discriminative Score OED", "RR-GID")
COLORS = {"Uniform SQD": "#888888", "A-OSQD": "#d62728",
          "Discriminative Score OED": "#1f77b4", "RR-GID": "#2ca02c"}
MARKERS = {"Uniform SQD": "o", "A-OSQD": "s",
           "Discriminative Score OED": "^", "RR-GID": "D"}


def _r1_bkl(budget: int) -> dict[str, list[float]]:
    fp = Path("results") / f"p9_r1_{budget}.jsonl"
    agg: dict[str, list[float]] = defaultdict(list)
    if fp.exists():
        for line in fp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                agg[r["policy"]].append(float(r["B_kl"]))
    return dict(agg)


def _r2_proj(campaign: str, budget: int) -> dict[str, list[float]]:
    fp = Path("results") / f"p10_r2_{campaign}_{budget}.jsonl"
    agg: dict[str, list[float]] = defaultdict(list)
    if fp.exists():
        for line in fp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                agg[r["policy"]].append(float(r["projection_loss"]))
    return dict(agg)


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    # --- Left: R1 B*KL vs budget ---
    for pol in POLICIES:
        xs, ys, errs = [], [], []
        for b in BUDGETS:
            d = _r1_bkl(b)
            if pol not in d:
                continue
            v = np.asarray(d[pol])
            xs.append(b); ys.append(v.mean()); errs.append(v.std() / np.sqrt(len(v)))
        if xs:
            ax1.errorbar(xs, ys, yerr=errs, marker=MARKERS[pol], capsize=3,
                         color=COLORS[pol], label=pol, linewidth=1.6)
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("Budget $B$")
    ax1.set_ylabel("$B\\cdot\\mathrm{KL}$ (relative to empirical base)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8)
    ax1.set_title("R1: well-specified semi-synthetic (200 reps)")

    # --- Right: R2 family-internal loss vs budget (avg over campaigns) ---
    for pol in POLICIES:
        xs, ys, errs = [], [], []
        for b in BUDGETS:
            vals = []
            for camp in ("batch7", "batches89", "batch10"):
                d = _r2_proj(camp, b)
                if pol in d:
                    vals.extend(d[pol])
            if not vals:
                continue
            v = np.asarray(vals)
            xs.append(b); ys.append(v.mean()); errs.append(v.std() / np.sqrt(len(v)))
        if xs:
            ax2.errorbar(xs, ys, yerr=errs, marker=MARKERS[pol], capsize=3,
                         color=COLORS[pol], label=pol, linewidth=1.6)
    ax2.set_xscale("log")
    ax2.set_xlabel("Budget $B$")
    ax2.set_ylabel("Family-internal loss $D_A(\\hat\\beta,\\beta^\\dagger)$")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8)
    ax2.set_title("R2: natural drift, avg over 3 campaigns (50 reps)")

    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/fig3_gas_r1_r2_budgets.png", dpi=200)
    print("saved figures/fig3_gas_r1_r2_budgets.png")


if __name__ == "__main__":
    main()
