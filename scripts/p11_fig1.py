"""P11 Fig.1: Synthetic S1 B*KL curves + J ablation (PDF Fig.1).

Left panel: B*KL vs budget for the final four policies (Uniform SQD, A-OSQD,
Discriminative Score OED, learned RR-GID) with Monte-Carlo SE bands.  Right
panel: J ablation at B=8000 (Fisher-scoring steps 0/1/2) for the three P4
oracle-gate policies.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BUDGETS = (2000, 4000, 8000, 16000, 32000)
POLICIES = ("Uniform SQD", "A-OSQD", "Discriminative Score OED", "learned RR-GID")
COLORS = {"Uniform SQD": "#888888", "A-OSQD": "#d62728",
          "Discriminative Score OED": "#1f77b4", "learned RR-GID": "#2ca02c",
          "oracle RR-GID": "#ff7f0e"}
MARKERS = {"Uniform SQD": "o", "A-OSQD": "s",
           "Discriminative Score OED": "^", "learned RR-GID": "D"}


def _load_bkl(budget: int) -> dict[str, list[float]]:
    """Aggregate B_kl by policy from P4 (oracle-gate) + P5 (four-policy) files."""
    agg: dict[str, list[float]] = defaultdict(list)
    for fp in (Path("results") / f"p4_formal_{budget}.jsonl",
               Path("results") / f"p5_four_{budget}.jsonl"):
        if not fp.exists():
            continue
        for line in fp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            agg[r["policy"]].append(float(r["B_kl"]))
    return dict(agg)


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    # --- Left: B*KL vs budget ---
    for pol in POLICIES:
        xs, ys, errs = [], [], []
        for b in BUDGETS:
            data = _load_bkl(b)
            if pol not in data:
                continue
            v = np.asarray(data[pol])
            xs.append(b); ys.append(v.mean()); errs.append(v.std() / np.sqrt(len(v)))
        if xs:
            ax1.errorbar(xs, ys, yerr=errs, marker=MARKERS[pol], capsize=3,
                         color=COLORS[pol], label=pol, linewidth=1.6)
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("Budget $B$")
    ax1.set_ylabel("$B\\cdot\\mathrm{KL}(Q_{\\beta^*}\\|Q_{\\hat\\beta})$")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8)
    ax1.set_title("S1: $B\\cdot\\mathrm{KL}$ vs budget (four policies)")

    # --- Right: J ablation at B=8000 (P4 oracle-gate policies) ---
    jvals = {"Uniform SQD": [], "A-OSQD": [], "oracle RR-GID": []}
    for j in (0, 1, 2):
        fp = Path("results") / (f"p4_exact_8000_J{j}.jsonl" if j < 2 else "p4_exact_8000.jsonl")
        agg = defaultdict(list)
        if fp.exists():
            for line in fp.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    agg[r["policy"]].append(float(r["B_kl"]))
        for pol in jvals:
            if pol in agg:
                jvals[pol].append(float(np.mean(agg[pol])))
    for pol in jvals:
        if len(jvals[pol]) == 3:
            ax2.plot([0, 1, 2], jvals[pol], marker="o", color=COLORS.get(pol, "#333"),
                     label=pol, linewidth=1.6)
    ax2.set_yscale("log")
    ax2.set_xlabel("Fisher-scoring steps $J$")
    ax2.set_ylabel("$B\\cdot\\mathrm{KL}$ (B=8000)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    ax2.set_title("J ablation (B=8000, oracle-gate)")

    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/fig1_s1_bkl_jablation.png", dpi=200)
    print("saved figures/fig1_s1_bkl_jablation.png")


if __name__ == "__main__":
    main()
