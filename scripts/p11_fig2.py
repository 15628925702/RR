"""P11 Fig.2: Synthetic S2 nonlinearity sweep + generator reuse frontier (PDF Fig.2).

Left: design ratio vs alpha for the four policies (oracle = 1).  Right:
amortized frontier -- average design regret vs cumulative train+inference
compute, showing RR-GID reuses a frozen generator at negligible marginal cost.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"oracle RR-GID": "#ff7f0e", "Uniform SQD": "#888888", "A-OSQD": "#d62728",
          "learned RR-GID": "#2ca02c", "Discriminative Score OED": "#1f77b4"}
MARKERS = {"oracle RR-GID": "*", "Uniform SQD": "o", "A-OSQD": "s",
           "learned RR-GID": "D", "Discriminative Score OED": "^"}


def main() -> None:
    s = json.load(open("results/p7_s2_summary.json", encoding="utf-8"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    # --- Left: design ratio vs alpha ---
    alphas = [r["alpha"] for r in s["alpha_sweep"]]
    for key, label in (("design_ratio_Uniform SQD", "Uniform SQD"),
                       ("design_ratio_A-OSQD", "A-OSQD"),
                       ("design_ratio_Discriminative Score OED", "Discriminative Score OED"),
                       ("design_ratio_learned RR-GID", "learned RR-GID")):
        ys = [r[key] for r in s["alpha_sweep"]]
        ax1.plot(alphas, ys, marker=MARKERS[label], color=COLORS[label], label=label, linewidth=1.6)
    ax1.axhline(1.0, ls="--", color="#999", lw=1)
    ax1.set_xlabel("Nonlinearity $\\alpha$")
    ax1.set_ylabel("Design ratio $\\Phi(\\hat p)/\\Phi(p^*)$")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    ax1.set_title("S2: design ratio vs $\\alpha$ (B=8000)")

    # --- Right: reuse frontier (regret vs cumulative compute) ---
    reuse = sorted(s["reuse"], key=lambda r: r["campaigns"])
    x_rr = [r["cumulative_compute_RR_GID_s"] for r in reuse]
    y_rr = [r["mean_design_regret_RR_GID"] for r in reuse]
    x_disc = [r["cumulative_compute_Discriminative_s"] for r in reuse]
    y_disc = [r["mean_design_regret_Discriminative"] for r in reuse]
    ax2.plot(x_rr, y_rr, marker="D", color=COLORS["learned RR-GID"], label="RR-GID (frozen G0)", linewidth=1.6)
    ax2.plot(x_disc, y_disc, marker="^", color=COLORS["Discriminative Score OED"],
             label="Discriminative (retrain)", linewidth=1.6)
    for r in reuse:
        ax2.annotate(f"T={r['campaigns']}", (r["cumulative_compute_RR_GID_s"], r["mean_design_regret_RR_GID"]),
                     fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax2.set_xscale("log")
    ax2.set_xlabel("Cumulative train+inference compute (s)")
    ax2.set_ylabel("Average design regret $\\Phi(\\hat p)/\\Phi(p^*)$")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8)
    ax2.set_title("S2: generator reuse amortized frontier")

    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/fig2_s2_alpha_reuse.png", dpi=200)
    print("saved figures/fig2_s2_alpha_reuse.png")


if __name__ == "__main__":
    main()
