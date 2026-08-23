"""P11 Fig.4: information fidelity + cumulative compute (PDF Fig.4).

Left: learned-generator operator error ``max_S ||I_hat_S - I_S||_op`` vs alpha
(S2), showing the frozen VAEAC stays accurate across nonlinearity.  Right:
cumulative inference compute vs alpha, contrasting frozen RR-GID with per-
campaign retrained Discriminative.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"learned RR-GID": "#2ca02c", "Discriminative Score OED": "#1f77b4"}


def main() -> None:
    s = json.load(open("results/p7_s2_summary.json", encoding="utf-8"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    alphas = [r["alpha"] for r in s["alpha_sweep"]]
    # --- Left: learned operator error vs alpha ---
    op_err = [r["max_operator_error_learned"] for r in s["alpha_sweep"]]
    ax1.plot(alphas, op_err, marker="D", color=COLORS["learned RR-GID"], linewidth=1.6)
    ax1.set_xlabel("Nonlinearity $\\alpha$")
    ax1.set_ylabel("$\\max_S\\|\\hat I_S - I_S\\|_{op}$")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("S2: learned-generator information fidelity")

    # --- Right: cumulative compute (T=50) per policy ---
    reuse = sorted(s["reuse"], key=lambda r: r["campaigns"])
    T = [r["campaigns"] for r in reuse]
    c_rr = [r["cumulative_compute_RR_GID_s"] for r in reuse]
    c_disc = [r["cumulative_compute_Discriminative_s"] for r in reuse]
    ax2.plot(T, c_rr, marker="D", color=COLORS["learned RR-GID"],
             label="RR-GID (frozen G0)", linewidth=1.6)
    ax2.plot(T, c_disc, marker="^", color=COLORS["Discriminative Score OED"],
             label="Discriminative (retrain)", linewidth=1.6)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("Campaigns $T$")
    ax2.set_ylabel("Cumulative train+inference compute (s)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8)
    ax2.set_title("S2: cumulative compute over reuse campaigns")

    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/fig4_info_compute.png", dpi=200)
    print("saved figures/fig4_info_compute.png")


if __name__ == "__main__":
    main()
