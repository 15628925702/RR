"""P11 Fig.1: Synthetic S1 B*KL curves + J ablation (PDF Fig.1)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

POLICIES = ["oracle RR-GID", "Uniform SQD", "A-OSQD"]
THEORY = {"oracle RR-GID": 31.9, "Uniform SQD": 64.8, "A-OSQD": 95.8}


def _mean_bkl(path: Path, policy: str, budget: int) -> float:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    vals = [r["B_kl"] for r in rows if r["policy"] == policy and r["budget"] == budget]
    return float(np.mean(vals))


def main() -> None:
    summary = json.load(open("results/p4_formal_summary.json", encoding="utf-8"))
    budgets = sorted((int(b) for b in summary), key=int)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for pol in POLICIES:
        ys = [summary[str(b)][pol]["mean_B_kl"] for b in budgets]
        errs = [summary[str(b)][pol]["mc_se"] for b in budgets]
        ax1.errorbar(budgets, ys, yerr=errs, marker="o", capsize=3, label=pol)
        ax1.axhline(THEORY[pol], ls="--", alpha=0.5)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Budget $B$")
    ax1.set_ylabel("$B\\cdot\\mathrm{KL}(Q_{\\beta^*} \\| Q_{\\hat\\beta})$")
    ax1.legend(fontsize=8)
    ax1.set_title("S1: $B\\cdot\\mathrm{KL}$ (dashed = $\\frac{1}{2}\\Phi$)")
    ax1.grid(True, which="both", alpha=0.3)

    # J ablation at B=8000
    jvals = {pol: [] for pol in POLICIES}
    for j in (0, 1, 2):
        for pol in POLICIES:
            if j == 2:
                v = summary["8000"][pol]["mean_B_kl"]
            else:
                v = _mean_bkl(Path(f"results/p4_exact_8000_J{j}.jsonl"), pol, 8000)
            jvals[pol].append(v)
    for pol in POLICIES:
        ax2.plot([0, 1, 2], jvals[pol], marker="o", label=pol)
    ax2.set_xlabel("Fisher-scoring steps $J$")
    ax2.set_ylabel("$B\\cdot\\mathrm{KL}$ (B=8000)")
    ax2.set_yscale("log")
    ax2.legend(fontsize=8)
    ax2.set_title("J ablation (B=8000)")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/fig1_s1_bkl_jablation.png", dpi=200)
    print("saved figures/fig1_s1_bkl_jablation.png")


if __name__ == "__main__":
    main()
