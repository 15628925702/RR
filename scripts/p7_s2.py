"""P7 Synthetic S2: nonlinearity (alpha) sweep design ratios and learned info error."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from rr_gid_cn.s1_gate import prepare_s1_oracle
from rr_gid_cn.synthetic_oracle import all_pairs, make_frozen_mixture, reference_scale
from rr_gid_cn.vaeac import VAEAC, VAEACGenerator, learned_information


def phi(p, fisher, infos):
    return float(np.trace(fisher @ np.linalg.pinv(np.tensordot(p, infos, axes=(0, 0)))))


def main() -> None:
    ckpt = torch.load("experiments/p6_vaeac_synthetic.pt", map_location="cuda", weights_only=False)
    base_model = VAEAC(dim=16, latent=128, hidden=512, seed=0)
    base_model.load_state_dict(ckpt["model"])
    base_model.z_std = ckpt["z_std"]
    rows = []
    for alpha in (0.0, 0.5, 1.0, 1.5):
        mix = make_frozen_mixture(seed=2026, alpha=alpha)
        scale = reference_scale(mix, 6000, 2026)
        panels = all_pairs()
        prepared = prepare_s1_oracle(mix, scale, panels, seed=2026, reference_size=2000,
                                     information_samples=32, conditional_samples=8, large_reference_size=5000)
        fisher = prepared["fisher"]
        info = prepared["information"]
        designs = prepared["designs"]
        p_star = designs["oracle RR-GID"]
        gen = VAEACGenerator(base_model, scale, alpha=alpha)
        _, learned_infos = learned_information(gen, prepared["beta_true"], panels,
                                               n_tilted=128, n_conditional=16, seed=1)
        # operator error of the learned generator vs oracle
        op_err = float(max(np.linalg.norm(learned_infos[i] - info[i], 2) for i in range(len(panels))))
        row = {"alpha": alpha}
        for name, p in designs.items():
            row[f"design_ratio_{name}"] = phi(p, fisher, info) / phi(p_star, fisher, info)
        # learned RR-GID design ratio (via FW on learned infos)
        from rr_gid_cn.policies import frank_wolfe, uniform_probabilities
        p_learned, _, _ = frank_wolfe(fisher, learned_infos, np.ones(len(panels)),
                                      uniform_probabilities(len(panels)), tolerance=1e-4, max_iter=300)
        row["design_ratio_learned RR-GID"] = phi(p_learned, fisher, info) / phi(p_star, fisher, info)
        row["max_operator_error_learned"] = op_err
        rows.append(row)
    summary = {"stage": "P7", "budget_smoke": 800, "alpha_sweep": rows}
    Path("results").mkdir(exist_ok=True)
    Path("results/p7_s2_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
