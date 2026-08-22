"""P9 R1 smoke: learned-generator panel ranking vs empirical oracle (PDF 8.2)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from rr_gid_cn.gas_preprocess import fit_scaler_pca, panel_library, transform_features
from rr_gid_cn.vaeac import VAEAC, VAEACGenerator, learned_information


def spearmanr(a, b):
    a_r = np.argsort(np.argsort(a)).astype(float)
    b_r = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(a_r, b_r)[0, 1])


def empirical_panel_information(ref_x, phi, coord_panels, k=60):
    """Kernel-weighted conditional-mean information I_S from the empirical base."""
    infos = []
    n = min(len(ref_x), 400)
    idx = np.random.default_rng(0).choice(len(ref_x), n, replace=False)
    X = ref_x[idx]
    P = phi[idx]
    for panel in coord_panels:
        Xp = X[:, list(panel)]
        projected = np.empty((n, P.shape[1]))
        for i in range(n):
            d2 = np.sum((Xp - Xp[i]) ** 2, axis=1)
            d2[i] = np.inf
            nn = np.argpartition(d2, k)[:k]
            w = np.exp(-d2[nn] / (2 * np.median(d2[nn]) ** 2 + 1e-8))
            w /= w.sum()
            projected[i] = w @ P[nn]
        infos.append(np.cov(projected, rowvar=False))
    return np.asarray(infos)


def main() -> None:
    data = np.load("data/gas/processed/gas_processed.npz")
    ref_train, ref_val = data["ref_train"], data["ref_val"]
    mean, std, pcs = fit_scaler_pca(ref_train)
    phi_val = transform_features(ref_val, mean, std, pcs)
    coord_panels = tuple(
        tuple(j for s in pair for j in range(s * 8, (s + 1) * 8)) for pair in panel_library()
    )
    oracle_info = empirical_panel_information(ref_val, phi_val, coord_panels)
    ckpt = torch.load("experiments/p9_gas_vaeac.pt", map_location="cuda", weights_only=False)
    model = VAEAC(dim=128, latent=64, hidden=256, seed=0)
    model.load_state_dict(ckpt["model"])
    model.z_std = ckpt["z_std"]
    gen = VAEACGenerator(model, np.ones(128), alpha=0.0,
                         feature_fn=lambda x: transform_features(x, mean, std, pcs))
    _, learned_info = learned_information(gen, np.zeros(16), coord_panels,
                                          n_tilted=128, n_conditional=16, seed=1,
                                          feature_fn=lambda x: transform_features(x, mean, std, pcs))
    # panel ranking by trace of the information matrix
    tr_oracle = np.trace(oracle_info, axis1=1, axis2=2)
    tr_learned = np.trace(learned_info, axis1=1, axis2=2)
    rho = spearmanr(tr_oracle, tr_learned)
    op_err = float(max(np.linalg.norm(learned_info[i] - oracle_info[i], 2) for i in range(len(coord_panels))))
    summary = {
        "stage": "P9", "r1_smoke": True,
        "empirical_base": int(len(ref_val)), "panel_ranking_spearman": rho,
        "max_operator_error_learned": op_err,
        "n_panels": len(coord_panels),
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/p9_r1_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
