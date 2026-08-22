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


def calibrate_beta(phi, seed=2026, target_ess=0.5):
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=phi.shape[1])
    direction /= np.linalg.norm(direction)
    target = target_ess * len(phi)
    lo, hi = 0.0, 8.0
    for _ in range(50):
        mag = (lo + hi) / 2
        logits = phi @ (mag * direction)
        w = np.exp(logits - logits.max())
        ess = w.sum() ** 2 / np.sum(w ** 2)
        if ess > target:
            lo = mag
        else:
            hi = mag
    return ((lo + hi) / 2) * direction


def empirical_panel_information(ref_x, phi, beta, coord_panels, k=60, n=400, seed=1):
    """Kernel-weighted conditional-mean information I_S under the tilt beta*."""
    infos = []
    logits = phi @ beta
    w = np.exp(logits - logits.max())
    w /= w.sum()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ref_x), size=n, replace=True, p=w)
    X = ref_x[idx]
    P = phi[idx]
    for panel in coord_panels:
        Xp = X[:, list(panel)]
        projected = np.empty((n, P.shape[1]))
        for i in range(n):
            d2 = np.sum((Xp - Xp[i]) ** 2, axis=1)
            d2[i] = np.inf
            nn = np.argpartition(d2, k)[:k]
            wk = np.exp(-d2[nn] / (2 * np.median(d2[nn]) ** 2 + 1e-8))
            wk /= wk.sum()
            projected[i] = wk @ P[nn]
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
    gas_fn = lambda x: transform_features(x, mean, std, pcs)
    beta_star = calibrate_beta(phi_val)
    oracle_info = empirical_panel_information(ref_val, phi_val, beta_star, coord_panels)
    ckpt = torch.load("experiments/p9_gas_vaeac.pt", map_location="cuda", weights_only=False)
    model = VAEAC(dim=128, latent=64, hidden=256, seed=0)
    model.load_state_dict(ckpt["model"])
    model.z_std = ckpt["z_std"]
    gen = VAEACGenerator(model, np.ones(128), alpha=0.0, feature_fn=gas_fn)
    _, learned_info = learned_information(gen, beta_star, coord_panels,
                                          n_tilted=128, n_conditional=16, seed=1,
                                          feature_fn=gas_fn)
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
