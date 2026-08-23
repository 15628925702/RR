"""P7 formal: Synthetic S2 nonlinearity sweep + generator reuse frontier (Fig.2).

Nonlinearity sweep (PDF 7.3, B=8000):
  alpha in {0, 0.5, 1.0, 1.5}; report the four policies' design ratio
  ``Phi(p)/Phi(p*)`` and the learned-generator operator error
  ``max_S ||I_hat_S - I_S||_op``.

Generator reuse (PDF 7.3):
  G0 is trained once (frozen P6 VAEAC). For T in {1, 5, 20, 50} campaigns we
  redraw a 12-feature map from the bounded unary/pairwise dictionary, redraw
  ``beta*`` and redraw a subset of candidate pair panels.  RR-GID reuses the
  frozen G0 (no retraining); Discriminative Score OED retrains its score
  network every campaign.  We report cumulative train+inference compute vs.
  average design regret ``Phi(phat)/Phi(p*)``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

_here = Path(__file__).resolve().parent
_src = _here.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from rr_gid_cn.policies import frank_wolfe, uniform_probabilities
from rr_gid_cn.s1_gate import a_optimal_information, discriminative_design, exact_panel_information
from rr_gid_cn.synthetic_oracle import (all_pairs, beta_direction_and_scale, feature_fn_from_dictionary,
                                        make_feature_dictionary, make_frozen_mixture, reference_scale,
                                        sample_feature_draw, sample_full)
from rr_gid_cn.vaeac import VAEACGenerator, learned_information, load_vaeac_checkpoint


def phi_value(p, fisher, infos):
    return float(np.trace(fisher @ np.linalg.pinv(np.tensordot(p, infos, axes=(0, 0)))))


def a_optimal_design(reference, panels):
    """A-OSQD design via the complete-reference-covariance A-optimal objective."""
    a_info = a_optimal_information(reference, panels)
    p_a, _, _ = frank_wolfe(np.eye(16), a_info, np.ones(len(panels)),
                            uniform_probabilities(len(panels)), tolerance=1e-4, max_iter=300)
    return p_a


def load_generator(ckpt_path: str):
    base_model, _ = load_vaeac_checkpoint(ckpt_path, device="cuda", expected_dim=16)
    return base_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/p7_formal.yaml"))
    ap.add_argument("--alpha-only", action="store_true")
    ap.add_argument("--reuse-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    with args.config.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)["p7"]
    if args.smoke:
        cfg["reference_size"] = 2000
        cfg["information_tilted"] = 32
        cfg["information_cond"] = 8
        cfg["large_reference_size"] = 5000
        cfg["validation_size"] = 1000
        cfg["reuse_campaigns"] = [1, 5]
        cfg["reuse_reps"] = 2
        cfg["gen_info_tilted"] = 64
        cfg["gen_info_cond"] = 8
        cfg["mlp_steps"] = 30

    base_model = load_generator(str(Path(cfg["generator_ckpt"])))
    summary = {"stage": "P7", "budget": int(cfg["budget"])}
    if not args.reuse_only:
        summary["alpha_sweep"] = run_alpha_sweep(cfg, base_model)
    if not args.alpha_only:
        summary["reuse"] = run_reuse(cfg, base_model)
    Path("results").mkdir(exist_ok=True)
    Path("results/p7_s2_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_alpha_sweep(cfg, base_model):
    rows = []
    for alpha in cfg["alpha_values"]:
        mix = make_frozen_mixture(seed=2026, alpha=alpha)
        scale = reference_scale(mix, cfg["reference_size"], 2026)
        panels = all_pairs()
        reference = sample_full(mix, cfg["reference_size"], 2026)
        beta_true = beta_direction_and_scale(reference, 2026, 0.5, scale)
        fisher, info = exact_panel_information(mix, beta_true, panels, reference, scale,
                                               cfg["information_tilted"], cfg["information_cond"], seed=2026 + 1)
        p_star, _, _ = frank_wolfe(fisher, info, np.ones(len(panels)),
                                   uniform_probabilities(len(panels)), tolerance=1e-4, max_iter=300)
        p_uniform = uniform_probabilities(len(panels))
        p_a = a_optimal_design(reference, panels)
        gen = VAEACGenerator(base_model, scale, alpha=alpha)
        _, learned_infos = learned_information(gen, beta_true, panels,
                                               n_tilted=cfg["gen_info_tilted"],
                                               n_conditional=cfg["gen_info_cond"], seed=1)
        op_err = float(max(np.linalg.norm(learned_infos[i] - info[i], 2) for i in range(len(panels))))
        p_learned, _, _ = frank_wolfe(fisher, learned_infos, np.ones(len(panels)),
                                      uniform_probabilities(len(panels)), tolerance=1e-4, max_iter=300)
        validation = sample_full(mix, cfg["validation_size"], 2026 + 555)
        p_disc = discriminative_design(reference, validation, beta_true, panels, scale,
                                       seed=11, hidden=64, steps=cfg["mlp_steps"])
        base = phi_value(p_star, fisher, info)
        row = {"alpha": alpha,
               "design_ratio_oracle RR-GID": 1.0,
               "design_ratio_Uniform SQD": phi_value(p_uniform, fisher, info) / base,
               "design_ratio_A-OSQD": phi_value(p_a, fisher, info) / base,
               "design_ratio_learned RR-GID": phi_value(p_learned, fisher, info) / base,
               "design_ratio_Discriminative Score OED": phi_value(p_disc, fisher, info) / base,
               "max_operator_error_learned": op_err}
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def run_reuse(cfg, base_model):
    """Reuse frontier: cumulative compute vs average design regret."""
    dicto = make_feature_dictionary()
    rows = []
    for T in cfg["reuse_campaigns"]:
        rep_ratios = {p: [] for p in ("RR-GID", "Discriminative Score OED", "Uniform SQD", "A-OSQD")}
        cum_rr = 0.0
        cum_disc = 0.0
        for rep in range(cfg["reuse_reps"]):
            feats = sample_feature_draw(dicto, 1000 + T * 100 + rep, r=cfg["r_features"])
            fn = feature_fn_from_dictionary(feats, np.ones(16))
            mix = make_frozen_mixture(seed=2026, alpha=1.0)
            scale = reference_scale(mix, cfg["reference_size"], 2026)
            ref = sample_full(mix, cfg["reference_size"], 2026 + rep)
            beta_true = beta_direction_and_scale(ref, 2000 + T * 100 + rep, 0.5, scale, feature_fn=fn)
            rng = np.random.default_rng(3000 + T * 100 + rep)
            panels = all_pairs()
            panels_sub = tuple(panels[i] for i in rng.choice(len(panels), size=cfg["reuse_panels"], replace=False))
            fisher, info = exact_panel_information(mix, beta_true, panels_sub, ref, scale,
                                                   cfg["information_tilted"], cfg["information_cond"],
                                                   4000 + rep, feature_fn=fn)
            p_star, _, _ = frank_wolfe(fisher, info, np.ones(len(panels_sub)),
                                       uniform_probabilities(len(panels_sub)), tolerance=1e-4, max_iter=300)
            # RR-GID: reuse frozen G0 with this campaign's feature fn
            gen = VAEACGenerator(base_model, scale, alpha=1.0, feature_fn=fn)
            t0 = time.time()
            _, learned_infos = learned_information(gen, beta_true, panels_sub,
                                                   n_tilted=cfg["gen_info_tilted"],
                                                   n_conditional=cfg["gen_info_cond"], seed=rep + 77,
                                                   feature_fn=fn)
            cum_rr += time.time() - t0
            p_learned, _, _ = frank_wolfe(fisher, learned_infos, np.ones(len(panels_sub)),
                                          uniform_probabilities(len(panels_sub)), tolerance=1e-4, max_iter=300)
            rep_ratios["RR-GID"].append(phi_value(p_learned, fisher, info) / phi_value(p_star, fisher, info))
            # Discriminative: retrain per campaign
            validation = sample_full(mix, cfg["validation_size"], 2026 + 555 + rep)
            t0 = time.time()
            p_disc = discriminative_design(ref, validation, beta_true, panels_sub, scale,
                                           seed=11 + rep, hidden=64, steps=cfg["mlp_steps"], feature_fn=fn)
            cum_disc += time.time() - t0
            rep_ratios["Discriminative Score OED"].append(phi_value(p_disc, fisher, info) / phi_value(p_star, fisher, info))
            # baselines (cheap, no retrain)
            rep_ratios["Uniform SQD"].append(phi_value(uniform_probabilities(len(panels_sub)), fisher, info) / phi_value(p_star, fisher, info))
            p_a = a_optimal_design(ref, panels_sub)
            rep_ratios["A-OSQD"].append(phi_value(p_a, fisher, info) / phi_value(p_star, fisher, info))
        n = max(cfg["reuse_reps"], 1)
        row = {"campaigns": T, "reps": cfg["reuse_reps"],
               "cumulative_compute_RR_GID_s": round(cum_rr, 1),
               "cumulative_compute_Discriminative_s": round(cum_disc, 1),
               "mean_design_regret_RR_GID": round(float(np.mean(rep_ratios["RR-GID"])), 4),
               "mean_design_regret_Discriminative": round(float(np.mean(rep_ratios["Discriminative Score OED"])), 4),
               "mean_design_regret_Uniform": round(float(np.mean(rep_ratios["Uniform SQD"])), 4),
               "mean_design_regret_A_OSQD": round(float(np.mean(rep_ratios["A-OSQD"])), 4),
               "se_regret_RR_GID": round(float(np.std(rep_ratios["RR-GID"]) / np.sqrt(n)), 4)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


if __name__ == "__main__":
    main()
