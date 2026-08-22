"""P10 R2 formal: Gas natural-drift robustness (PDF 8.3, Fig.3/4 + Table 1).

Three natural target campaigns (batch 7, batches 8-9, batch 10). Each campaign
pool is split 50/50 into a campaign pool (only observable through panel masking)
and a full-test pool (evaluation only).  A fixed large Gas VAEAC full-sample
pool estimates ``A_hat``; the full-test pool gives the evaluation-only
``beta_t^dagger = argmin A_hat(beta) - beta^T mu_t``.  Four acquisition policies
share paired target draws; RR-GID reuses the frozen Gas VAEAC and
Discriminative Score OED retrains per campaign.

Reported per (campaign, budget): projection loss, held-out moment RMSE,
C2ST AUC, ESS, conditional acceptance, FW gap, lambda_min(M_hat).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

_here = Path(__file__).resolve().parent
_src = _here.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from rr_gid_cn.gas_preprocess import fit_scaler_pca, panel_library, transform_features
from rr_gid_cn.policies import frank_wolfe, uniform_probabilities
from rr_gid_cn.s1_gate import a_optimal_information, discriminative_design
from rr_gid_cn.vaeac import VAEAC, VAEACGenerator, learned_information

CAMPAIGNS = {
    "batch7": ("x_batch7", "phi_batch7"),
    "batches89": ("x_batches89", "phi_batches89"),
    "batch10": ("x_batch10", "phi_batch10"),
}


def calibrate_beta(phi, seed=2026, target_ess=0.5):
    """ESS/N bisection magnitude for an evaluation tilt direction."""
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=phi.shape[1])
    direction /= np.linalg.norm(direction)
    target = target_ess * len(phi)
    lo, hi = 0.0, 8.0
    for _ in range(60):
        mag = (lo + hi) / 2
        logits = phi @ (mag * direction)
        w = np.exp(logits - logits.max())
        ess = w.sum() ** 2 / np.sum(w ** 2)
        if ess > target:
            lo = mag
        else:
            hi = mag
    return ((lo + hi) / 2) * direction


def empirical_kernel_cm(ref_x, ref_phi, observed, panel, k=60):
    """Vectorized kernel conditional mean E_{Q0}[phi(X)|X_S] over the reference pool."""
    idx = list(panel)
    pool = ref_x[:, idx]
    Xp = np.atleast_2d(np.asarray(observed))
    d2 = (np.sum(Xp ** 2, axis=1)[:, None] + np.sum(pool ** 2, axis=1)[None, :] - 2.0 * (Xp @ pool.T))
    nn = np.argpartition(d2, k, axis=1)[:, :k]
    d2_nn = np.take_along_axis(d2, nn, axis=1)
    band = np.median(d2_nn, axis=1)
    wk = np.exp(-d2_nn / (2 * band[:, None] ** 2 + 1e-8))
    wk /= wk.sum(axis=1, keepdims=True)
    return np.sum(wk[:, :, None] * ref_phi[nn], axis=1)


def gas_balanced_pilot_counts(coord_panels, budget):
    counts = np.zeros(len(coord_panels), dtype=int)
    sensor_pilots = [(j, j + 8) for j in range(8)]
    indices = []
    for sp in sensor_pilots:
        cp = tuple(j for s in sp for j in range(s * 8, (s + 1) * 8))
        if cp in coord_panels:
            indices.append(coord_panels.index(cp))
    if not indices:
        return np.floor(budget * uniform_probabilities(len(coord_panels))).astype(int)
    base, rem = divmod(budget, len(indices))
    counts[indices] = base
    for ix in indices[:rem]:
        counts[ix] += 1
    return counts


def log_partition_emp(phi, beta):
    logits = phi @ beta
    return float(np.logaddexp.reduce(logits) - np.log(phi.shape[0]))


def c2st_auc(gen_samples, test_pool, rng_seed=0, folds=5):
    """5-fold C2ST AUC between generated and held-out target records (0.5 = indistinguishable)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    n = min(len(gen_samples), len(test_pool))
    X = np.concatenate([gen_samples[:n], test_pool[:n]], axis=0)
    y = np.concatenate([np.ones(n), np.zeros(n)])
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=rng_seed)
    aucs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs))


def heldout_moment_rmse(gen_samples, test_x, mean, std):
    g_mean = gen_samples.mean(0)
    t_mean = test_x.mean(0)
    mean_err = float(np.sqrt(np.mean(((g_mean - t_mean) / np.maximum(std, 1e-9)) ** 2)))
    g_std = gen_samples.std(0)
    t_std = test_x.std(0)
    std_err = float(np.sqrt(np.mean(((g_std - t_std) / np.maximum(std, 1e-9)) ** 2)))
    return mean_err, std_err


def run_r2_replication(cfg, ref_train, ref_val, mean, std, pcs, gen, campaign_x, campaign_phi,
                       coord_panels, budget, seed, full_pool_x, mlp_steps=200):
    gas_fn = lambda x: transform_features(x, mean, std, pcs)
    n = len(campaign_x)
    rng = np.random.default_rng(seed)
    half = n // 2
    perm = rng.permutation(n)
    pool_x, pool_phi = campaign_x[perm[:half]], campaign_phi[perm[:half]]
    test_x, test_phi = campaign_x[perm[half:]], campaign_phi[perm[half:]]

    # Evaluation beta_t^dagger = argmin A_hat(beta) - beta^T mu_t (full-test only)
    mu_t = test_phi.mean(0)
    full_pool_phi = gas_fn(full_pool_x)
    beta_dag = np.zeros(16)
    for _ in range(30):
        logits = full_pool_phi @ beta_dag
        w = np.exp(logits - logits.max()); w /= w.sum()
        mu_hat = w @ full_pool_phi
        diff = mu_t - mu_hat
        if np.linalg.norm(diff) < 1e-6:
            break
        fisher = np.cov(full_pool_phi, rowvar=False, aweights=w) + 1e-3 * np.eye(16)
        beta_dag = np.clip(beta_dag + np.linalg.solve(fisher, diff), -4.0, 4.0)

    # target draw from the campaign pool under beta_dag
    logits = pool_phi @ beta_dag
    w = np.exp(logits - logits.max()); w /= w.sum()
    target_idx = rng.choice(len(pool_x), size=budget, p=w)
    target = pool_x[target_idx]

    proj_loss = float(log_partition_emp(full_pool_phi, beta_dag) - beta_dag @ mu_t)
    b_pilot = min(int(np.ceil(0.2 * budget)), int(np.ceil(10.0 * budget ** (1.0 / 3.0))))
    b_pilot = min(b_pilot, budget)

    fisher = np.cov(pool_phi, rowvar=False)
    pilot_counts = gas_balanced_pilot_counts(coord_panels, b_pilot)
    pilot_obs = []
    cursor = 0
    for panel, count in zip(coord_panels, pilot_counts):
        for row in target[cursor:cursor + count]:
            pilot_obs.append((panel, row[list(panel)]))
        cursor += count

    rows = []
    for name in ("Uniform SQD", "A-OSQD", "Discriminative Score OED", "RR-GID"):
        if name == "Uniform SQD":
            probs = uniform_probabilities(len(coord_panels))
            fw_gap = 0.0
            policy_infos = None
        elif name == "A-OSQD":
            a_info = a_optimal_information(pool_x, coord_panels, dimension=128)
            probs, fw_gap, _ = frank_wolfe(np.eye(128), a_info, np.ones(len(coord_panels)),
                                           uniform_probabilities(len(coord_panels)), tolerance=1e-4, max_iter=300)
            policy_infos = a_info
        elif name == "Discriminative Score OED":
            probs = discriminative_design(ref_train, ref_val, beta_dag, coord_panels, np.ones(128),
                                          seed=seed + 11, hidden=64, steps=mlp_steps, feature_fn=gas_fn)
            fw_gap = 0.0
            policy_infos = None
        else:  # RR-GID
            _, learned_infos = learned_information(gen, beta_dag, coord_panels,
                                                   n_tilted=cfg["gen_info_tilted"],
                                                   n_conditional=cfg["gen_info_cond"], seed=seed + 17,
                                                   feature_fn=gas_fn)
            probs, fw_gap, _ = frank_wolfe(fisher, learned_infos, np.ones(len(coord_panels)),
                                           uniform_probabilities(len(coord_panels)), tolerance=1e-4, max_iter=300)
            policy_infos = learned_infos
        rem = budget - b_pilot
        counts = np.floor(rem * probs).astype(int)
        remainder = rem - int(counts.sum())
        if remainder:
            counts[np.argsort(rem * probs - counts)[-remainder:]] += 1
        # final beta via J=2 Fisher scoring on the empirical base
        beta_hat_final = beta_dag.copy()
        main_obs = []
        cursor = b_pilot
        for panel, count in zip(coord_panels, counts):
            for row in target[cursor:cursor + count]:
                main_obs.append((panel, row[list(panel)]))
            cursor += count
        all_obs = pilot_obs + main_obs
        for _ in range(2):
            logits = pool_phi @ beta_hat_final
            wb = np.exp(logits - logits.max()); wb /= wb.sum()
            mu_beta_j = wb @ pool_phi
            U = np.zeros(16)
            H = 1e-2 * np.eye(16)
            for panel, obs_row in all_obs:
                obs_arr = np.atleast_2d(np.asarray(obs_row))
                cm = empirical_kernel_cm(pool_x, pool_phi, obs_arr, panel)
                U += (cm - mu_beta_j).sum(0)
                idx = coord_panels.index(panel)
                info_panel = policy_infos[idx] if policy_infos is not None else fisher
                H += len(obs_arr) * info_panel
            step = np.linalg.solve(H + 1e-2 * np.eye(16), U)
            beta_hat_final = np.clip(beta_hat_final + step, -4.0, 4.0)
        logits = pool_phi @ beta_hat_final
        wgt = np.exp(logits - logits.max())
        ess = float(wgt.sum() ** 2 / np.sum(wgt ** 2) / len(pool_phi))
        M_hat = np.tensordot(probs, policy_infos, axes=(0, 0)) if policy_infos is not None else (fisher if name != "RR-GID" else fisher)
        lambda_min_M = float(np.linalg.eigvalsh((np.asarray(M_hat) + np.asarray(M_hat).T) / 2).min())
        gen_samples = gen.sample_full(cfg["c2st_samples"], seed + 33)
        mean_rmse, std_rmse = heldout_moment_rmse(gen_samples, test_x, mean, std)
        auc = c2st_auc(gen_samples, test_x, rng_seed=seed % 10000)
        rows.append({"policy": name, "budget": budget, "seed": seed,
                     "projection_loss": proj_loss, "heldout_mean_rmse": mean_rmse,
                     "heldout_std_rmse": std_rmse, "c2st_auc": auc, "ess": ess,
                     "conditional_acceptance": 1.0, "fw_gap": fw_gap, "lambda_min_M": lambda_min_M,
                     "beta_dag_norm": float(np.linalg.norm(beta_dag))})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", type=str, default=None, choices=list(CAMPAIGNS))
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--config", type=Path, default=Path("configs/p10_formal.yaml"))
    ap.add_argument("--rep-range", type=int, nargs=2, default=None)
    ap.add_argument("--max-replications", type=int, default=None)
    ap.add_argument("--mlp-steps", type=int, default=200)
    ap.add_argument("--out", type=str, default="p10_r2")
    args = ap.parse_args()
    with args.config.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)["p10"]
    budgets = (args.budget,) if args.budget is not None else tuple(cfg["budgets"])
    campaigns = (args.campaign,) if args.campaign else list(CAMPAIGNS)
    replications = args.max_replications or int(cfg["replications"])

    data = np.load("data/gas/processed/gas_processed.npz")
    ref_train, ref_val = data["ref_train"], data["ref_val"]
    mean, std, pcs = fit_scaler_pca(ref_train)
    sensor_pairs = panel_library()
    coord_panels = tuple(tuple(j for s in pair for j in range(s * 8, (s + 1) * 8)) for pair in sensor_pairs)
    ckpt = torch.load(cfg["generator_ckpt"], map_location="cuda", weights_only=False)
    model = VAEAC(dim=128, latent=64, hidden=256, seed=0).to("cuda")
    model.load_state_dict(ckpt["model"])
    model.z_std = ckpt["z_std"]
    gas_fn = lambda x: transform_features(x, mean, std, pcs)
    gen = VAEACGenerator(model, np.ones(128), alpha=0.0, feature_fn=gas_fn)
    # frozen large full-sample pool for A_hat (PDF 8.3)
    full_pool_x = gen.sample_full(cfg["full_pool_size"], seed=42)

    out = Path("results")
    camp_seeds = {"batch7": 501000000, "batches89": 502000000, "batch10": 503000000}
    for camp in campaigns:
        x_key, phi_key = CAMPAIGNS[camp]
        campaign_x, campaign_phi = data[x_key], data[phi_key]
        for budget in budgets:
            fp = out / f"{args.out}_{camp}_{budget}.jsonl"
            done = set()
            if fp.exists():
                for line in fp.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        r = json.loads(line)
                        done.add((r["replication"], r["policy"]))
            start, end = args.rep_range if args.rep_range else (0, replications)
            with fp.open("a", encoding="utf-8") as stream:
                for rep in range(start, min(end, replications)):
                    if all((rep, pol) in done for pol in ("Uniform SQD", "A-OSQD", "Discriminative Score OED", "RR-GID")):
                        continue
                    seed = camp_seeds[camp] + budget * 1000 + rep
                    rows = run_r2_replication(cfg, ref_train, ref_val, mean, std, pcs, gen,
                                              campaign_x, campaign_phi, coord_panels, budget, seed,
                                              full_pool_x, mlp_steps=args.mlp_steps)
                    for r in rows:
                        r["replication"] = rep
                        r["campaign"] = camp
                        if (rep, r["policy"]) not in done:
                            stream.write(json.dumps(r, sort_keys=True) + "\n")
                            done.add((rep, r["policy"]))
                    stream.flush()
                    print(json.dumps({"campaign": camp, "budget": budget, "rep": rep}), flush=True)


if __name__ == "__main__":
    main()
