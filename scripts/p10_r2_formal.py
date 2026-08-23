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
from rr_gid_cn.vaeac import VAEACGenerator, learned_information, load_vaeac_checkpoint

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


def bregman_projection_loss(phi_pool, beta_hat, beta_dag):
    """Eq. (17): ``D_Ahat(beta_hat, beta_dag)`` for the frozen family.

    The gradient is evaluated at ``beta_dag`` from the same fixed full-sample
    pool.  Using ``A(beta_hat)-beta_dag^T mu_t`` alone is the objective value,
    not the Bregman projection loss, and can be negative when the numerical
    projection has not reached its unconstrained optimum.
    """
    phi_pool = np.asarray(phi_pool, dtype=float)
    beta_hat, beta_dag = np.asarray(beta_hat, dtype=float), np.asarray(beta_dag, dtype=float)
    logits = phi_pool @ beta_dag
    logits -= np.max(logits)
    w = np.exp(logits); w /= np.sum(w)
    grad_dag = w @ phi_pool
    raw = (log_partition_emp(phi_pool, beta_hat)
           - log_partition_emp(phi_pool, beta_dag)
           - float(grad_dag @ (beta_hat - beta_dag)))
    # log-sum-exp is convex; only roundoff at machine precision is clipped.
    if raw < -1e-8:
        raise FloatingPointError(f"negative Bregman divergence {raw:.6g}")
    return float(max(raw, 0.0))


def fit_pc2(reference_train, mean, std):
    """Freeze the second within-sensor PC used by the 32 held-out functions."""
    z = (np.asarray(reference_train, dtype=float) - mean) / np.maximum(std, 1e-12)
    pcs2 = []
    for sensor in range(16):
        _, _, vh = np.linalg.svd(z[:, sensor * 8:(sensor + 1) * 8], full_matrices=False)
        vec = vh[1]
        if vec[np.argmax(np.abs(vec))] < 0:
            vec = -vec
        pcs2.append(vec)
    return np.asarray(pcs2)


def heldout_function_values(x, mean, std, pcs2):
    """The fixed 32 held-out functions: 16 PC2 unary + 16 interactions."""
    z = (np.asarray(x, dtype=float) - mean) / np.maximum(std, 1e-12)
    scores = np.stack([
        np.einsum("...i,i->...", z[..., s * 8:(s + 1) * 8], pcs2[s])
        for s in range(16)
    ], axis=-1)
    unary = np.tanh(scores)
    interaction = np.tanh(scores * np.roll(scores, -1, axis=-1))
    return np.concatenate([unary, interaction], axis=-1)


def conditional_acceptance_diagnostic(generator, beta, observations, panels, n=8, seed=0):
    """Measure exact conditional tilt acceptance on a fixed diagnostic subset."""
    total_accept, total_proposals, ess = 0.0, 0.0, []
    for i, (obs, panel) in enumerate(zip(observations, panels)):
        _, acc, e = generator.tilted_conditional_diagnostics(
            beta, obs, panel, n, seed=seed + i)
        total_accept += float(acc) * n
        total_proposals += n
        ess.append(float(e))
    if not ess:
        return float("nan"), float("nan")
    return float(total_accept / max(total_proposals, 1.0)), float(np.mean(ess))


def gas_a_optimal_information(phi_ref, sensor_pairs):
    """A-OSQD panel Fisher info in the 16-dim phi space (PDF Sec. 6 / 8.1)."""
    x = np.asarray(phi_ref, dtype=float)
    dim = x.shape[1]
    full_cov = np.cov(x, rowvar=False)
    inv_full = np.linalg.inv(full_cov + 1e-6 * np.eye(dim))
    infos = []
    for a, b in sensor_pairs:
        idx = sorted(set([a, b] + ([a + 8] if a + 8 < dim else []) + ([b + 8] if b + 8 < dim else [])))
        info = np.zeros((dim, dim))
        info[np.ix_(idx, idx)] = inv_full[np.ix_(idx, idx)]
        infos.append(info)
    return np.asarray(infos)


def empirical_panel_infos(pool_x, pool_phi, coord_panels, k=60, n_sub=800):
    """Kernel cross-completion panel information (PDF Eq. 9) on the empirical pool."""
    rng = np.random.default_rng(0)
    n = len(pool_x)
    sub = np.minimum(n, n_sub)
    idx = rng.choice(n, size=sub, replace=False)
    Xs, Ps = pool_x[idx], pool_phi[idx]
    mu = Ps.mean(0)
    infos = []
    for panel in coord_panels:
        obs = Xs[:, list(panel)]
        a = empirical_kernel_cm(pool_x, pool_phi, obs, panel) - mu
        b = empirical_kernel_cm(pool_x, pool_phi, obs, panel) - mu
        a = a - a.mean(0); b = b - b.mean(0)
        info = (a.T @ b + b.T @ a) / max(2 * (len(a) - 1), 1)
        vals, vecs = np.linalg.eigh((info + info.T) / 2)
        infos.append((vecs * np.maximum(vals, 1e-10)) @ vecs.T)
    return np.asarray(infos)


def c2st_auc(gen_samples, test_pool, rng_seed=0, folds=5, max_iter=5000):
    """5-fold C2ST AUC between generated and held-out target records (0.5 = indistinguishable).

    Inputs are standardized to unit variance per feature before the 5-fold
    logistic-regression classifier so the discriminator actually converges on
    the 128-dim standardized feature space (PDF 8.3's C2ST).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    n = min(len(gen_samples), len(test_pool))
    X = np.concatenate([gen_samples[:n], test_pool[:n]], axis=0)
    y = np.concatenate([np.ones(n), np.zeros(n)])
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=rng_seed)
    aucs = []
    for tr, te in skf.split(X, y):
        # per-fold standardization on the training partition only (no leakage)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        clf = LogisticRegression(max_iter=max_iter)
        clf.fit((X[tr] - mu) / sd, y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba((X[te] - mu) / sd)[:, 1]))
    return float(np.mean(aucs))


def heldout_moment_rmse(gen_samples, test_x, mean, std, pcs2):
    g = heldout_function_values(gen_samples, mean, std, pcs2)
    t = heldout_function_values(test_x, mean, std, pcs2)
    mean_err = float(np.sqrt(np.mean((g.mean(0) - t.mean(0)) ** 2)))
    std_err = float(np.sqrt(np.mean((g.std(0) - t.std(0)) ** 2)))
    return mean_err, std_err


def run_r2_replication(cfg, ref_train, ref_val, mean, std, pcs, pcs2, gen, campaign_x, campaign_phi,
                       coord_panels, sensor_pairs, budget, seed, full_pool_x, mlp_steps=200,
                       emp_infos=None, gas_ao_infos=None, phi_ref=None):
    import time as _t
    t0 = _t.time()
    gas_fn = lambda x: transform_features(x, mean, std, pcs)
    n = len(campaign_x)
    rng = np.random.default_rng(seed)
    half = n // 2
    perm = rng.permutation(n)
    pool_x, pool_phi = campaign_x[perm[:half]], campaign_phi[perm[:half]]
    test_x, test_phi = campaign_x[perm[half:]], campaign_phi[perm[half:]]
    print(f"[rep {seed} t+{_t.time()-t0:.1f}s split done", flush=True)

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
        step_dag = np.linalg.solve(fisher, diff)
        if np.linalg.norm(step_dag) > 2.0:
            step_dag = step_dag * (2.0 / np.linalg.norm(step_dag))
        beta_dag = np.clip(beta_dag + step_dag, -4.0, 4.0)
    if np.linalg.norm(beta_dag) > 2.0:
        beta_dag = beta_dag * (2.0 / np.linalg.norm(beta_dag))

    print(f"[rep {seed} t+{_t.time()-t0:.1f}s beta_dag done norm={np.linalg.norm(beta_dag):.2f}", flush=True)
    # Natural-drift records are the acquisition target.  The full-test split
    # and beta_dag are evaluation-only and must not influence acquisition.
    target_idx = rng.choice(len(pool_x), size=budget, replace=True)
    target = pool_x[target_idx]

    b_pilot = min(int(np.ceil(0.2 * budget)), int(np.ceil(10.0 * budget ** (1.0 / 3.0))))
    b_pilot = min(b_pilot, budget)

    fisher = np.cov(pool_phi, rowvar=False)
    # empirical kernel panel infos shared by non-generator policies and H
    # (computed once in main on the reference pool when not passed in).
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
            a_info = gas_ao_infos if gas_ao_infos is not None else gas_a_optimal_information(pool_phi, sensor_pairs)
            probs, fw_gap, _ = frank_wolfe(np.eye(16), a_info, np.ones(len(coord_panels)),
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
        beta_hat_final = np.zeros(16, dtype=float)
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
            grouped: dict[int, list[np.ndarray]] = {}
            for panel, obs_row in all_obs:
                grouped.setdefault(coord_panels.index(panel), []).append(np.asarray(obs_row))
            for pidx, rows_list in grouped.items():
                obs_batch = np.asarray(rows_list)
                cm = empirical_kernel_cm(ref_val, phi_ref, obs_batch, coord_panels[pidx])
                U += (cm - mu_beta_j).sum(0)
                info_panel = policy_infos[pidx] if policy_infos is not None else emp_infos[pidx]
                H += len(obs_batch) * info_panel
            step = np.linalg.solve(H + 1e-2 * np.eye(16), U)
            if np.linalg.norm(step) > 2.0:
                step = step * (2.0 / np.linalg.norm(step))
            beta_hat_final = np.clip(beta_hat_final + step, -4.0, 4.0)
        logits = pool_phi @ beta_hat_final
        wgt = np.exp(logits - logits.max())
        ess = float(wgt.sum() ** 2 / np.sum(wgt ** 2) / len(pool_phi))
        # M_hat(p) = sum_S p_S I_S: empirical kernel infos for non-generator
        # policies, learned-generator infos for RR-GID.
        info_for_M = policy_infos if policy_infos is not None else emp_infos
        M_hat = np.tensordot(probs, info_for_M, axes=(0, 0))
        lambda_min_M = float(np.linalg.eigvalsh((np.asarray(M_hat) + np.asarray(M_hat).T) / 2).min())
        gen_samples, uncond_acceptance, gen_ess = gen.tilted_full_diagnostics(
            beta_hat_final, cfg["c2st_samples"], seed + 33)
        mean_rmse, std_rmse = heldout_moment_rmse(gen_samples, test_x, mean, std, pcs2)
        auc = c2st_auc(gen_samples, test_x, rng_seed=seed % 10000)
        diagnostic_obs = [obs for _, obs in pilot_obs[:8]]
        diagnostic_panels = [panel for panel, _ in pilot_obs[:8]]
        cond_acceptance, cond_ess = conditional_acceptance_diagnostic(
            gen, beta_hat_final, diagnostic_obs, diagnostic_panels, n=8, seed=seed + 100)
        projection_loss = bregman_projection_loss(full_pool_phi, beta_hat_final, beta_dag)
        print(f"[rep {seed} t+{_t.time()-t0:.1f}s policy {name} done", flush=True)
        rows.append({"policy": name, "budget": budget, "seed": seed,
                     "projection_loss": projection_loss, "heldout_mean_rmse": mean_rmse,
                     "heldout_std_rmse": std_rmse, "c2st_auc": auc, "ess": ess,
                     "conditional_acceptance": cond_acceptance,
                     "unconditional_acceptance": uncond_acceptance,
                     "generator_ess": gen_ess, "conditional_ess": cond_ess,
                     "fw_gap": fw_gap, "lambda_min_M": lambda_min_M,
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
    pcs2 = fit_pc2(ref_train, mean, std)
    sensor_pairs = panel_library()
    coord_panels = tuple(tuple(j for s in pair for j in range(s * 8, (s + 1) * 8)) for pair in sensor_pairs)
    gas_fn = lambda x: transform_features(x, mean, std, pcs)
    model, ckpt = load_vaeac_checkpoint(cfg["generator_ckpt"], device="cuda", expected_dim=128)
    gen = VAEACGenerator(model, ckpt.get("scale", np.ones(128)), alpha=0.0,
                         device="cuda", feature_fn=gas_fn)
    # frozen large full-sample pool for A_hat (PDF 8.3)
    full_pool_x = gen.sample_full(cfg["full_pool_size"], seed=42)
    # empirical kernel panel infos + A-OSQD infos, computed once on ref pool
    phi_ref = gas_fn(ref_val)
    emp_infos = empirical_panel_infos(ref_val, phi_ref, coord_panels)
    gas_ao_infos = gas_a_optimal_information(phi_ref, sensor_pairs)

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
                    rows = run_r2_replication(cfg, ref_train, ref_val, mean, std, pcs, pcs2, gen,
                                              campaign_x, campaign_phi, coord_panels, sensor_pairs,
                                              budget, seed, full_pool_x, mlp_steps=args.mlp_steps,
                                              emp_infos=emp_infos, gas_ao_infos=gas_ao_infos,
                                              phi_ref=phi_ref)
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
