"""P9 R1 formal: Gas well-specified semi-synthetic budget curves (PDF 8.2).

Builds an empirical base distribution from the reference-validation records,
resamples a target under ``w_i(beta*) ~ exp(beta*^T phi(x_i))`` with ESS/N ~ 0.5,
and runs the four acquisition policies over B in {400, 800, 1600, 3200} with the
frozen 128-dim Gas VAEAC generator providing Q0 full/conditional sampling and
the empirical base providing the exact-ish ground-truth target law for KL.

Per PDF 8.2: ``b_B = min(0.2 B, ceil(10 B^(1/3)))``, ``J = 2``, four policies
share the same target draws, and the final RR estimator uses the empirical base
for the KL risk.  Discriminative Score OED retrains per replication.
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
from rr_gid_cn.vaeac import (VAEACGenerator, imp_conditional_mean_from_generator,
                             learned_information, load_vaeac_checkpoint)


def gas_a_optimal_information(phi_ref, sensor_pairs):
    """A-OSQD panel Fisher info in the 16-dim phi space (PDF Sec. 6 / 8.1).

    Each sensor-pair panel (a, b) observes the phi coordinates it spans: the two
    unary coordinates ``{a, b}`` and the two pairwise coordinates ``{8+a, 8+b}``
    (when those sensors exist).  Uses the complete-reference phi covariance.
    """
    x = np.asarray(phi_ref, dtype=float)
    dim = x.shape[1]
    full_cov = np.cov(x, rowvar=False)
    inv_full = np.linalg.inv(full_cov + 1e-6 * np.eye(dim))
    infos = []
    for a, b in sensor_pairs:
        idx = [a, b]
        if a + 8 < dim:
            idx.append(a + 8)
        if b + 8 < dim:
            idx.append(b + 8)
        idx = sorted(set(idx))
        info = np.zeros((dim, dim))
        info[np.ix_(idx, idx)] = inv_full[np.ix_(idx, idx)]
        infos.append(info)
    return np.asarray(infos)


def gas_balanced_pilot_counts(coord_panels, budget):
    """PDF 8.1 Gas balanced pilot: equal mass on the 8 direct (j, j+8) pairs."""
    counts = np.zeros(len(coord_panels), dtype=int)
    sensor_pilots = [(j, j + 8) for j in range(8)]
    indices = []
    for sp in sensor_pilots:
        cpanel = tuple(j for s in sp for j in range(s * 8, (s + 1) * 8))
        if cpanel in coord_panels:
            indices.append(coord_panels.index(cpanel))
    if not indices:
        return np.floor(budget * uniform_probabilities(len(coord_panels))).astype(int)
    base, rem = divmod(budget, len(indices))
    counts[indices] = base
    for idx in indices[:rem]:
        counts[idx] += 1
    return counts


def calibrate_beta(phi, seed=2026, target_ess=0.5):
    """ESS/N bisection for the empirical tilt magnitude (PDF 8.2, ESS/N ~ 0.5)."""
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


def empirical_full_kl(beta_true, beta_est, ref_x, phi):
    """KL(Q_beta* || Q_beta_hat) relative to the empirical base (exact sum).

    ``KL = E_{Q_beta*}[log dQ_beta*/dQ_beta_hat]`` with both laws normalized
    against the same empirical Q0 reference pool (PDF 8.2's full-law KL).
    """
    log_weights_true = phi @ beta_true
    log_weights_est = phi @ beta_est
    log_true = log_weights_true - np.logaddexp.reduce(log_weights_true)
    log_est = log_weights_est - np.logaddexp.reduce(log_weights_est)
    return float(np.sum(np.exp(log_true) * (log_true - log_est)))


def empirical_conditional_mean(phi, ref_pool, observed, panel, k=60):
    """Kernel-weighted E_{Q0}[phi(X) | X_S] from the empirical reference pool.

    ``ref_pool`` is the ``(n_ref, 128)`` reference-record array; ``observed`` is
    ``(n, |panel|)``.  Nearest neighbours are found in panel-coordinate space and
    the pooled phi rows are kernel-averaged.  Vectorized over the observed batch.
    """
    idx = list(panel)
    pool = ref_pool[:, idx]
    Xp = np.atleast_2d(np.asarray(observed))
    # pairwise squared distances: (n, n_ref)
    d2 = (np.sum(Xp ** 2, axis=1)[:, None] + np.sum(pool ** 2, axis=1)[None, :]
          - 2.0 * (Xp @ pool.T))
    nn_idx = np.argpartition(d2, k, axis=1)[:, :k]
    # per-row bandwidth: median of the k nearest distances
    d2_nn = np.take_along_axis(d2, nn_idx, axis=1)
    band = np.median(d2_nn, axis=1)
    wk = np.exp(-d2_nn / (2 * band[:, None] ** 2 + 1e-8))
    wk /= wk.sum(axis=1, keepdims=True)
    # weighted average of phi over the k nearest neighbours: (n, r)
    # phi[nn_idx] is (n, k, r); multiply by wk[:, :, None] and sum over k.
    out = np.sum(wk[:, :, None] * phi[nn_idx], axis=1)
    return out


def gas_pilot_ht_moment(phi_val, pilot_obs, pilot_counts, coord_panels):
    """Horvitz-Thompson estimate of E_{Q_beta}[phi] from panel observations.

    Every Gas feature ``phi_j = tanh(z_j)`` (j<8) has support ``{j}`` and every
    pairwise feature ``phi_{8+j} = tanh(z_j z_{j+8})`` (j<8) has support
    ``{j, j+8}`` in *feature* coordinates.  A sensor-pair panel ``(a, b)``
    observes feature block ``{a*8..(a+1)*8} ∪ {b*8..(b+1)*8}``; it covers unary
    feature ``j`` iff ``j in {a, b}`` and pairwise feature ``8+j`` iff both
    ``j`` and ``j+8`` are in ``{a, b}``.  The HT estimator re-weights each
    observed feature value by its panel inclusion probability.
    """
    n0 = int(pilot_counts.sum())
    if n0 <= 0:
        return np.zeros(16), np.zeros(16)
    r = 16
    values = np.zeros(r)
    rho = np.zeros(r)
    # panel is a tuple of feature indices; a unary feature a (a in 0..7) is
    # covered iff sensor a is in the panel's sensor set.
    sensor_of = {j: j // 8 for j in range(128)}
    panel_sensors = [set(sensor_of[j] for j in panel) for panel in coord_panels]
    for a in range(8):
        rho[a] = sum(c for c, ps in zip(pilot_counts, panel_sensors) if a in ps) / n0 if n0 else 0.0
    for a in range(8):  # pairwise features phi_{8+a} = tanh(z_a z_{a+8})
        rho[8 + a] = sum(c for c, ps in zip(pilot_counts, panel_sensors)
                         if {a, a + 8}.issubset(ps)) / n0 if n0 else 0.0
    # collect observed feature values
    obs_by_feature = {a: [] for a in range(r)}
    for panel, observed in pilot_obs:
        ps = set(sensor_of[j] for j in panel)
        full = np.zeros(128); full[list(panel)] = observed
        phi_row = gas_transform(full[None, :])[0]
        for a in range(8):
            if a in ps:
                obs_by_feature[a].append(phi_row[a])
        for a in range(8):
            if {a, a + 8}.issubset(ps):
                obs_by_feature[8 + a].append(phi_row[8 + a])
    for a in range(r):
        if rho[a] > 0 and obs_by_feature[a]:
            values[a] = float(np.sum(obs_by_feature[a])) / (n0 * rho[a])
    return values, rho


def solve_pilot_beta_empirical(mu_pil, phi_val, steps=30, theta_bound=4.0, norm_cap=2.0):
    """Moment-matching beta for the empirical exponential family (minimizes A-P).

    ``norm_cap`` keeps the tilt overlap bounded so the generator accept-reject
    sampler in ``learned_information`` does not collapse (PDF 7.1 flags low
    overlap runs explicitly).
    """
    beta = np.zeros(16)
    features = phi_val
    target = np.asarray(mu_pil, dtype=float)
    for _ in range(max(steps, 100)):
        z = features @ beta
        log_w = z - z.max()
        w = np.exp(log_w); w /= w.sum()
        mu = w @ features
        diff = target - mu
        if np.linalg.norm(diff) < 1e-8:
            break
        fisher = np.cov(features, rowvar=False, aweights=w) + 1e-3 * np.eye(16)
        direction = np.linalg.solve(fisher, diff)
        step = 1.0
        while step > 1e-5:
            candidate = np.clip(beta + step * direction, -theta_bound, theta_bound)
            if norm_cap is not None:
                nrm = np.linalg.norm(candidate)
                if nrm > norm_cap:
                    candidate = candidate * (norm_cap / nrm)
            zc = features @ candidate
            lc = zc - zc.max()
            A_c = np.log(np.exp(lc).sum()) - np.log(len(features))
            A_b = np.log(np.exp(z - z.max()).sum()) - np.log(len(features))
            if A_c - candidate @ target <= A_b - beta @ target + 1e-9:
                break
            step *= 0.5
        if step <= 1e-5 or np.linalg.norm(candidate - beta) < 1e-8:
            break
        beta = candidate
    return beta


def run_gas_replication(cfg, ref_train, ref_val, mean, std, pcs, gen, budget, seed,
                        coord_panels, sensor_pairs, mlp_steps=200, oracle_infos=None, oracle_fisher=None):
    """One R1 replication: four policies on the same target draw."""
    import time as _t
    t0 = _t.time()
    phi_val = gas_transform(ref_val)
    beta_true = calibrate_beta(phi_val, seed=2026)
    logits = phi_val @ beta_true
    w = np.exp(logits - logits.max()); w /= w.sum()
    rng = np.random.default_rng(seed)
    target_idx = rng.choice(len(ref_val), size=budget, p=w)
    target = ref_val[target_idx]
    b_pilot = min(int(np.ceil(0.2 * budget)), int(np.ceil(10.0 * budget ** (1.0 / 3.0))))
    b_pilot = min(b_pilot, budget)
    fisher = oracle_fisher if oracle_fisher is not None else np.cov(phi_val, rowvar=False)
    infos = oracle_infos
    p_star, _, _ = frank_wolfe(fisher, infos, np.ones(len(coord_panels)),
                               uniform_probabilities(len(coord_panels)), tolerance=1e-4, max_iter=300)
    phi_star = float(np.trace(fisher @ np.linalg.pinv(np.tensordot(p_star, infos, axes=(0, 0)))))
    print(f"[rep {seed} t+{_t.time()-t0:.1f}s oracle ready]", flush=True)

    # --- policies
    pilot_counts = gas_balanced_pilot_counts(coord_panels, b_pilot)
    pilot_obs = []
    cursor = 0
    for panel, count in zip(coord_panels, pilot_counts):
        for row in target[cursor:cursor + count]:
            pilot_obs.append((panel, row[list(panel)]))
        cursor += count
    # pilot beta: empirical HT moment + moment-matching solve
    mu_pil, _ = gas_pilot_ht_moment(phi_val, pilot_obs, pilot_counts, coord_panels)
    beta_hat = solve_pilot_beta_empirical(mu_pil, phi_val)
    print(f"[rep {seed} t+{_t.time()-t0:.1f}s pilot beta done]", flush=True)

    rows = []
    policy_names = ["Uniform SQD", "A-OSQD", "Discriminative Score OED", "RR-GID"]
    for name in policy_names:
        t_pol = _t.time()
        if name == "Uniform SQD":
            probs = uniform_probabilities(len(coord_panels))
        elif name == "A-OSQD":
            a_info = gas_a_optimal_information(phi_val, sensor_pairs)
            probs, _, _ = frank_wolfe(np.eye(16), a_info, np.ones(len(coord_panels)),
                                      uniform_probabilities(len(coord_panels)), tolerance=1e-4, max_iter=300)
        elif name == "Discriminative Score OED":
            validation = ref_val[:cfg["validation_size"]]
            probs = discriminative_design(ref_train, validation, beta_hat, coord_panels, np.ones(128),
                                          seed=seed + 11, hidden=64, steps=mlp_steps, feature_fn=gas_transform)
        else:  # RR-GID
            _, learned_infos = learned_information(gen, beta_hat, coord_panels,
                                                   n_tilted=cfg["gen_info_tilted"],
                                                   n_conditional=cfg["gen_info_cond"], seed=seed + 17,
                                                   feature_fn=gas_transform)
            probs, _, _ = frank_wolfe(fisher, learned_infos, np.ones(len(coord_panels)),
                                      uniform_probabilities(len(coord_panels)), tolerance=1e-4, max_iter=300)
        # main allocation (largest remainder, honoring budget - pilot)
        rem = budget - b_pilot
        counts = np.floor(rem * probs).astype(int)
        remainder = rem - int(counts.sum())
        if remainder:
            counts[np.argsort(rem * probs - counts)[-remainder:]] += 1
        # main + pilot observations
        main_obs = []
        cursor = b_pilot
        for panel, count in zip(coord_panels, counts):
            for row in target[cursor:cursor + count]:
                main_obs.append((panel, row[list(panel)]))
            cursor += count
        all_obs = pilot_obs + main_obs
        # final beta via J=2 Fisher scoring on the empirical base (PDF Alg.2 step 6)
        beta_hat_final = beta_hat.copy()
        for _ in range(2):
            # tilted mean mu_beta^(j) on the empirical base pool
            logits = phi_val @ beta_hat_final
            wb = np.exp(logits - logits.max()); wb /= wb.sum()
            mu_beta_j = wb @ phi_val
            U = np.zeros(16)
            H = 1e-2 * np.eye(16)
            # group observations by panel so the kernel conditional mean is
            # computed once per panel over a batch instead of per observation.
            grouped_rows: dict[int, list[np.ndarray]] = {}
            for panel, obs_row in all_obs:
                grouped_rows.setdefault(coord_panels.index(panel), []).append(np.asarray(obs_row))
            for pidx, rows_list in grouped_rows.items():
                obs_batch = np.asarray(rows_list)
                # Final RR estimator uses the frozen Gas VAEAC conditional
                # interface.  The empirical kernel remains only the
                # reference/oracle diagnostic and is not a hidden fallback.
                cm = imp_conditional_mean_from_generator(
                    gen, beta_hat_final, obs_batch, coord_panels[pidx],
                    n=32, seed=seed + 7000 + 100 * _ + pidx,
                    feature_fn=gas_transform)
                U += (cm - mu_beta_j).sum(0)
                H += len(obs_batch) * infos[pidx]
            step = np.linalg.solve(H, U)
            # cap the scoring step norm so the update does not diverge to the
            # Theta boundary in one shot (PDF Alg.2 step 6 keeps overlap bounded).
            step_norm = np.linalg.norm(step)
            if step_norm > 2.0:
                step = step * (2.0 / step_norm)
            beta_hat_final = np.clip(beta_hat_final + step, -4.0, 4.0)
        kl = empirical_full_kl(beta_true, beta_hat_final, ref_val, phi_val)
        rows.append({"policy": name, "budget": budget, "seed": seed, "kl": kl,
                     "B_kl": budget * kl, "design_ratio": float(kl / max(phi_star / (2 * budget), 1e-12)),
                     "pilot_budget": int(b_pilot)})
        print(f"[rep {seed} t+{_t.time()-t0:.1f}s policy {name} done "
              f"beta_norm={np.linalg.norm(beta_hat_final):.3f} kl={kl:.4f}", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--config", type=Path, default=Path("configs/p9_formal.yaml"))
    ap.add_argument("--rep-range", type=int, nargs=2, default=None)
    ap.add_argument("--max-replications", type=int, default=None)
    ap.add_argument("--mlp-steps", type=int, default=200)
    ap.add_argument("--out", type=str, default="p9_r1")
    args = ap.parse_args()
    with args.config.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)["p9"]
    budgets = (args.budget,) if args.budget is not None else tuple(cfg["budgets"])
    replications = args.max_replications or int(cfg["replications"])

    data = np.load("data/gas/processed/gas_processed.npz")
    ref_train, ref_val = data["ref_train"], data["ref_val"]
    mean, std, pcs = fit_scaler_pca(ref_train)
    sensor_pairs = panel_library()
    coord_panels = tuple(
        tuple(j for s in pair for j in range(s * 8, (s + 1) * 8)) for pair in sensor_pairs
    )
    gas_fn = lambda x: transform_features(x, mean, std, pcs)
    global gas_transform
    gas_transform = gas_fn
    model, ckpt = load_vaeac_checkpoint(cfg["generator_ckpt"], device="cuda", expected_dim=128)
    gen = VAEACGenerator(model, ckpt.get("scale", np.ones(128)), alpha=0.0,
                         device="cuda", feature_fn=gas_fn)

    # Precompute the empirical-kernel oracle panel information once (PDF 8.2).
    # It is defined on the reference-validation pool, independent of the target
    # draw, so every replication reuses the same ground-truth panel ranking.
    phi_val_pool = gas_fn(ref_val)
    oracle_fisher = np.cov(phi_val_pool, rowvar=False)
    oracle_infos = []
    rng_oracle = np.random.default_rng(1234)
    n_oracle = min(len(ref_val), int(cfg.get("oracle_sub", 600)))
    oracle_idx = rng_oracle.choice(len(ref_val), size=n_oracle, replace=False)
    oracle_ref = ref_val[oracle_idx]
    oracle_phi = phi_val_pool[oracle_idx]
    for panel in coord_panels:
        idx = list(panel)
        obs = oracle_ref[:, idx]
        projected = empirical_conditional_mean(oracle_phi, oracle_ref, obs, panel)
        projected = projected - oracle_phi.mean(0)
        oracle_infos.append(np.cov(projected, rowvar=False))
    oracle_infos = np.asarray(oracle_infos)

    out = Path("results")
    for budget in budgets:
        fp = out / f"{args.out}_{budget}.jsonl"
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
                seed = 40000000 + budget * 1000 + rep
                rows = run_gas_replication(cfg, ref_train, ref_val, mean, std, pcs, gen, budget, seed,
                                           coord_panels, sensor_pairs, mlp_steps=args.mlp_steps,
                                           oracle_infos=oracle_infos, oracle_fisher=oracle_fisher)
                for r in rows:
                    r["replication"] = rep
                    if (rep, r["policy"]) not in done:
                        stream.write(json.dumps(r, sort_keys=True) + "\n")
                        done.add((rep, r["policy"]))
                stream.flush()
                print(json.dumps({"budget": budget, "rep": rep, "policies": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
