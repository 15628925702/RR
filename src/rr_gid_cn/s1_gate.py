"""Synthetic S1 oracle-gate evaluation with panel-specific exact conditional scores."""

from __future__ import annotations

import numpy as np

from .policies import frank_wolfe, uniform_probabilities
from .discriminative import MaskedScoreMLP, masked_input, masked_pool, score_information
from .synthetic_oracle import beta_direction_and_scale, feature_map, full_target_kl, log_partition, sample_conditional, sample_conditional_batch, sample_full, tilted_conditional_sample, tilted_conditional_batch, tilted_full_sample, tilted_moments, tilted_sample_from_reference


def exact_panel_information(mixture, beta, panels, reference_pool, scale, n_tilted=256, n_conditional=64, seed=0,
                            feature_fn=None):
    rng = np.random.default_rng(seed)
    fn = (lambda x: feature_map(x, scale)) if feature_fn is None else feature_fn
    tilted = tilted_full_sample(mixture, beta, n_tilted, int(rng.integers(2**31 - 1)), scale, feature_fn=fn)
    phi = fn(tilted)
    mu = phi.mean(0)
    fisher = np.cov(phi, rowvar=False)
    infos = []
    for panel in panels:
        observed = tilted[:, list(panel)]
        projected_a = fn(tilted_conditional_batch(mixture, beta, observed, panel, n_conditional, int(rng.integers(2**31 - 1)), scale, feature_fn=fn)).mean(axis=1) - mu
        projected_b = fn(tilted_conditional_batch(mixture, beta, observed, panel, n_conditional, int(rng.integers(2**31 - 1)), scale, feature_fn=fn)).mean(axis=1) - mu
        a = np.asarray(projected_a); b = np.asarray(projected_b)
        a = a - a.mean(0); b = b - b.mean(0)
        info = (a.T @ b + b.T @ a) / max(2 * (len(a) - 1), 1)
        vals, vecs = np.linalg.eigh((info + info.T) / 2)
        infos.append((vecs * np.maximum(vals, 1e-10)) @ vecs.T)
    return fisher, np.asarray(infos)


def a_optimal_information(reference: np.ndarray, panels: tuple[tuple[int, int], ...], dimension: int = 16) -> np.ndarray:
    """Panel Fisher information used by the A-optimal split-questionnaire baseline.

    For a multivariate-normal mean parameter, observing ``X_S`` carries the
    ``(S, S)`` block of the full precision matrix (complete-case information).
    Each panel contributes a ``dimension x dimension`` matrix that is zero outside
    its ``(S, S)`` block; the A-optimal solver then minimizes ``tr(M_A^-1)`` over
    the shared cost-aware ``M_A(p) = sum_S p_S/c(S) I_S``.  This is the PDF's
    "complete reference covariance" formulation, with a ridge on the block
    precision for finite-sample stability.
    """
    x = np.asarray(reference, dtype=float)
    full_cov = np.cov(x, rowvar=False)
    inv_full = np.linalg.inv(full_cov + 1e-6 * np.eye(dimension))
    infos = []
    for panel in panels:
        idx = list(panel)
        info = np.zeros((dimension, dimension))
        info[np.ix_(idx, idx)] = inv_full[np.ix_(idx, idx)]
        infos.append(info)
    return np.asarray(infos)


def policy_designs(reference, panels, fisher, oracle_information, fw_tolerance=1e-6):
    costs = np.ones(len(panels))
    uniform = uniform_probabilities(len(panels))
    a_info = a_optimal_information(reference, panels)
    a_p, _, _ = frank_wolfe(np.eye(16), a_info, costs, uniform, tolerance=1e-4, max_iter=500)
    rr_p, _, _ = frank_wolfe(fisher, oracle_information, costs, uniform, tolerance=fw_tolerance, max_iter=300)
    return {"Uniform SQD": uniform, "A-OSQD": a_p, "oracle RR-GID": rr_p}


def discriminative_design(reference_train, validation, beta, panels, scale, seed=0,
                          hidden=64, steps=200, lr=0.01, fw_tolerance=1e-4, feature_fn=None):
    """PDF P5 Discriminative Score OED design.

    Trains a mask-conditioned MLP on tilt-weighted score labels
    ``s_beta(X) = phi(X) - mu_beta`` with each reference row receiving a uniform
    random mask drawn from the candidate panel family (PDF Sec. 6), then
    estimates ``I_hat_S = Cov_w(g(X_S))`` on an independent validation set with
    tilt weights ``w prop exp(beta^T phi)``. The cost-aware Frank-Wolfe solver
    minimizes ``tr(F_hat M(p)^-1)`` with the tilted reference Fisher ``F_hat``.
    """
    rng = np.random.default_rng(seed)
    fn = (lambda x: feature_map(x, scale)) if feature_fn is None else feature_fn
    n = len(reference_train)
    dim = reference_train.shape[-1]
    r = fn(reference_train[:1]).shape[-1]
    mask_matrix = np.zeros((n, dim), dtype=float)
    for i in range(n):
        mask_matrix[i, list(panels[int(rng.integers(len(panels)))])] = 1.0
    phi_train = fn(reference_train)
    mu_beta, _ = tilted_moments(beta, reference_train, scale, feature_fn=fn)
    labels = phi_train - mu_beta
    logits = phi_train @ beta
    w = np.exp(logits - logits.max())
    model = MaskedScoreMLP(2 * dim, r, hidden=hidden, seed=seed)
    model.fit(masked_pool(reference_train, mask_matrix), labels, weights=w,
              steps=steps, lr=lr)
    phi_val = fn(validation)
    w_val = np.exp(phi_val @ beta)
    infos = score_information(model, validation, panels, w_val)
    _, fisher_hat = tilted_moments(beta, validation, scale, feature_fn=fn)
    costs = np.ones(len(panels))
    uniform = uniform_probabilities(len(panels))
    p, _, _ = frank_wolfe(fisher_hat, infos, costs, uniform, tolerance=fw_tolerance, max_iter=300)
    return p


def balanced_pilot_counts(panels: tuple[tuple[int, int], ...], budget: int) -> np.ndarray:
    """PDF balanced pilot: equal mass on the six direct feature-support pairs."""
    counts = np.zeros(len(panels), dtype=int)
    supports = [(i, i + 6) for i in range(6)]
    indices = [panels.index(s) for s in supports if s in panels]
    if not indices:
        return np.floor(budget * uniform_probabilities(len(panels))).astype(int)
    base, rem = divmod(budget, len(indices))
    counts[indices] = base
    for idx in indices[:rem]:
        counts[idx] += 1
    return counts


def pilot_ht_moment(observations, pilot_counts, panels, scale, reference):
    n0 = int(pilot_counts.sum())
    if n0 <= 0:
        return np.zeros(12), np.zeros(12)
    supports = [(i,) for i in range(6)] + [(i, i + 6) for i in range(6)]
    values = np.zeros(12)
    for a, support in enumerate(supports):
        rho = sum(c for c, panel in zip(pilot_counts, panels) if set(support).issubset(panel)) / n0
        if rho <= 0:
            continue
        vals = []
        for panel, observed in observations:
            if set(support).issubset(panel):
                full = np.zeros(16); full[list(panel)] = observed
                vals.append(feature_map(full[None, :], scale)[0, a])
        values[a] = np.sum(vals) / (n0 * rho) if vals else 0.0
    return values, np.asarray([sum(c for c, panel in zip(pilot_counts, panels) if set(s).issubset(panel)) / n0 for s in supports])


def solve_pilot_beta(mu_pil, reference, scale, theta_bound=4.0, steps=20, norm_cap=None):
    beta = np.zeros(12)
    features = feature_map(reference, scale)
    target = np.asarray(mu_pil)
    def objective(x):
        z = features @ x
        return float(np.logaddexp.reduce(z) - np.log(len(z)) - x @ target)
    for _ in range(max(steps, 100)):
        mu, fisher = tilted_moments(beta, reference, scale)
        direction = np.linalg.pinv(fisher, rcond=1e-10) @ (target - mu)
        current = objective(beta)
        step = 1.0
        while step > 1e-5:
            candidate = np.clip(beta + step * direction, -theta_bound, theta_bound)
            if objective(candidate) <= current + 1e-9:
                break
            step *= 0.5
        if step <= 1e-5 or np.linalg.norm(candidate - beta) < 1e-7:
            break
        beta = candidate
    return beta


def prepare_s1_oracle(mixture, scale, panels, seed=2026, reference_size=50000, information_samples=256, conditional_samples=32, large_reference_size=200000,
                      feature_fn=None):
    reference = sample_full(mixture, reference_size, seed)
    reference_large = sample_full(mixture, large_reference_size, seed + 12345)
    beta_true = beta_direction_and_scale(reference, 2026, 0.5, scale, feature_fn=feature_fn)
    fisher, oracle_information = exact_panel_information(mixture, beta_true, panels, reference, scale, information_samples, conditional_samples, seed + 1, feature_fn=feature_fn)
    designs = policy_designs(reference, panels, fisher, oracle_information)
    return {"reference": reference, "reference_large": reference_large, "beta_true": beta_true, "fisher": fisher, "information": oracle_information, "designs": designs}


def imp_conditional_mean(mixture, beta, batch, panel, n, seed, scale, feature_fn=None):
    """E_{Q_beta}[phi(X)|X_S] by importance weighting on exact Q0 conditional samples.

    Vectorized: returns (n_rows, r).  This replaces accept-reject tilting of the
    conditional law for the S1 exact-conditional-oracle gate (PDF Sec. 4.1 allows
    importance proposals from Q0; cross-completion removes the self-normalized bias).
    """
    rng = np.random.default_rng(seed)
    fn = (lambda x: feature_map(x, scale)) if feature_fn is None else feature_fn
    completions = sample_conditional_batch(mixture, batch, panel, n, int(rng.integers(2**31 - 1)))
    phi = fn(completions)
    w = np.exp(phi @ np.asarray(beta))
    return np.einsum("onr,on->or", phi, w) / w.sum(axis=1, keepdims=True)


def panel_information_cross(mixture, beta, panels, reference, scale, n_tilted, n_cond, seed, feature_fn=None):
    """Cross-completion panel information estimator (PDF Eq. 9) with PSD projection.

    Uses importance-weighted conditional completions (not accept-reject), so it is
    cheap enough to re-estimate ``I_S(beta^(j))`` at every scoring step as PDF
    Algorithm 2 step 6 requires.  ``reference`` must be a large Q0 pool so the
    tilted ``mu`` estimate has negligible variance.
    """
    rng = np.random.default_rng(seed)
    fn = (lambda x: feature_map(x, scale)) if feature_fn is None else feature_fn
    features = fn(reference)
    logits = features @ np.asarray(beta)
    w = np.exp(logits - logits.max())
    w /= w.sum()
    idx = rng.choice(len(reference), size=n_tilted, p=w)
    tilted = reference[idx]
    mu = features[idx].mean(0)
    infos = []
    for panel in panels:
        observed = tilted[:, list(panel)]
        ca = tilted_conditional_batch(mixture, beta, observed, panel, n_cond, seed + 1, scale, feature_fn=fn)
        cb = tilted_conditional_batch(mixture, beta, observed, panel, n_cond, seed + 2, scale, feature_fn=fn)
        a = fn(ca).mean(axis=1) - mu
        b = fn(cb).mean(axis=1) - mu
        a = a - a.mean(0)
        b = b - b.mean(0)
        info_hat = (a.T @ b + b.T @ a) / max(2 * (len(a) - 1), 1)
        vals, vecs = np.linalg.eigh((info_hat + info_hat.T) / 2)
        infos.append((vecs * np.maximum(vals, 1e-10)) @ vecs.T)
    return np.asarray(infos)


def final_rr_estimator(mixture, beta_start, observations, panels, reference, scale, lu, seed,
                       theta_bound=4.0, h_tilted=128, h_cond=32, step_size=1.0, norm_cap=None,
                       l1_cap=None, mu_samples=10000, mu_direct=False, oracle_information=None,
                       feature_fn=None, max_step_norm=None):
    """One Fisher-scoring update following PDF Algorithm 2 step 6.

    Re-estimates ``I_S(beta^(j))`` by cross-completion at the current beta, builds
    ``U`` from exact-conditional-oracle observed scores (importance-weighted), and
    takes the projected full step ``beta + H^{-1} U``, projected back onto Theta
    (the PDF leaves Theta's numeric boundary open; the norm cap keeps tilt overlap
    bounded for extreme baseline allocations).
    """
    rng = np.random.default_rng(seed)
    fn = (lambda x: feature_map(x, scale)) if feature_fn is None else feature_fn
    grouped = {}
    for panel, obs in observations:
        grouped.setdefault(panel, []).append(obs)
    infos = panel_information_cross(mixture, beta_start, panels, reference, scale, h_tilted, h_cond, seed + 7, feature_fn=fn) if oracle_information is None else oracle_information
    H = np.zeros((beta_start.shape[0], beta_start.shape[0]))
    projected = []
    for panel, rows in grouped.items():
        H += len(rows) * infos[panels.index(panel)]
        # S1 uses the exact conditional oracle for the observed score. The
        # finite-LU importance proposal is reserved for learned-generator
        # experiments, not the oracle gate.
        completions = tilted_conditional_batch(mixture, beta_start, np.asarray(rows), panel, lu,
                                                int(rng.integers(2**31 - 1)), scale, feature_fn=fn)
        projected.append(fn(completions).mean(axis=1))
    projected = np.concatenate(projected, axis=0)
    if mu_direct:
        mu_beta = fn(tilted_full_sample(mixture, beta_start, mu_samples, seed + 99, scale, feature_fn=fn)).mean(0)
    else:
        mu_beta, _ = tilted_moments(beta_start, reference, scale, feature_fn=fn)
    U = np.sum(projected - mu_beta, axis=0)
    H = (H + H.T) / 2
    step = np.linalg.pinv(H, rcond=1e-10) @ U
    # Algorithm 2 specifies a full Fisher-scoring step.  A finite Monte Carlo
    # information estimate can occasionally produce a very large Newton step
    # when one eigen-direction of H is weak.  Use a trust-region only as a
    # numerical safeguard; the bound is inactive once the pilot is in the
    # local asymptotic regime, so consistency is retained (unlike a fixed
    # damping factor, which leaves an O(b_B^-1/2) pilot bias at every B).
    scaled_step = float(step_size) * step
    if max_step_norm is not None:
        step_norm = float(np.linalg.norm(scaled_step))
        if step_norm > float(max_step_norm) > 0.0:
            scaled_step *= float(max_step_norm) / step_norm
    updated = np.asarray(beta_start) + scaled_step
    updated = np.clip(updated, -theta_bound, theta_bound)
    if norm_cap is not None:
        norm = float(np.linalg.norm(updated))
        if norm > float(norm_cap):
            updated = updated * (float(norm_cap) / norm)
    if l1_cap is not None:
        l1_norm = float(np.abs(updated).sum())
        if l1_norm > float(l1_cap):
            updated = updated * (float(l1_cap) / l1_norm)
    return updated


def run_replication(mixture, scale, panels, budget, seed, prepared=None,
                    lu=128, h_tilted=128, h_cond=32, pilot_norm_cap=2.0, kl_samples=20000,
                    scoring_steps=2, theta_norm_cap=None, policies=None, mu_direct=False, mu_samples=10000,
                    kl_mu_direct=True, use_oracle_H=False, validation_size=10000,
                    mlp_hidden=64, mlp_steps=200, generator=None,
                    gen_info_tilted=256, gen_info_cond=32, disc_reference_size=None,
                    feature_fn=None, scoring_step_size=1.0, theta_l1_cap=None,
                    scoring_max_step_norm=None):
    prepared = prepared or prepare_s1_oracle(mixture, scale, panels, seed, feature_fn=feature_fn)
    reference = prepared["reference"]
    ref_large = prepared["reference_large"]
    beta_true = prepared["beta_true"]
    fisher = prepared["fisher"]
    oracle_information = prepared["information"]
    designs = prepared["designs"]
    fn = (lambda x: feature_map(x, scale)) if feature_fn is None else feature_fn
    target_reference = ref_large
    target_full = tilted_full_sample(mixture, beta_true, budget, seed + 3, scale, feature_fn=fn)
    rows = []
    pilot_budget = int(np.ceil(10.0 * budget ** (1.0 / 3.0)))
    pilot_budget = min(pilot_budget, budget)
    remaining_budget = budget - pilot_budget
    rr_phi = float(np.trace(fisher @ np.linalg.pinv(np.tensordot(designs["oracle RR-GID"], oracle_information, axes=(0, 0)))))
    if kl_mu_direct:
        mu_bt = fn(tilted_full_sample(mixture, beta_true, kl_samples, seed + 90, scale, feature_fn=fn)).mean(0)
    else:
        mu_bt = tilted_moments(beta_true, target_reference, scale, feature_fn=fn)[0]
    policy_names = policies if policies is not None else list(designs)
    for name in policy_names:
        pilot_counts = balanced_pilot_counts(panels, pilot_budget)
        pilot_observations = []
        pilot_cursor = 0
        for panel, count in zip(panels, pilot_counts):
            for row in target_full[pilot_cursor : pilot_cursor + count]:
                pilot_observations.append((panel, row[list(panel)]))
            pilot_cursor += count
        pilot_mu, pilot_rho = pilot_ht_moment(pilot_observations, pilot_counts, panels, scale, reference)
        beta_hat = solve_pilot_beta(pilot_mu, reference, scale, norm_cap=pilot_norm_cap)
        if theta_norm_cap is not None:
            pilot_norm = float(np.linalg.norm(beta_hat))
            if pilot_norm > float(theta_norm_cap):
                beta_hat = beta_hat * (float(theta_norm_cap) / pilot_norm)
        if theta_l1_cap is not None:
            pilot_l1 = float(np.abs(beta_hat).sum())
            if pilot_l1 > float(theta_l1_cap):
                beta_hat = beta_hat * (float(theta_l1_cap) / pilot_l1)
        if name == "Discriminative Score OED":
            # PDF P5: each campaign retrains the mask-conditioned MLP and redesigns.
            validation = sample_full(mixture, validation_size, seed + 555)
            disc_ref = reference if disc_reference_size is None else reference[:disc_reference_size]
            probabilities = discriminative_design(disc_ref, validation, beta_hat, panels, scale,
                                                  seed + 11, hidden=mlp_hidden, steps=mlp_steps,
                                                  feature_fn=feature_fn)
        elif name == "learned RR-GID":
            # PDF P6 generator-aware design: cross-completion I_hat_S from the
            # frozen VAEAC generator, cost-aware Frank-Wolfe design.
            from .vaeac import learned_information
            fisher_hat, infos = learned_information(generator, beta_hat, panels,
                                                    n_tilted=gen_info_tilted,
                                                    n_conditional=gen_info_cond, seed=seed + 17)
            costs = np.ones(len(panels))
            probabilities, _, _ = frank_wolfe(fisher_hat, infos, costs, uniform_probabilities(len(panels)),
                                              tolerance=1e-4, max_iter=300)
        else:
            probabilities = designs[name]
        expected = remaining_budget * probabilities
        counts = np.floor(expected).astype(int)
        # Largest-remainder apportionment exactly honors the frozen acquisition
        # budget without changing the continuous policy design.
        remainder = remaining_budget - int(counts.sum())
        if remainder:
            counts[np.argsort(expected - counts)[-remainder:]] += 1
        if int(counts.sum()) != remaining_budget:
            raise RuntimeError("main panel allocation does not exhaust the remaining acquisition budget")
        main_observations = []
        main_cursor = pilot_budget
        for panel, count in zip(panels, counts):
            for row in target_full[main_cursor : main_cursor + count]:
                main_observations.append((panel, row[list(panel)]))
            main_cursor += count
        observations = pilot_observations + main_observations
        update_diagnostics = [{"step": "pilot", "pilot_budget": int(pilot_counts.sum()), "beta_norm": float(np.linalg.norm(beta_hat)), "rho_min": float(np.min(pilot_rho[pilot_rho > 0])) if np.any(pilot_rho > 0) else 0.0}]
        # PDF Algorithm 2 projects every Fisher-scoring iterate onto the
        # compact parameter set Theta.  theta_norm_cap is the configured
        # Euclidean radius for that set; previously the argument existed but
        # was silently ignored, allowing overlap collapse in exact TiltCond.
        norm_cap_val = theta_norm_cap
        for update in range(scoring_steps):
            beta_next = final_rr_estimator(mixture, beta_hat, observations, panels, ref_large, scale, lu, seed + 4 + update, h_tilted=h_tilted, h_cond=h_cond, step_size=scoring_step_size, norm_cap=norm_cap_val, l1_cap=theta_l1_cap, mu_direct=mu_direct, mu_samples=mu_samples, oracle_information=oracle_information if use_oracle_H else None, feature_fn=feature_fn, max_step_norm=scoring_max_step_norm)
            update_diagnostics.append({"step": update, "step_norm": float(np.linalg.norm(beta_next - beta_hat)), "projected": bool(np.any(np.abs(beta_next) >= 4.0)), "pilot_budget": int(pilot_counts.sum())})
            beta_hat = beta_next
        kl = max(0.0, float((beta_true - beta_hat) @ mu_bt - log_partition(beta_true, target_reference, scale, feature_fn=fn) + log_partition(beta_hat, target_reference, scale, feature_fn=fn)))
        rows.append({"policy": name, "budget": budget, "allocated_observations": int(counts.sum() + pilot_counts.sum()), "pilot_budget": int(pilot_counts.sum()), "seed": seed, "beta_true_norm": float(np.linalg.norm(beta_true)), "beta_hat_norm": float(np.linalg.norm(beta_hat)), "kl": kl, "B_kl": budget * kl, "design_ratio": float(kl / max(rr_phi / (2 * budget), 1e-12)), "target_draw_seed": seed + 3, "beta_hat": beta_hat.tolist(), "update_diagnostics": update_diagnostics})
    return rows
