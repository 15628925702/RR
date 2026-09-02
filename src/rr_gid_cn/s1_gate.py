"""Synthetic S1 oracle-gate evaluation with panel-specific exact conditional scores."""

from __future__ import annotations

import hashlib
from time import perf_counter

import numpy as np

from .policies import frank_wolfe, objective, uniform_probabilities
from .discriminative import MaskedScoreMLP, masked_input, masked_pool, score_information
from .synthetic_oracle import beta_direction_and_scale, feature_map, full_target_kl, log_partition, reset_workload_counters, sample_conditional, sample_conditional_batch, sample_full, tilted_conditional_sample, tilted_conditional_batch, tilted_conditional_feature_mean_batch, tilted_conditional_mean_qmc, tilted_conditional_mean_exact, tilted_full_sample, tilted_moments, tilted_sample_from_reference, workload_counters, workload_qmc_orders, workload_rate_summary


def _sync_cuda() -> None:
    """Synchronize optional CUDA work before a wall-clock stage boundary."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except (ImportError, RuntimeError):
        pass


def _counter_delta(after, before):
    return {key: int(after.get(key, 0) - before.get(key, 0)) for key in after}


def _runtime_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            index = int(torch.cuda.current_device())
            return f"cuda:{index}:{torch.cuda.get_device_name(index)}"
    except Exception:
        pass
    return "cpu"


def _peak_memory() -> dict[str, float | None]:
    cpu_rss_mb = None
    try:
        import resource
        cpu_rss_mb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    except Exception:
        cpu_rss_mb = None
    gpu_alloc_mb = None
    gpu_reserved_mb = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_alloc_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
            gpu_reserved_mb = float(torch.cuda.max_memory_reserved()) / (1024.0 * 1024.0)
    except Exception:
        pass
    return {
        "peak_cpu_rss_mb": cpu_rss_mb,
        "peak_gpu_alloc_mb": gpu_alloc_mb,
        "peak_gpu_reserved_mb": gpu_reserved_mb,
    }


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
        projected_a = fn(tilted_conditional_batch(mixture, beta, observed, panel, n_conditional, int(rng.integers(2**31 - 1)), scale, feature_fn=feature_fn)).mean(axis=1) - mu
        projected_b = fn(tilted_conditional_batch(mixture, beta, observed, panel, n_conditional, int(rng.integers(2**31 - 1)), scale, feature_fn=feature_fn)).mean(axis=1) - mu
        a = np.asarray(projected_a); b = np.asarray(projected_b)
        a = a - a.mean(0); b = b - b.mean(0)
        info = (a.T @ b + b.T @ a) / max(2 * (len(a) - 1), 1)
        vals, vecs = np.linalg.eigh((info + info.T) / 2)
        infos.append((vecs * np.maximum(vals, 1e-10)) @ vecs.T)
    return fisher, np.asarray(infos)


def a_optimal_information(reference: np.ndarray, panels: tuple[tuple[int, int], ...], dimension: int = 16) -> np.ndarray:
    """Panel Fisher information used by the A-optimal split-questionnaire baseline.

    For a multivariate-normal mean parameter, observing ``X_S`` carries
    ``Sigma_SS^-1`` embedded into the full parameter dimension.
    Each panel contributes a ``dimension x dimension`` matrix that is zero outside
    its ``(S, S)`` block; the A-optimal solver then minimizes ``tr(M_A^-1)`` over
    the shared cost-aware ``M_A(p) = sum_S p_S/c(S) I_S``.  This is the PDF's
    "observed covariance submatrix" formulation, with a ridge on that submatrix
    for finite-sample stability.  It is generally not the corresponding block
    of the full precision matrix.
    """
    x = np.asarray(reference, dtype=float)
    full_cov = np.cov(x, rowvar=False)
    infos = []
    for panel in panels:
        idx = list(panel)
        cov_panel = full_cov[np.ix_(idx, idx)]
        precision_panel = np.linalg.inv(cov_panel + 1e-6 * np.eye(len(idx)))
        info = np.zeros((dimension, dimension))
        info[np.ix_(idx, idx)] = precision_panel
        infos.append(info)
    return np.asarray(infos)


def policy_designs(reference, panels, fisher, oracle_information, fw_tolerance=1e-6,
                   return_certificates=False):
    costs = np.ones(len(panels))
    uniform = uniform_probabilities(len(panels))
    a_info = a_optimal_information(reference, panels)
    a_p, a_gap, a_iterations = frank_wolfe(
        np.eye(16), a_info, costs, uniform, tolerance=1e-4, max_iter=500
    )
    rr_p, rr_gap, rr_iterations = frank_wolfe(
        fisher, oracle_information, costs, uniform,
        tolerance=fw_tolerance, max_iter=300,
    )
    designs = {"Uniform SQD": uniform, "A-OSQD": a_p, "oracle RR-GID": rr_p}
    certificates = {
        "A-OSQD": {
            "fw_gap": float(a_gap),
            "fw_iterations": int(a_iterations),
            "objective": objective(np.eye(16), a_info, a_p, costs),
        },
        "oracle RR-GID": {
            "fw_gap": float(rr_gap),
            "fw_iterations": int(rr_iterations),
            "objective": objective(fisher, oracle_information, rr_p, costs),
        },
    }
    return (designs, certificates) if return_certificates else designs


def design_metrics(fisher, information, oracle_probabilities, main_probabilities,
                   pilot_counts, main_counts):
    """Return true allocation objectives, separate from realized KL risk."""
    costs = np.ones(len(main_probabilities), dtype=float)
    phi_oracle = objective(fisher, information, oracle_probabilities, costs)
    phi_main = objective(fisher, information, main_probabilities, costs)
    total_counts = np.asarray(pilot_counts, dtype=float) + np.asarray(main_counts, dtype=float)
    if total_counts.sum() <= 0:
        raise ValueError("total allocation counts must be positive")
    total_probabilities = total_counts / total_counts.sum()
    phi_total = objective(fisher, information, total_probabilities, costs)
    return {
        "phi_oracle": float(phi_oracle),
        "phi_main": float(phi_main),
        "phi_total_counts": float(phi_total),
        "design_ratio_main": float(phi_main / phi_oracle),
        "design_ratio_total_counts": float(phi_total / phi_oracle),
    }


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


def _project_theta(beta, theta_bound=4.0, norm_cap=None, l1_cap=None):
    """Project a pilot iterate into the frozen compact parameter set.

    The PDF leaves the numerical shape of ``Theta`` open; our formal config
    uses coordinate, Euclidean and L1 bounds to keep the exact tilt overlap
    finite.  Radial projection onto each active bound is deterministic and
    preserves the target interior while avoiding unconstrained pilot blow-up.
    """
    x = np.clip(np.asarray(beta, dtype=float), -float(theta_bound), float(theta_bound))
    if norm_cap is not None:
        nrm = float(np.linalg.norm(x))
        if nrm > float(norm_cap) > 0.0:
            x *= float(norm_cap) / nrm
    if l1_cap is not None:
        l1 = float(np.abs(x).sum())
        if l1 > float(l1_cap) > 0.0:
            x *= float(l1_cap) / l1
    return x


def solve_pilot_beta(mu_pil, reference, scale, theta_bound=4.0, steps=20, norm_cap=None, l1_cap=None):
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
            candidate = _project_theta(beta + step * direction, theta_bound, norm_cap, l1_cap)
            if objective(candidate) <= current + 1e-9:
                break
            step *= 0.5
        if step <= 1e-5 or np.linalg.norm(candidate - beta) < 1e-7:
            break
        beta = candidate
    return _project_theta(beta, theta_bound, norm_cap, l1_cap)


def prepare_s1_oracle(mixture, scale, panels, seed=2026, reference_size=50000, information_samples=256, conditional_samples=32, large_reference_size=200000,
                      feature_fn=None):
    reference = sample_full(mixture, reference_size, seed)
    reference_large = sample_full(mixture, large_reference_size, seed + 12345)
    beta_true = beta_direction_and_scale(reference, 2026, 0.5, scale, feature_fn=feature_fn)
    fisher, oracle_information = exact_panel_information(mixture, beta_true, panels, reference, scale, information_samples, conditional_samples, seed + 1, feature_fn=feature_fn)
    designs, design_certificates = policy_designs(
        reference, panels, fisher, oracle_information, return_certificates=True
    )
    parameter_digest = hashlib.sha256()
    for value in (scale, beta_true, reference, reference_large, fisher, oracle_information):
        parameter_digest.update(np.ascontiguousarray(value).tobytes())
    artifact_metadata = {
        "schema_version": "p4-oracle-artifact-v2",
        "mixture_seed": int(seed),
        "alpha": float(mixture.alpha),
        "reference_size": int(reference_size),
        "large_reference_size": int(large_reference_size),
        "information_tilted_samples": int(information_samples),
        "information_conditional_samples": int(conditional_samples),
        "information_method": "cross_completion_rejection",
        "parameter_sha256": parameter_digest.hexdigest(),
    }
    return {
        "reference": reference,
        "reference_large": reference_large,
        "beta_true": beta_true,
        "fisher": fisher,
        "information": oracle_information,
        "designs": designs,
        "design_certificates": design_certificates,
        "oracle_constant": {
            "phi_oracle": design_certificates["oracle RR-GID"]["objective"],
            "half_phi_oracle": 0.5 * design_certificates["oracle RR-GID"]["objective"],
            **design_certificates["oracle RR-GID"],
        },
        "artifact_metadata": artifact_metadata,
    }


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


def panel_information_cross(mixture, beta, panels, reference, scale, n_tilted, n_cond, seed, feature_fn=None, conditional_method="rejection", qmc_order=8, qmc_start_order=8, qmc_max_order=16, qmc_atol=2e-6, qmc_rtol=2e-5, qmc_chunk_rows=128):
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
        if conditional_method in ("qmc", "exact_adaptive"):
            # Chunk rows to keep the (rows x 2**order x d) QMC tensor bounded
            # at formal budgets.  The chunks use disjoint deterministic seeds
            # and are concatenated exactly as one cross-completion sample.
            chunks_a, chunks_b = [], []
            chunk_rows = int(qmc_chunk_rows)
            for start in range(0, len(observed), chunk_rows):
                stop = min(start + chunk_rows, len(observed))
                if conditional_method == "exact_adaptive":
                    chunks_a.append(tilted_conditional_mean_exact(mixture, beta, observed[start:stop], panel, seed=seed + 1 + start, scale=scale, feature_fn=feature_fn, start_order=qmc_start_order, max_order=qmc_max_order, atol=qmc_atol, rtol=qmc_rtol))
                    chunks_b.append(tilted_conditional_mean_exact(mixture, beta, observed[start:stop], panel, seed=seed + 2 + start, scale=scale, feature_fn=feature_fn, start_order=qmc_start_order, max_order=qmc_max_order, atol=qmc_atol, rtol=qmc_rtol))
                else:
                    chunks_a.append(tilted_conditional_mean_qmc(mixture, beta, observed[start:stop], panel, qmc_order, seed=seed + 1 + start, scale=scale, feature_fn=feature_fn))
                    chunks_b.append(tilted_conditional_mean_qmc(mixture, beta, observed[start:stop], panel, qmc_order, seed=seed + 2 + start, scale=scale, feature_fn=feature_fn))
            a = np.concatenate(chunks_a, axis=0) - mu
            b = np.concatenate(chunks_b, axis=0) - mu
        else:
            a = tilted_conditional_feature_mean_batch(mixture, beta, observed, panel, n_cond, seed + 1, scale, feature_fn=feature_fn) - mu
            b = tilted_conditional_feature_mean_batch(mixture, beta, observed, panel, n_cond, seed + 2, scale, feature_fn=feature_fn) - mu
        a = a - a.mean(0)
        b = b - b.mean(0)
        info_hat = (a.T @ b + b.T @ a) / max(2 * (len(a) - 1), 1)
        vals, vecs = np.linalg.eigh((info_hat + info_hat.T) / 2)
        infos.append((vecs * np.maximum(vals, 1e-10)) @ vecs.T)
    return np.asarray(infos)


def active_panel_information_cross(mixture, beta, active_panels, reference, scale,
                                   n_tilted, n_cond, seed, feature_fn=None, conditional_method="rejection", qmc_order=8, qmc_start_order=8, qmc_max_order=16, qmc_atol=2e-6, qmc_rtol=2e-5, qmc_chunk_rows=128):
    """Estimate current-step information only for panels present in ``D_B``.

    Algorithm 2 step 6 requires re-estimation for every *active* panel.  The
    design-stage library can contain 120 candidates, while rounding often
    leaves only a much smaller subset active.  Sharing the same tilted full
    draws across that subset preserves Equation (9) and avoids estimating
    matrices that never enter ``H_j``.
    """
    if not active_panels:
        return {}
    estimates = panel_information_cross(
        mixture, beta, tuple(active_panels), reference, scale,
        n_tilted, n_cond, seed, feature_fn=feature_fn, conditional_method=conditional_method, qmc_order=qmc_order, qmc_start_order=qmc_start_order, qmc_max_order=qmc_max_order, qmc_atol=qmc_atol, qmc_rtol=qmc_rtol, qmc_chunk_rows=qmc_chunk_rows,
    )
    return dict(zip(active_panels, estimates))


def largest_remainder_counts(probabilities, budget):
    probabilities = np.asarray(probabilities, dtype=float)
    budget = int(budget)
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    if probabilities.ndim != 1 or np.any(probabilities < 0):
        raise ValueError("probabilities must be a nonnegative vector")
    if budget == 0:
        return np.zeros(len(probabilities), dtype=int)
    expected = budget * probabilities
    counts = np.floor(expected).astype(int)
    remainder = budget - int(counts.sum())
    if remainder:
        counts[np.argsort(expected - counts)[-remainder:]] += 1
    if int(counts.sum()) != budget:
        raise RuntimeError("panel allocation does not exhaust the acquisition budget")
    return counts


def gold_oracle_start_step(
    oracle,
    beta,
    observations,
    panels,
    information,
    seed,
    *,
    theta_bound=4.0,
    norm_cap=None,
    l1_cap=None,
    step_size=1.0,
    max_step_norm=None,
    h_mode="frozen",
    estimated_h=None,
    score_mode="gold",
    qmc_order=10,
    lu=128,
):
    """One ladder Fisher-scoring step. G0 uses frozen gold H; G1 uses current-β gold H."""
    beta = np.asarray(beta, dtype=float)
    grouped = {}
    for panel, obs in observations:
        grouped.setdefault(tuple(panel), []).append(np.asarray(obs, dtype=float))
    if not grouped:
        raise ValueError("ladder step requires observations")
    H = np.zeros((beta.shape[0], beta.shape[0]))
    for panel, rows in grouped.items():
        if h_mode == "gold_current":
            panel_h = oracle.panel_information(beta, panel)["projected"]
        elif h_mode == "estimated":
            if estimated_h is None or panel not in estimated_h:
                raise ValueError(f"estimated H missing panel {panel}")
            panel_h = estimated_h[panel]
        elif h_mode == "frozen":
            panel_h = information[panels.index(panel)]
        else:
            raise ValueError(f"unsupported h_mode: {h_mode}")
        H += len(rows) * np.asarray(panel_h)
    H = (H + H.T) / 2
    try:
        chol = np.linalg.cholesky(H)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError("ladder H is not positive definite") from exc
    mu = np.asarray(oracle.mu(beta), dtype=float)
    projected = []
    query_converged = True
    rng = np.random.default_rng(seed)
    for panel, rows in grouped.items():
        rows = np.asarray(rows)
        scramble_seed = int(rng.integers(2**31 - 1))
        if score_mode == "gold":
            mean, diag = oracle.conditional_mean(
                beta,
                panel,
                rows,
                seed_offset=scramble_seed,
                return_diagnostics=True,
            )
            query_converged = query_converged and bool(diag.get("converged", False))
        elif score_mode == "qmc":
            mean = tilted_conditional_mean_qmc(
                oracle.mixture, beta, rows, panel, int(qmc_order),
                seed=scramble_seed, scale=oracle.scale, feature_fn=oracle.feature_fn,
            )
        elif score_mode == "rejection":
            mean = tilted_conditional_feature_mean_batch(
                oracle.mixture, beta, rows, panel, int(lu),
                scramble_seed, oracle.scale, feature_fn=oracle.feature_fn,
            )
        else:
            raise ValueError(f"unsupported score_mode: {score_mode}")
        projected.append(mean)
    U = np.sum(np.concatenate(projected, axis=0) - mu, axis=0)
    step = np.linalg.solve(chol.T, np.linalg.solve(chol, U))
    scaled = float(step_size) * step
    if max_step_norm is not None:
        step_norm = float(np.linalg.norm(scaled))
        if step_norm > float(max_step_norm) > 0.0:
            scaled *= float(max_step_norm) / step_norm
    updated = _project_theta(beta + scaled, theta_bound, norm_cap, l1_cap)
    eig = np.linalg.eigvalsh(H)
    return updated, {
        "lambda_min_H": float(eig.min()),
        "lambda_max_H": float(eig.max()),
        "H_condition_number": float(eig.max() / max(eig.min(), 1e-30)),
        "score_norm": float(np.linalg.norm(U)),
        "newton_decrement": float(U @ step),
        "raw_step_norm": float(np.linalg.norm(step)),
        "applied_step_norm": float(np.linalg.norm(updated - beta)),
        "linear_algebra": "cholesky_solve",
        "h_mode": str(h_mode),
        "score_mode": str(score_mode),
        "conditional_converged": bool(query_converged) if score_mode == "gold" else None,
        "n_observations": int(sum(len(rows) for rows in grouped.values())),
        "active_panels": int(len(grouped)),
    }


def final_rr_estimator(mixture, beta_start, observations, panels, reference, scale, lu, seed,
                       theta_bound=4.0, h_tilted=128, h_cond=32, step_size=1.0, norm_cap=None,
                       l1_cap=None, mu_samples=10000, mu_direct=False, oracle_information=None,
                       feature_fn=None, max_step_norm=None, return_diagnostics=False,
                       conditional_method="rejection", qmc_order=10, qmc_start_order=8,
                       qmc_max_order=16, qmc_atol=2e-6, qmc_rtol=2e-5):
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
    active_panels = tuple(grouped)
    _sync_cuda()
    h_started = perf_counter()
    if oracle_information is None:
        active_infos = active_panel_information_cross(
            mixture, beta_start, active_panels, reference, scale,
            h_tilted, h_cond, seed + 7, feature_fn=feature_fn, conditional_method=conditional_method, qmc_order=qmc_order, qmc_start_order=qmc_start_order, qmc_max_order=qmc_max_order, qmc_atol=qmc_atol, qmc_rtol=qmc_rtol,
        )
    else:
        active_infos = {panel: oracle_information[panels.index(panel)] for panel in active_panels}
    H = np.zeros((beta_start.shape[0], beta_start.shape[0]))
    for panel, rows in grouped.items():
        H += len(rows) * active_infos[panel]
    _sync_cuda()
    time_h = perf_counter() - h_started
    projected = []
    score_started = perf_counter()
    for panel, rows in grouped.items():
        # S1 uses the exact conditional oracle for the observed score. The
        # finite-LU importance proposal is reserved for learned-generator
        # experiments, not the oracle gate.
        if conditional_method == "exact_adaptive":
            projected.append(tilted_conditional_mean_exact(
                mixture, beta_start, np.asarray(rows), panel,
                seed=int(rng.integers(2**31 - 1)), scale=scale, feature_fn=feature_fn,
                start_order=qmc_start_order, max_order=qmc_max_order,
                atol=qmc_atol, rtol=qmc_rtol,
            ))
        elif conditional_method == "qmc":
            projected.append(tilted_conditional_mean_qmc(
                mixture, beta_start, np.asarray(rows), panel, qmc_order,
                seed=int(rng.integers(2**31 - 1)), scale=scale, feature_fn=feature_fn,
            ))
        else:
            projected.append(tilted_conditional_feature_mean_batch(
                mixture, beta_start, np.asarray(rows), panel, lu,
                int(rng.integers(2**31 - 1)), scale, feature_fn=feature_fn,
            ))
    _sync_cuda()
    time_score = perf_counter() - score_started
    projected = np.concatenate(projected, axis=0)
    mu_started = perf_counter()
    if mu_direct:
        mu_beta = fn(tilted_full_sample(mixture, beta_start, mu_samples, seed + 99, scale, feature_fn=fn)).mean(0)
    else:
        mu_beta, _ = tilted_moments(beta_start, reference, scale, feature_fn=fn)
    _sync_cuda()
    time_mu = perf_counter() - mu_started
    U = np.sum(projected - mu_beta, axis=0)
    H = (H + H.T) / 2
    solve_started = perf_counter()
    step = np.linalg.pinv(H, rcond=1e-10) @ U
    _sync_cuda()
    time_linear_solve = perf_counter() - solve_started
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
    diagnostics = {
        "lambda_min_H": float(np.linalg.eigvalsh(H).min()),
        "lambda_max_H": float(np.linalg.eigvalsh(H).max()),
        "score_norm": float(np.linalg.norm(U)),
        "raw_step_norm": float(np.linalg.norm(step)),
        "applied_step_norm": float(np.linalg.norm(updated - beta_start)),
        "time_mu": float(time_mu),
        "time_H": float(time_h),
        "time_score": float(time_score),
        "time_linear_solve": float(time_linear_solve),
    }
    return (updated, diagnostics) if return_diagnostics else updated


def run_replication(mixture, scale, panels, budget, seed, prepared=None,
                    lu=128, h_tilted=128, h_cond=32, pilot_norm_cap=2.0, kl_samples=20000,
                    scoring_steps=2, theta_norm_cap=None, policies=None, mu_direct=False, mu_samples=10000,
                    kl_mu_direct=True, frozen_beta_star_information=False,
                    use_oracle_H=None, validation_size=10000,
                    mlp_hidden=64, mlp_steps=200, generator=None,
                    gen_info_tilted=256, gen_info_cond=32, disc_reference_size=None,
                    feature_fn=None, scoring_step_size=1.0, theta_l1_cap=None,
                    scoring_max_step_norm=None, conditional_method="rejection", qmc_order=10,
                    qmc_start_order=8, qmc_max_order=16, qmc_atol=2e-6, qmc_rtol=2e-5,
                    pilot_schedule=None, pilot_budget=None, seed_manifest_entry=None,
                    reproducibility=None):
    reset_workload_counters()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    prepared = prepared or prepare_s1_oracle(mixture, scale, panels, seed, feature_fn=feature_fn)
    if use_oracle_H is not None:
        if frozen_beta_star_information not in (False, bool(use_oracle_H)):
            raise ValueError("conflicting frozen_beta_star_information/use_oracle_H values")
        frozen_beta_star_information = bool(use_oracle_H)
    reference = prepared["reference"]
    ref_large = prepared["reference_large"]
    beta_true = prepared["beta_true"]
    fisher = prepared["fisher"]
    oracle_information = prepared["information"]
    designs = prepared["designs"]
    fn = (lambda x: feature_map(x, scale)) if feature_fn is None else feature_fn
    target_reference = ref_large
    _sync_cuda()
    target_started = perf_counter()
    seeds = dict(seed_manifest_entry or {})
    replication_seed = int(seeds.get("replication_seed", seed))
    target_draw_seed = int(seeds.get("target_draw_seed", seed + 3))
    pilot_or_design_seed = int(seeds.get("pilot_or_design_seed", seed + 11))
    score_seed_root = int(seeds.get("score_seed_root", seed + 4))
    information_seed_root = int(seeds.get("information_seed_root", seed + 7000))
    target_full = tilted_full_sample(
        mixture, beta_true, budget, target_draw_seed, scale, feature_fn=fn
    )
    _sync_cuda()
    time_target_sampling = perf_counter() - target_started
    target_draw_sha256 = hashlib.sha256(np.ascontiguousarray(target_full).tobytes()).hexdigest()
    rows = []
    if pilot_budget is None:
        if pilot_schedule is None:
            pilot_schedule = {
                "kind": "power", "exponent": 1.0 / 3.0, "multiplier": 10.0,
                "max_fraction": 1.0, "min_per_support": 0, "rounding_rule": "ceil",
                "legacy_named_ablation": "ten_times_B_one_third",
            }
        from .p4_integrity import compute_pilot_budget
        pilot_budget = compute_pilot_budget(pilot_schedule, budget)
    pilot_budget = int(pilot_budget)
    if pilot_budget < 0 or pilot_budget > budget:
        raise ValueError("runner-supplied pilot_budget is outside [0, budget]")
    remaining_budget = budget - pilot_budget
    rr_phi = float(np.trace(fisher @ np.linalg.pinv(np.tensordot(designs["oracle RR-GID"], oracle_information, axes=(0, 0)))))
    kl_preparation_started = perf_counter()
    if kl_mu_direct:
        mu_bt = fn(tilted_full_sample(mixture, beta_true, kl_samples, seed + 90, scale, feature_fn=fn)).mean(0)
    else:
        mu_bt = tilted_moments(beta_true, target_reference, scale, feature_fn=fn)[0]
    _sync_cuda()
    time_kl_preparation = perf_counter() - kl_preparation_started
    shared_workload = workload_counters()
    policy_names = policies if policies is not None else list(designs)
    for name in policy_names:
        policy_started = perf_counter()
        policy_workload_before = workload_counters()
        pilot_build_started = perf_counter()
        pilot_counts = balanced_pilot_counts(panels, pilot_budget)
        pilot_observations = []
        pilot_cursor = 0
        for panel, count in zip(panels, pilot_counts):
            for row in target_full[pilot_cursor : pilot_cursor + count]:
                pilot_observations.append((panel, row[list(panel)]))
            pilot_cursor += count
        pilot_mu, pilot_rho = pilot_ht_moment(pilot_observations, pilot_counts, panels, scale, reference)
        time_pilot_build = perf_counter() - pilot_build_started
        pilot_solve_started = perf_counter()
        beta_hat = solve_pilot_beta(
            pilot_mu,
            reference,
            scale,
            norm_cap=theta_norm_cap if theta_norm_cap is not None else pilot_norm_cap,
            l1_cap=theta_l1_cap,
        )
        time_pilot_solve = perf_counter() - pilot_solve_started
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
            validation = sample_full(mixture, validation_size, pilot_or_design_seed + 544)
            disc_ref = reference if disc_reference_size is None else reference[:disc_reference_size]
            design_started = perf_counter()
            probabilities = discriminative_design(disc_ref, validation, beta_hat, panels, scale,
                                                  pilot_or_design_seed, hidden=mlp_hidden, steps=mlp_steps,
                                                  feature_fn=feature_fn)
            time_design_information = perf_counter() - design_started
            time_fw = 0.0
        elif name == "learned RR-GID":
            # PDF P6 generator-aware design: cross-completion I_hat_S from the
            # frozen VAEAC generator, cost-aware Frank-Wolfe design.
            from .vaeac import learned_information
            design_started = perf_counter()
            fisher_hat, infos = learned_information(generator, beta_hat, panels,
                                                    n_tilted=gen_info_tilted,
                                                    n_conditional=gen_info_cond, seed=information_seed_root)
            time_design_information = perf_counter() - design_started
            costs = np.ones(len(panels))
            fw_started = perf_counter()
            probabilities, _, _ = frank_wolfe(fisher_hat, infos, costs, uniform_probabilities(len(panels)),
                                              tolerance=1e-4, max_iter=300)
            time_fw = perf_counter() - fw_started
        else:
            probabilities = designs[name]
            time_design_information = 0.0
            time_fw = 0.0
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
        active_panel_count = len({panel for panel, _obs in observations})
        pilot_mu_true, _ = tilted_moments(beta_true, reference, scale, feature_fn=fn)
        update_diagnostics = [{"step": "pilot", "pilot_budget": int(pilot_counts.sum()), "beta_norm": float(np.linalg.norm(beta_hat)), "rho_min": float(np.min(pilot_rho[pilot_rho > 0])) if np.any(pilot_rho > 0) else 0.0,
                              "pilot_mu_residual_norm": float(np.linalg.norm(pilot_mu - pilot_mu_true)),
                              "pilot_beta_error_norm": float(np.linalg.norm(beta_hat - beta_true))}]
        # PDF Algorithm 2 projects every Fisher-scoring iterate onto the
        # compact parameter set Theta.  theta_norm_cap is the configured
        # Euclidean radius for that set; previously the argument existed but
        # was silently ignored, allowing overlap collapse in exact TiltCond.
        norm_cap_val = theta_norm_cap
        for update in range(scoring_steps):
            beta_next, step_diagnostics = final_rr_estimator(mixture, beta_hat, observations, panels, ref_large, scale, lu, score_seed_root + update, h_tilted=h_tilted, h_cond=h_cond, step_size=scoring_step_size, norm_cap=norm_cap_val, l1_cap=theta_l1_cap, mu_direct=mu_direct, mu_samples=mu_samples, oracle_information=oracle_information if frozen_beta_star_information else None, feature_fn=feature_fn, max_step_norm=scoring_max_step_norm, return_diagnostics=True, conditional_method=conditional_method, qmc_order=qmc_order, qmc_start_order=qmc_start_order, qmc_max_order=qmc_max_order, qmc_atol=qmc_atol, qmc_rtol=qmc_rtol)
            update_diagnostics.append({"step": update, "step_norm": float(np.linalg.norm(beta_next - beta_hat)), "projected": bool(np.any(np.abs(beta_next) >= 4.0)), "pilot_budget": int(pilot_counts.sum()), **step_diagnostics})
            beta_hat = beta_next
        # Keep the untruncated plug-in Bregman value for the numerical gate.
        # ``kl`` remains the historical non-negative compatibility field, but
        # acceptance must inspect ``kl_raw`` rather than silently hiding a
        # negative estimate behind max(0, ...).
        kl_started = perf_counter()
        kl_raw = float((beta_true - beta_hat) @ mu_bt - log_partition(beta_true, target_reference, scale, feature_fn=fn) + log_partition(beta_hat, target_reference, scale, feature_fn=fn))
        _sync_cuda()
        time_kl = time_kl_preparation + perf_counter() - kl_started
        kl = max(0.0, kl_raw)
        step_times = {
            key: float(sum(
                diag.get(key, 0.0) for diag in update_diagnostics
                if isinstance(diag.get("step"), int)
            ))
            for key in ("time_mu", "time_H", "time_score", "time_linear_solve")
        }
        attributed = (
            time_target_sampling + time_pilot_build + time_pilot_solve
            + time_design_information + time_fw + time_kl + sum(step_times.values())
        )
        time_total = time_target_sampling + time_kl_preparation + (perf_counter() - policy_started)
        runtime = {
            "time_target_sampling": float(time_target_sampling),
            "time_pilot_build": float(time_pilot_build),
            "time_pilot_solve": float(time_pilot_solve),
            "time_design_information": float(time_design_information),
            "time_fw": float(time_fw),
            **step_times,
            "time_kl": float(time_kl),
            "time_total": float(time_total),
            "time_attributed": float(attributed),
            "attributed_fraction": float(attributed / max(time_total, 1e-12)),
            "device": _runtime_device(),
            "memory": _peak_memory(),
            "active_panels": int(active_panel_count),
            "qmc_orders": workload_qmc_orders(),
            "conditional_acceptance": workload_rate_summary(),
        }
        policy_workload = _counter_delta(workload_counters(), policy_workload_before)
        total_workload = {
            key: int(shared_workload.get(key, 0) + policy_workload.get(key, 0))
            for key in shared_workload
        }
        metrics = design_metrics(
            fisher, oracle_information, designs["oracle RR-GID"],
            probabilities, pilot_counts, counts,
        )
        B_kl_raw = budget * kl_raw
        risk_ratio_raw = B_kl_raw / max(0.5 * metrics["phi_oracle"], 1e-12)
        row = {
            "policy": name, "budget": budget,
            "allocated_observations": int(counts.sum() + pilot_counts.sum()),
            "pilot_schedule": pilot_schedule,
            "pilot_budget": int(pilot_counts.sum()),
            "pilot_counts": pilot_counts.tolist(),
            "main_counts": counts.tolist(),
            "total_counts": (pilot_counts + counts).tolist(),
            "seed": replication_seed,
            "replication_seed": replication_seed,
            "target_draw_seed": target_draw_seed,
            "pilot_or_design_seed": pilot_or_design_seed,
            "score_seed_root": score_seed_root,
            "information_seed_root": information_seed_root,
            "beta_true_norm": float(np.linalg.norm(beta_true)),
            "beta_hat_norm": float(np.linalg.norm(beta_hat)),
            "kl_raw": kl_raw, "B_kl_raw": B_kl_raw,
            "kl": kl, "B_kl": budget * kl,
            "risk_ratio_raw": float(risk_ratio_raw),
            "design_ratio_legacy": float(kl / max(rr_phi / (2 * budget), 1e-12)),
            "design_ratio_legacy_definition": "realized_risk_ratio_clipped_not_for_formal_use",
            **metrics,
            "fw_gap": float(prepared.get("oracle_constant", {}).get(
                "fw_gap", prepared.get("design_certificates", {}).get(
                    "oracle RR-GID", {}
                ).get("fw_gap", np.nan)
            )),
            "fw_iterations": int(prepared.get("oracle_constant", {}).get(
                "fw_iterations", prepared.get("design_certificates", {}).get(
                    "oracle RR-GID", {}
                ).get("fw_iterations", -1)
            )),
            "frozen_beta_star_information": bool(frozen_beta_star_information),
            "target_draw_sha256": target_draw_sha256,
            "beta_hat": beta_hat.tolist(),
            "update_diagnostics": update_diagnostics,
            "runtime": runtime, "workload": total_workload,
        }
        if reproducibility:
            row.update(reproducibility)
        rows.append(row)
    return rows
