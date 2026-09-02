from __future__ import annotations

import numpy as np

from rr_gid_cn.conditional_backend import (
    ReferenceMomentCache,
    build_conditional_feature_basis,
    cholesky_solve,
    evaluate_conditional_basis,
    uncached_qmc_mean,
)
from rr_gid_cn.p4_integrity import compute_pilot_budget
from rr_gid_cn.synthetic_oracle import (
    feature_map,
    make_frozen_mixture,
    reference_scale,
    sample_full,
)


def _cpu_qmc(monkeypatch):
    import rr_gid_cn.synthetic_oracle as oracle

    monkeypatch.setattr(oracle, "_cuda_device", lambda: None)


def test_t1_cached_matches_uncached_qmc(monkeypatch):
    _cpu_qmc(monkeypatch)
    mixture = make_frozen_mixture(seed=2026, alpha=1.0)
    scale = reference_scale(mixture, n=400, seed=7)
    panel = (0, 1)
    rows = sample_full(mixture, 8, seed=19)[:3][:, list(panel)]
    beta = np.linspace(-0.3, 0.4, 12)
    order, seed = 4, 123
    basis = build_conditional_feature_basis(
        mixture, rows, panel, order=order, seed=seed, scale=scale, dtype="float64",
    )
    cached = evaluate_conditional_basis(beta, basis)
    fresh = evaluate_conditional_basis(beta, basis)
    assert np.max(np.abs(cached - fresh)) <= 1e-12
    uncached = uncached_qmc_mean(mixture, beta, rows, panel, order, seed, scale)
    assert np.max(np.abs(cached - uncached)) <= 1e-10


def test_t2_beta_zero_is_q0_conditional_mean(monkeypatch):
    _cpu_qmc(monkeypatch)
    mixture = make_frozen_mixture(seed=2026, alpha=1.0)
    scale = reference_scale(mixture, n=400, seed=7)
    panel = (2, 5)
    rows = sample_full(mixture, 4, seed=21)[:, list(panel)]
    basis = build_conditional_feature_basis(
        mixture, rows, panel, order=4, seed=5, scale=scale, dtype="float64",
    )
    cached = evaluate_conditional_basis(np.zeros(12), basis)
    phi = basis.phi_nodes.astype(np.float64)
    prob = np.exp(basis.component_log_posterior)
    prob = prob / prob.sum(axis=1, keepdims=True)
    q0 = np.einsum("kc,kcnr->kr", prob, phi) / phi.shape[2]
    assert np.max(np.abs(cached - q0)) <= 1e-10


def test_t4_cholesky_matches_numpy_solve():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(6, 6))
    h = a.T @ a + 0.2 * np.eye(6)
    u = rng.normal(size=6)
    step, diag = cholesky_solve(h, u, ridge=1e-10)
    expected = np.linalg.solve(h, u)
    assert np.linalg.norm(step - expected) <= 1e-10
    assert diag["ridge"] == 0.0
    assert diag["linear_algebra"] == "cholesky"


def test_anchored_pilot_schedule_matches_guide_table():
    schedule = {
        "kind": "anchored_power",
        "anchor_budget": 8000,
        "anchor_pilot": 400,
        "exponent": 0.5,
        "max_fraction": 0.2,
    }
    expected = {2000: 200, 4000: 283, 8000: 400, 16000: 566, 32000: 800}
    for budget, target in expected.items():
        assert compute_pilot_budget(schedule, budget) == target


def test_reference_moment_cache_matches_feature_map():
    mixture = make_frozen_mixture(seed=3)
    scale = reference_scale(mixture, n=200, seed=4)
    reference = sample_full(mixture, 64, seed=5)
    cache = ReferenceMomentCache.build(reference, scale, beta_true=np.zeros(12))
    beta = np.linspace(-0.2, 0.15, 12)
    mu, fisher = cache.moments(beta)
    features = feature_map(reference, scale)
    logits = features @ beta
    w = np.exp(logits - logits.max())
    w /= w.sum()
    assert np.linalg.norm(mu - w @ features) <= 1e-12
    assert cache.kl(np.zeros(12), np.zeros(12)) == 0.0
