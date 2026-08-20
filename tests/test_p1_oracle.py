import numpy as np

from rr_gid_cn.synthetic_oracle import (
    all_pairs,
    feature_map,
    inverse_warp,
    make_frozen_mixture,
    reference_scale,
    sample_conditional,
    sample_full,
    warp,
)


def test_warp_inverse_and_panels():
    z = np.linspace(-2, 2, 17)
    assert np.max(np.abs(inverse_warp(warp(z, 1.0), 1.0) - z)) < 1e-10
    assert len(all_pairs()) == 120


def test_conditional_sampler_shape_and_moments():
    mix = make_frozen_mixture()
    full = sample_full(mix, 2500, seed=4)
    panel = (0, 1)
    x_s = full[0, list(panel)]
    draws = sample_conditional(mix, x_s, panel, 2000, seed=5)
    assert draws.shape == (2000, 16)
    assert np.allclose(draws[:, list(panel)], x_s)
    assert np.isfinite(draws).all()


def test_feature_map_and_fisher_psd():
    mix = make_frozen_mixture()
    samples = sample_full(mix, 4000, seed=6)
    phi = feature_map(samples)
    assert phi.shape == (4000, 12)
    assert np.all((phi >= -1) & (phi <= 1))
    centered = phi - phi.mean(0)
    eig = np.linalg.eigvalsh(centered.T @ centered / (len(phi) - 1))
    assert eig.min() >= -1e-10


def test_same_seed_oracle_summary_repeats():
    a = sample_full(make_frozen_mixture(), 32, seed=19)
    b = sample_full(make_frozen_mixture(), 32, seed=19)
    assert a.tobytes() == b.tobytes()


def test_reference_scale_is_deterministic_and_positive():
    mix = make_frozen_mixture()
    a = reference_scale(mix, n=500, seed=21)
    b = reference_scale(mix, n=500, seed=21)
    assert np.array_equal(a, b)
    assert np.all(a > 0)
