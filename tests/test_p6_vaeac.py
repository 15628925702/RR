import numpy as np

from rr_gid_cn.vaeac import VAEACGenerator


def test_arbitrary_conditioning_and_shapes():
    reference = np.arange(800, dtype=float).reshape(100, 8)
    gen = VAEACGenerator(reference)
    observed = np.array([3.0, 4.0])
    draws = gen.sample_conditional(observed, (1, 4), 50, seed=1)
    assert draws.shape == (50, 8)
    assert np.all(draws[:, [1, 4]] == observed)


def test_tilted_sample_diagnostics():
    reference = np.random.default_rng(2).normal(size=(100, 4))
    gen = VAEACGenerator(reference)
    samples, acceptance, ess = gen.tilted_sample(np.zeros(2), lambda x: x[:, :2], 20, seed=3)
    assert samples.shape == (20, 4)
    assert acceptance == 1.0
    assert 0 < ess <= 1

