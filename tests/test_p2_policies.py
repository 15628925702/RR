import numpy as np

from rr_gid_cn.policies import frank_wolfe, objective, round_cost_share, uniform_probabilities


def test_uniform_and_rounding_budget():
    p = uniform_probabilities(4)
    allocation = round_cost_share(p, 100, np.ones(4, dtype=int))
    assert np.isclose(p.sum(), 1)
    assert allocation.spent <= 100
    assert np.all(allocation.counts >= 0)


def test_frank_wolfe_gap_and_objective():
    F = np.eye(2)
    I = np.asarray([np.eye(2), 2 * np.eye(2), 3 * np.eye(2)])
    c = np.ones(3)
    p, gap, _ = frank_wolfe(F, I, c, np.full(3, 1 / 3), tolerance=1e-6)
    assert np.isclose(p.sum(), 1)
    assert gap <= 1e-6 or objective(F, I, p, c) <= 2 / 3 + 1e-4

