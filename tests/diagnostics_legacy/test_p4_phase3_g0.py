import numpy as np
import pytest

from rr_gid_cn.s1_gate import (
    design_metrics,
    gold_oracle_start_step,
    largest_remainder_counts,
)
from scripts.p4_phase3_g0 import g0_gate


class _ToyOracle:
    def mu(self, beta):
        return np.zeros(len(beta))

    def conditional_mean(self, beta, panel, x_s, *, seed_offset=0, return_diagnostics=False):
        mean = np.ones((len(x_s), len(beta))) * 0.02
        diagnostics = {"converged": True}
        if return_diagnostics:
            return mean, diagnostics
        return mean


def test_largest_remainder_counts_exhaust_budget():
    counts = largest_remainder_counts(np.array([0.4, 0.6]), 10)
    assert int(counts.sum()) == 10
    assert counts.tolist() == [4, 6]


def test_g0_cholesky_step_moves_along_score_and_j0_gate():
    information = np.stack([np.eye(2), np.eye(2)])
    observations = [((0, 1), np.array([0.0, 0.0]))] * 4 + [((1, 2), np.array([0.0, 0.0]))] * 6
    panels = ((0, 1), (1, 2))
    updated, diag = gold_oracle_start_step(
        _ToyOracle(), np.zeros(2), observations, panels, information, seed=7
    )
    assert diag["linear_algebra"] == "cholesky_solve"
    assert diag["conditional_converged"] is True
    np.testing.assert_allclose(updated, np.full(2, 0.02))
    with pytest.raises(np.linalg.LinAlgError, match="not positive definite"):
        gold_oracle_start_step(
            _ToyOracle(),
            np.zeros(2),
            observations,
            panels,
            np.stack([np.zeros((2, 2)), np.zeros((2, 2))]),
            seed=7,
        )


def test_g0_gate_requires_design_ratio_one_and_j0_at_truth():
    payload = {"design_ratio_tolerance": 1e-10, "risk_ratio_mean_max": 10.0}
    j0 = {
        "scoring_steps": 0,
        "design_ratio_main": 1.0,
        "kl_raw": 0.0,
        "kl_numerical_tolerance": 1e-8,
        "beta_error_norm": 0.0,
        "risk_ratio_raw": 0.0,
        "update_diagnostics": [{"step": "oracle_start"}],
    }
    j2 = {
        "scoring_steps": 2,
        "design_ratio_main": 1.0,
        "kl_raw": 0.01,
        "kl_numerical_tolerance": 1e-8,
        "beta_error_norm": 0.1,
        "risk_ratio_raw": 1.2,
        "update_diagnostics": [{"step": 0, "linear_algebra": "cholesky_solve"}],
    }
    report = g0_gate([j0, j2], payload, half_phi=1.0)
    assert report["passed"] is True
    bad = dict(j2)
    bad["design_ratio_main"] = 1.2
    assert g0_gate([j0, bad], payload, half_phi=1.0)["passed"] is False


def test_g0_design_metrics_on_pstar_is_one():
    fisher = np.eye(2)
    information = np.array([np.diag([2.0, 0.4]), np.diag([0.4, 2.0])])
    p_star = np.array([0.5, 0.5])
    metrics = design_metrics(fisher, information, p_star, p_star, [0, 0], [5, 5])
    assert metrics["design_ratio_main"] == pytest.approx(1.0)
