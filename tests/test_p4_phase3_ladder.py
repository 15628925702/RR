import numpy as np
import pytest

from rr_gid_cn.s1_gate import gold_oracle_start_step
from scripts.p4_phase3_ladder import _ladder_lu, ladder_gate, validate_ladder_config


class _ToyOracle:
    def __init__(self):
        self.mixture = None
        self.scale = np.ones(2)
        self.feature_fn = None
        self.n_info = 0

    def mu(self, beta):
        return np.zeros(len(beta))

    def conditional_mean(self, beta, panel, x_s, *, seed_offset=0, return_diagnostics=False):
        mean = np.ones((len(x_s), len(beta))) * 0.02
        if return_diagnostics:
            return mean, {"converged": True}
        return mean

    def panel_information(self, beta, panel):
        self.n_info += 1
        return {"projected": 2.0 * np.eye(len(beta))}


def test_gold_current_h_uses_panel_information():
    oracle = _ToyOracle()
    observations = [((0, 1), np.zeros(2))] * 4 + [((1, 2), np.zeros(2))] * 6
    updated, diag = gold_oracle_start_step(
        oracle,
        np.zeros(2),
        observations,
        ((0, 1), (1, 2)),
        np.stack([np.eye(2), np.eye(2)]),
        seed=3,
        h_mode="gold_current",
    )
    assert oracle.n_info == 2
    assert diag["h_mode"] == "gold_current"
    # H = 4*2I + 6*2I = 20 I; U = 10 * 0.02 * 1 = 0.2; step = 0.01
    np.testing.assert_allclose(updated, np.full(2, 0.01))
    clipped, clip_diag = gold_oracle_start_step(
        oracle,
        np.zeros(2),
        observations,
        ((0, 1), (1, 2)),
        np.stack([np.eye(2), np.eye(2)]),
        seed=3,
        h_mode="gold_current",
        max_step_norm=0.005,
    )
    np.testing.assert_allclose(np.linalg.norm(clipped), 0.005)


def test_ladder_gate_oracle_requires_unit_design_ratio():
    payload = {
        "allocation": "oracle",
        "max_scoring_steps": 4,
        "design_ratio_tolerance": 1e-10,
        "risk_ratio_mean_max": 10.0,
    }
    rows = [
        {
            "scoring_steps": 0,
            "design_ratio_main": 1.0,
            "kl_raw": 0.4,
            "kl_numerical_tolerance": 1e-8,
            "risk_ratio_raw": 8.0,
            "update_diagnostics": [{"step": "pilot"}],
        },
        {
            "scoring_steps": 4,
            "design_ratio_main": 1.0,
            "kl_raw": 0.02,
            "kl_numerical_tolerance": 1e-8,
            "risk_ratio_raw": 1.1,
            "update_diagnostics": [{"step": 0, "linear_algebra": "cholesky_solve"}],
        },
    ]
    assert ladder_gate(rows, payload, half_phi=1.0)["passed"] is True
    rows[1]["design_ratio_main"] = 1.2
    assert ladder_gate(rows, payload, half_phi=1.0)["passed"] is False


def test_ladder_lu_log2_schedule():
    assert _ladder_lu({"lu": 128}, 8000) == 128
    assert _ladder_lu({"lu_schedule": {"scale": 32.0, "offset": 1.0}}, 8000) == 415


def test_validate_rejects_g0_on_ladder_runner():
    with pytest.raises(ValueError, match="G1"):
        validate_ladder_config({"phase": 3, "not_formal": True, "schema_version": "p4-phase3-ladder-config-v1", "ladder": "G0"})
