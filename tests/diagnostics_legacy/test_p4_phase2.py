import copy
import json

import numpy as np
import pytest
import yaml

from scripts import p4_phase2_score_centering as score_centering_module
from rr_gid_cn.oracle_measure import (
    ConditionalQMC,
    FullLawQMC,
    InformationQMC,
    OracleMeasure,
    cholesky_objective,
)
from rr_gid_cn.synthetic_oracle import FrozenMixture, tilted_conditional_mean_exact
from scripts.p4_phase2_score_centering import (
    active_panel_indices,
    load_verified_base_artifact,
    validate_high_precision_config,
)


def _gaussian_oracle():
    mean = np.array([0.3, -0.2])
    covariance = np.array([[1.4, 0.45], [0.45, 0.9]])
    mixture = FrozenMixture(
        weights=np.ones(1),
        means=mean[None, :],
        covariances=covariance[None, :, :],
        alpha=0.0,
    )
    feature_fn = lambda x: np.asarray(x, dtype=float)
    oracle = OracleMeasure(
        mixture,
        np.ones(2),
        FullLawQMC(order=12, scrambles=6, seed=101),
        ConditionalQMC(
            start_order=5,
            max_order=11,
            scrambles=4,
            atol=2e-3,
            rtol=2e-3,
            scramble_se_atol=2e-3,
            scramble_se_rtol=2e-3,
            seed=211,
        ),
        InformationQMC(outer_order=7, outer_scrambles=3, seed=307),
        feature_fn=feature_fn,
    )
    return oracle, mean, covariance


def test_t04_multi_scramble_adaptive_records_early_and_hard_orders():
    oracle, _, _ = _gaussian_oracle()
    observed = np.array([[0.1], [0.7]])
    easy_value, easy = tilted_conditional_mean_exact(
        oracle.mixture,
        np.zeros(2),
        observed,
        (0,),
        seed=19,
        scale=np.ones(2),
        feature_fn=oracle.feature_fn,
        start_order=5,
        max_order=10,
        atol=0.08,
        rtol=0.0,
        scrambles=4,
        scramble_se_atol=0.08,
        scramble_se_rtol=0.0,
        return_diagnostics=True,
    )
    hard_value, hard = tilted_conditional_mean_exact(
        oracle.mixture,
        np.array([0.0, 0.45]),
        observed,
        (0,),
        seed=19,
        scale=np.ones(2),
        feature_fn=oracle.feature_fn,
        start_order=5,
        max_order=10,
        atol=0.006,
        rtol=0.0,
        scrambles=4,
        scramble_se_atol=0.006,
        scramble_se_rtol=0.0,
        return_diagnostics=True,
    )
    assert easy["converged"] and hard["converged"]
    assert easy["final_order"] < hard["final_order"]
    assert easy["final_order"] < easy["max_order"]
    assert hard["scrambles"] == 4 and hard["scramble_se"] >= 0
    assert np.all(np.isfinite(easy_value)) and np.all(np.isfinite(hard_value))
    with pytest.raises(RuntimeError, match="scramble_se"):
        tilted_conditional_mean_exact(
            oracle.mixture,
            np.array([0.0, 0.45]),
            observed,
            (0,),
            seed=19,
            scale=np.ones(2),
            feature_fn=oracle.feature_fn,
            start_order=5,
            max_order=6,
            atol=1e-12,
            rtol=0.0,
            scrambles=4,
            scramble_se_atol=1e-12,
            scramble_se_rtol=0.0,
        )


def test_t05_single_q0_low_dimensional_analytic_moments_conditional_and_kl():
    oracle, mean, covariance = _gaussian_oracle()
    beta = np.array([0.2, -0.15])
    moments = oracle.moments(beta)
    np.testing.assert_allclose(moments["A"], beta @ mean + 0.5 * beta @ covariance @ beta, atol=8e-4)
    np.testing.assert_allclose(moments["mu"], mean + covariance @ beta, atol=1.5e-3)
    np.testing.assert_allclose(moments["F_projected"], covariance, atol=2e-3)
    observed = np.array([[-0.4], [0.8]])
    conditional, diagnostics = oracle.conditional_mean(
        beta, (0,), observed, return_diagnostics=True
    )
    gain = covariance[1, 0] / covariance[0, 0]
    conditional_variance = covariance[1, 1] - covariance[1, 0] ** 2 / covariance[0, 0]
    expected_second = mean[1] + gain * (observed[:, 0] - mean[0]) + conditional_variance * beta[1]
    np.testing.assert_allclose(conditional[:, 0], observed[:, 0], atol=1e-12)
    np.testing.assert_allclose(conditional[:, 1], expected_second, atol=5e-3)
    assert diagnostics["converged"] and diagnostics["scrambles"] == 4
    beta_hat = beta + np.array([0.03, -0.02])
    expected_kl = 0.5 * (beta - beta_hat) @ covariance @ (beta - beta_hat)
    kl = oracle.kl(beta, beta_hat)
    assert kl["raw"] >= -kl["numerical_tolerance"]
    assert kl["raw"] == pytest.approx(expected_kl, abs=2e-4)


def test_t06_gold_objective_uses_positive_definite_cholesky_route():
    fisher = np.eye(2)
    information = np.array([np.diag([2.0, 0.2]), np.diag([0.2, 2.0])])
    value, diagnostics = cholesky_objective(fisher, information, np.array([0.5, 0.5]))
    assert value == pytest.approx(2 / 1.1)
    assert diagnostics["lambda_min"] > 0 and diagnostics["condition_number"] == pytest.approx(1.0)
    with pytest.raises(np.linalg.LinAlgError, match="not positive definite"):
        cholesky_objective(fisher, np.zeros((1, 2, 2)), np.ones(1))


def _hp_payload():
    return yaml.safe_load(
        open("configs/p4_phase2_gold_hp1_20260827.yaml", encoding="utf-8")
    )["p4_phase2_score_centering"]


def test_hp_score_centering_config_is_frozen_and_stricter():
    payload = _hp_payload()
    validate_high_precision_config(payload)
    assert payload["phase"] == 2 and payload["not_formal"] is True
    assert payload["risk_fraction_of_half_phi_max"] == pytest.approx(0.01)
    assert payload["outer_qmc"]["order"] > 6
    assert payload["outer_qmc"]["scrambles"] > 3
    assert payload["conditional_qmc"]["max_order"] > 10
    assert payload["conditional_qmc"]["scrambles"] > 4


def test_cond5e5_config_tightens_t04_without_relaxing_c01_gate():
    payload = yaml.safe_load(
        open("configs/p4_phase2_gold_full16_cond5e5_20260827.yaml", encoding="utf-8")
    )["p4_phase2_score_centering"]
    validate_high_precision_config(payload)
    assert payload["Bmax"] == 32000
    assert payload["risk_fraction_of_half_phi_max"] == pytest.approx(0.01)
    assert payload["conditional_qmc"]["terminal_delta_atol"] == pytest.approx(5e-5)
    assert payload["conditional_qmc"]["scramble_se_atol"] == pytest.approx(5e-5)
    assert payload["conditional_qmc"]["terminal_delta_atol"] < 7.5e-4
    assert payload["outer_qmc"]["order"] == 16
    assert payload["base_artifact"]["directory"] == "results/p4/phase2_gold_full16_20260827"


def test_hp_active_panel_coverage_cannot_omit_nonignored_probability():
    probabilities = np.array([0.6, 2e-12, 0.4 - 3.02e-10, 3e-10])
    active, report = active_panel_indices(
        probabilities, numerical_probability_threshold=1e-9, required_top_count=2
    )
    assert active.tolist() == [0, 2]
    assert report["truncated_mass"] == pytest.approx(3.02e-10)
    assert report["covered_probability_mass"] == pytest.approx(1.0 - 3.02e-10)
    with pytest.raises(ValueError, match="non-ignored"):
        active_panel_indices(
            probabilities,
            numerical_probability_threshold=1e-9,
            required_top_count=2,
            proposed_indices=[0],
        )


def test_hp_gate_cannot_be_configured_above_one_percent():
    payload = _hp_payload()
    payload["risk_fraction_of_half_phi_max"] = 0.0100001
    with pytest.raises(ValueError, match="0.01"):
        validate_high_precision_config(payload)


def test_hp_artifact_mismatch_fails_before_numerical_work(tmp_path):
    payload = _hp_payload()
    payload["base_artifact"]["npz_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="npz SHA256 mismatch"):
        load_verified_base_artifact(payload, repository_root=".")


def test_t04_easy_query_does_not_fill_max_order():
    from rr_gid_cn.synthetic_oracle import reset_workload_counters, workload_qmc_orders

    oracle, _, _ = _gaussian_oracle()
    reset_workload_counters()
    _, easy = tilted_conditional_mean_exact(
        oracle.mixture,
        np.zeros(2),
        np.array([[0.1], [0.7]]),
        (0,),
        seed=19,
        scale=np.ones(2),
        feature_fn=oracle.feature_fn,
        start_order=5,
        max_order=10,
        atol=0.08,
        rtol=0.0,
        scrambles=4,
        scramble_se_atol=0.08,
        scramble_se_rtol=0.0,
        return_diagnostics=True,
    )
    assert easy["converged"]
    assert easy["final_order"] < easy["max_order"]
    assert max(workload_qmc_orders()) == easy["final_order"]
    assert max(workload_qmc_orders()) < easy["max_order"]
    assert str(easy["max_order"]) not in easy["n_active_by_order"]


def test_hp_document_estimator_uses_oracle_mu_not_outer_phi(monkeypatch):
    reference_mean = np.array([1.0, -0.5])
    offset = np.array([0.25, 0.2])

    def fake_exact(*args, **kwargs):
        rows = len(np.atleast_2d(args[2]))
        value = np.tile(reference_mean + offset, (rows, 1))
        diagnostics = {
            "converged": True,
            "max_abs_delta": 0.0,
            "scramble_se": 0.0,
            "row_final_order": np.full(rows, 6, dtype=int),
            "n_active_by_order": {"6": 0},
        }
        return value, diagnostics

    monkeypatch.setattr(
        score_centering_module,
        "tilted_conditional_mean_exact",
        fake_exact,
    )

    class FakeOracle:
        mixture = object()
        scale = np.ones(2)
        feature_fn = None

        @staticmethod
        def mu(beta):
            return reference_mean

    cfg = {
        "scrambles": 4,
        "start_order": 5,
        "max_order": 6,
        "terminal_delta_atol": 1e-12,
        "terminal_delta_rtol": 0.0,
        "scramble_se_atol": 1e-12,
        "scramble_se_rtol": 0.0,
    }
    result = score_centering_module._conditional_summary(
        FakeOracle(),
        np.zeros(2),
        (0,),
        np.array([[0.0], [1.0], [2.0]]),
        np.array([0.2, 0.3, 0.5]),
        cfg,
        7,
    )
    np.testing.assert_allclose(result["score_mean"], offset)
    assert result["estimator"] == "same_q0_oracle_mu"
    assert not np.allclose(result["score_mean"], 0.0)

    class OtherMu(FakeOracle):
        @staticmethod
        def mu(beta):
            return np.array([99.0, 99.0])

    shifted = score_centering_module._conditional_summary(
        OtherMu(),
        np.zeros(2),
        (0,),
        np.array([[0.0], [1.0], [2.0]]),
        np.array([0.2, 0.3, 0.5]),
        cfg,
        7,
    )
    assert not np.allclose(shifted["score_mean"], offset)


def _checkpoint_result(value):
    vector = np.array([value, value + 0.5])
    return {
        "delta": vector,
        "outer_numerical_se": vector / 10,
        "conditional_numerical_se": vector / 20,
        "conditional_terminal_delta": vector / 30,
        "sampling_se": vector / 40,
        "row_final_order": np.array([6.0, 7.0, value]),
        "conditional_query_scramble_se_max": float(value / 50),
        "conditional_query_terminal_delta_max": float(value / 60),
        "conditional_converged": True,
        "estimator": "same_q0_oracle_mu",
        "max_final_order": 7,
        "mean_final_order": float(value),
    }


def test_weighted_allocation_vectors_match_gold_pstar_weighting():
    p_star = np.array([0.4, 0.0, 0.6, 0.0])
    panel_results = {
        0: {
            "delta": np.array([1.0, 2.0]),
            "outer_numerical_se": np.array([0.1, 0.2]),
            "conditional_numerical_se": np.array([0.01, 0.02]),
            "sampling_se": np.array([0.001, 0.002]),
        },
        2: {
            "delta": np.array([3.0, 4.0]),
            "outer_numerical_se": np.array([0.3, 0.4]),
            "conditional_numerical_se": np.array([0.03, 0.04]),
            "sampling_se": np.array([0.003, 0.004]),
        },
    }
    vectors = score_centering_module._weighted_allocation_vectors(
        panel_results, [0, 2], p_star
    )
    np.testing.assert_allclose(
        vectors["delta"], 0.4 * np.array([1.0, 2.0]) + 0.6 * np.array([3.0, 4.0])
    )
    np.testing.assert_allclose(
        vectors["outer_numerical_se"],
        np.sqrt(
            np.square(0.4 * np.array([0.1, 0.2]))
            + np.square(0.6 * np.array([0.3, 0.4]))
        ),
    )
    with pytest.raises(ValueError, match="allocation requires all active panels"):
        score_centering_module._weighted_allocation_vectors(
            panel_results, [0, 1, 2], p_star
        )


def _checkpoint_identity():
    return {
        "config_canonical_sha256": "1" * 64,
        "precision_id": "test",
        "base_artifact_sha256": {
            "config": "2" * 64,
            "metadata": "3" * 64,
            "npz": "4" * 64,
        },
        "source_numerical_kernel_sha256": {
            "src/kernel.py": "5" * 64,
        },
    }


def test_checkpoint_resume_skips_completed_panel(tmp_path, monkeypatch):
    output = tmp_path / "run"
    identity = _checkpoint_identity()
    store = score_centering_module.PanelCheckpointStore(
        output, identity, resume=False
    )
    store.save_panel(0, _checkpoint_result(1.0))
    resumed = score_centering_module.PanelCheckpointStore(
        output, identity, resume=True
    )
    computed = []

    def fake_panel_score_mean(oracle, beta, panel, payload, rank, **kwargs):
        computed.append((panel, rank))
        return _checkpoint_result(2.0)

    monkeypatch.setattr(
        score_centering_module, "_panel_score_mean", fake_panel_score_mean
    )
    results = score_centering_module._collect_panel_results(
        object(),
        np.zeros(2),
        [(0, 1), (0, 2)],
        np.array([0, 1]),
        {},
        resumed,
        rank_map={0: 0, 1: 1},
    )
    assert computed == [((0, 2), 1)]
    np.testing.assert_array_equal(results[0]["delta"], np.array([1.0, 1.5]))
    assert set(resumed.manifest["completed_panels"]) == {"0", "1"}


def test_scramble_partial_resume_skips_completed_outer(tmp_path, monkeypatch):
    output = tmp_path / "run"
    identity = _checkpoint_identity()
    store = score_centering_module.PanelCheckpointStore(
        output, identity, resume=False
    )
    calls = []

    class FakeOracle:
        def mu(self, beta):
            return np.zeros(len(beta))

        def _outer_qmc(self, order, seed, beta):
            x = np.zeros((4, 2))
            weights = np.full(4, 0.25)
            return x, 0.0, weights

        def tilt_full(self, beta, draws, seed):
            return np.zeros((draws, 2))

    def fake_summary(
        oracle,
        beta,
        panel,
        observed,
        weights,
        cfg,
        seed,
        *,
        return_row_means=False,
    ):
        calls.append((int(seed), bool(return_row_means)))
        if len(calls) == 2:
            raise RuntimeError("simulated crash")
        dimension = len(beta)
        values = np.ones(dimension)
        return {
            "score_mean": values,
            "conditional_se": np.full(dimension, 0.01),
            "terminal_delta": np.full(dimension, 0.001),
            "query_scramble_se_max": 0.01,
            "query_terminal_delta_max": 0.001,
            "converged": True,
            "row_means": np.ones((len(observed), dimension)),
            "row_final_order": np.array([6, 7]),
            "max_final_order": 7,
            "mean_final_order": 6.5,
        }

    monkeypatch.setattr(score_centering_module, "_conditional_summary", fake_summary)
    payload = {
        "outer_qmc": {"order": 7, "scrambles": 2, "seed": 11},
        "conditional_qmc": {
            "start_order": 5,
            "max_order": 8,
            "scrambles": 2,
            "terminal_delta_atol": 1e-3,
            "terminal_delta_rtol": 0.0,
            "scramble_se_atol": 1e-3,
            "scramble_se_rtol": 0.0,
            "seed": 22,
        },
        "sampling": {"draws": 8, "seed": 33},
    }
    with pytest.raises(RuntimeError, match="simulated crash"):
        score_centering_module._panel_score_mean(
            FakeOracle(),
            np.zeros(2),
            (0, 1),
            payload,
            0,
            checkpoint_store=store,
            panel_index=34,
        )
    assert len(calls) == 2
    resumed = score_centering_module.PanelCheckpointStore(
        output, identity, resume=True
    )
    result = score_centering_module._panel_score_mean(
        FakeOracle(),
        np.zeros(2),
        (0, 1),
        payload,
        0,
        checkpoint_store=resumed,
        panel_index=34,
    )
    assert [item[1] for item in calls[2:]] == [False, True]
    np.testing.assert_allclose(result["delta"], np.ones(2))
    assert resumed.load_scramble_progress(34)["sampling"] is not None


def test_t04_unconverged_scramble_still_runs_sampling(monkeypatch):
    calls = []

    class FakeOracle:
        def mu(self, beta):
            return np.zeros(len(beta))

        def _outer_qmc(self, order, seed, beta):
            return np.zeros((4, 2)), 0.0, np.full(4, 0.25)

        def tilt_full(self, beta, draws, seed):
            # sampling now runs even after T-04 non-convergence (doc-aligned)
            return np.zeros((draws, 2))

    def fake_summary(*args, **kwargs):
        return_row_means = kwargs.get("return_row_means", False)
        calls.append(return_row_means)
        dimension = 2
        return {
            "score_mean": np.ones(dimension),
            "conditional_se": np.full(dimension, 0.01),
            "terminal_delta": np.full(dimension, 0.001),
            "query_scramble_se_max": 0.01,
            "query_terminal_delta_max": 0.001,
            "converged": False,
            "row_means": np.ones(dimension) if return_row_means else None,
            "row_final_order": np.array([16, 16]),
            "max_final_order": 16,
            "mean_final_order": 16.0,
            "n_active_by_order": {"16": 4},
        }

    monkeypatch.setattr(score_centering_module, "_conditional_summary", fake_summary)
    result = score_centering_module._panel_score_mean(
        FakeOracle(),
        np.zeros(2),
        (0, 1),
        {
            "outer_qmc": {"order": 7, "scrambles": 4, "seed": 11},
            "conditional_qmc": {
                "start_order": 5,
                "max_order": 16,
                "scrambles": 8,
                "terminal_delta_atol": 5e-5,
                "terminal_delta_rtol": 0.0,
                "scramble_se_atol": 5e-5,
                "scramble_se_rtol": 0.0,
                "seed": 22,
            },
            "sampling": {"draws": 8, "seed": 33},
        },
        0,
    )
    # Doc-aligned: all 4 scrambles run even though T-04 didn't fully converge,
    # and sampling still runs so the point estimate carries a real SE.
    assert calls == [False, False, False, False, True]
    assert result["conditional_converged"] is False
    assert result["max_final_order"] == 16


@pytest.mark.parametrize(
    "mutate",
    [
        lambda identity: identity.update(config_canonical_sha256="9" * 64),
        lambda identity: identity["base_artifact_sha256"].update(npz="9" * 64),
        lambda identity: identity["source_numerical_kernel_sha256"].update(
            {"src/kernel.py": "9" * 64}
        ),
    ],
)
def test_checkpoint_resume_rejects_identity_mismatch(tmp_path, mutate):
    output = tmp_path / "run"
    identity = _checkpoint_identity()
    score_centering_module.PanelCheckpointStore(output, identity, resume=False)
    changed = copy.deepcopy(identity)
    mutate(changed)
    with pytest.raises(ValueError, match="identity mismatch"):
        score_centering_module.PanelCheckpointStore(output, changed, resume=True)


def test_checkpoint_resume_rejects_corrupt_panel(tmp_path):
    output = tmp_path / "run"
    identity = _checkpoint_identity()
    store = score_centering_module.PanelCheckpointStore(
        output, identity, resume=False
    )
    store.save_panel(3, _checkpoint_result(1.0))
    with (output / "panel_003.npz").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        score_centering_module.PanelCheckpointStore(output, identity, resume=True)


def test_checkpoint_resume_rejects_finalized_directory(tmp_path):
    output = tmp_path / "run"
    identity = _checkpoint_identity()
    score_centering_module.PanelCheckpointStore(output, identity, resume=False)
    (output / "diagnostics.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="finalized immutable"):
        score_centering_module.PanelCheckpointStore(output, identity, resume=True)


def test_checkpoint_manifest_is_atomic_and_schema_is_closed(
    tmp_path, monkeypatch
):
    output = tmp_path / "run"
    replacements = []
    real_replace = score_centering_module.os.replace

    def recording_replace(source, destination):
        replacements.append((source, destination))
        real_replace(source, destination)

    monkeypatch.setattr(score_centering_module.os, "replace", recording_replace)
    identity = _checkpoint_identity()
    score_centering_module.PanelCheckpointStore(output, identity, resume=False)
    manifest_path = output / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(manifest) == {
        "schema_version",
        "identity",
        "completed_panels",
    }
    assert manifest["schema_version"] == score_centering_module.CHECKPOINT_SCHEMA_VERSION
    assert manifest["identity"] == identity
    assert manifest["completed_panels"] == {}
    assert replacements == [
        (output / ".checkpoint_manifest.json.tmp", manifest_path)
    ]
    assert not (output / ".checkpoint_manifest.json.tmp").exists()
