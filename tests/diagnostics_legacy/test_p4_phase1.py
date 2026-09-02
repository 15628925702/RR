import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from rr_gid_cn.p4_integrity import (
    compute_pilot_budget,
    load_seed_manifest,
    validate_artifact_metadata,
    validate_expected_grid,
    validate_experiment_mode,
)
from rr_gid_cn.s1_gate import a_optimal_information, design_metrics


def _validator_module():
    spec = importlib.util.spec_from_file_location("p4_validate", Path("scripts/p4_validate.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_t01_a_osqd_embeds_inverse_observed_covariance():
    rng = np.random.default_rng(91)
    covariance = np.array([
        [2.0, 0.8, 0.7],
        [0.8, 1.5, 0.6],
        [0.7, 0.6, 1.2],
    ])
    reference = rng.multivariate_normal(np.zeros(3), covariance, size=20000)
    info = a_optimal_information(reference, ((0, 1),), dimension=3)[0]
    empirical = np.cov(reference, rowvar=False)
    expected = np.linalg.inv(empirical[np.ix_([0, 1], [0, 1])] + 1e-6 * np.eye(2))
    wrong = np.linalg.inv(empirical + 1e-6 * np.eye(3))[:2, :2]
    np.testing.assert_allclose(info[:2, :2], expected, rtol=1e-12, atol=1e-12)
    assert np.linalg.norm(info[:2, :2] - wrong) > 0.05
    assert np.all(info[2] == 0) and np.all(info[:, 2] == 0)


def test_t02_true_design_ratio_is_separate_from_risk():
    fisher = np.eye(2)
    information = np.array([np.diag([2.0, 0.4]), np.diag([0.4, 2.0])])
    oracle = np.array([0.5, 0.5])
    same = design_metrics(fisher, information, oracle, oracle, [1, 1], [4, 4])
    changed = design_metrics(fisher, information, oracle, [0.8, 0.2], [1, 1], [7, 1])
    assert same["design_ratio_main"] == pytest.approx(1.0)
    assert changed["design_ratio_main"] > 1.0
    assert "risk_ratio_raw" not in changed


def test_t03_exact_score_mode_guard():
    finite = {
        "experiment_mode": "finite_lu_rejection",
        "conditional_method": "rejection",
        "sandwich_benchmark": True,
        "exact_observed_score": True,
    }
    with pytest.raises(ValueError, match="cannot be labelled"):
        validate_experiment_mode(finite, formal=True)
    finite["exact_observed_score"] = False
    validate_experiment_mode(finite, formal=True)
    with pytest.raises(ValueError, match="QMC"):
        validate_experiment_mode({
            "experiment_mode": "oracle_gold_qmc",
            "conditional_method": "rejection",
            "qmc_error_diagnostics": True,
        }, formal=True)
    with pytest.raises(ValueError, match="use_oracle_H"):
        validate_experiment_mode({
            "experiment_mode": "validated_fixed_qmc",
            "conditional_method": "qmc",
            "qmc_error_diagnostics": True,
            "use_oracle_H": False,
        }, formal=True)


@pytest.mark.parametrize("field", [
    "mixture_seed", "alpha", "reference_size", "parameter_sha256",
    "information_method", "code_commit",
])
def test_t06_artifact_mismatch_is_rejected(field):
    expected = {
        "mixture_seed": 2026, "alpha": 1.0, "reference_size": 50000,
        "parameter_sha256": "beta-and-scale", "information_method": "cross",
        "code_commit": "abc",
    }
    actual = dict(expected)
    actual[field] = "corrupted"
    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_artifact_metadata(actual, expected)


def test_t07_manifest_duplicate_missing_and_seed_identity(tmp_path):
    rows = [{
        "budget": 2000, "replication": 0, "replication_seed": 10,
        "target_draw_seed": 13, "pilot_or_design_seed": 21,
        "score_seed_root": 1010, "information_seed_root": 7010,
    }]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    manifest, first_hash = load_seed_manifest(path)
    assert manifest[(2000, 0)]["target_draw_seed"] == 13
    _, second_hash = load_seed_manifest(path)
    assert first_hash == second_hash
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(json.dumps({"rows": rows + rows}), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_seed_manifest(duplicate)
    missing = copy.deepcopy(rows)
    del missing[0]["score_seed_root"]
    path.write_text(json.dumps({"rows": missing}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing seed fields"):
        load_seed_manifest(path)


def test_t08_expected_cartesian_grid_corruptions_fail():
    budgets, reps, policies = [100, 200], [0, 1], ["U", "A", "O"]
    rows = [
        {"budget": b, "replication": r, "policy": p}
        for b in budgets for r in reps for p in policies
    ]
    validate_expected_grid(rows, budgets, reps, policies)
    for predicate in (
        lambda row: row["budget"] != 200,
        lambda row: row["replication"] != 1,
        lambda row: row["policy"] != "A",
    ):
        with pytest.raises(ValueError, match="grid mismatch"):
            validate_expected_grid([row for row in rows if predicate(row)], budgets, reps, policies)
    with pytest.raises(ValueError, match="duplicates"):
        validate_expected_grid(rows + [dict(rows[0])], budgets, reps, policies)


def test_validator_catches_wrong_artifact_method_and_duplicate():
    validator = _validator_module()
    base = {
        "budget": 100, "replication": 0, "allocated_observations": 100,
        "kl_raw": 0.1, "B_kl_raw": 10.0, "conditional_method": "qmc",
        "experiment_mode": "validated_fixed_qmc", "artifact_sha256": "artifact",
        "config_sha256": "config", "source_sha256": {"code": "hash"},
        "target_draw_seed": 13, "design_ratio_main": 1.0, "fw_gap": 1e-8,
        "pilot_schedule": {"kind": "power"}, "pilot_budget": 6,
        "pilot_counts": [3, 3],
    }
    rows = [{**base, "policy": policy} for policy in ("U", "A", "O")]
    kwargs = dict(
        budgets=[100], replications=[0], policies=["U", "A", "O"],
        conditional_method="qmc", experiment_mode="validated_fixed_qmc",
        integration_tolerance=1e-8, design_ratio_tolerance=1e-10,
        fw_tolerance=1e-6, artifact_sha256="artifact", config_sha256="config",
        source_sha256={"code": "hash"},
        manifest={(100, 0): {"target_draw_seed": 13}},
    )
    assert validator.validate_rows(rows, **kwargs) == []
    for corrupt in (
        [{**row, "artifact_sha256": "wrong"} for row in rows],
        [{**row, "conditional_method": "rejection"} for row in rows],
        rows + [dict(rows[0])],
    ):
        assert validator.validate_rows(corrupt, **kwargs)


def test_pilot_schedule_changes_actual_count():
    schedule = {
        "kind": "power", "exponent": 1 / 3, "multiplier": 10,
        "max_fraction": 1.0, "min_per_support": 1, "rounding_rule": "ceil",
    }
    larger = {**schedule, "multiplier": 12}
    assert compute_pilot_budget(larger, 8000) > compute_pilot_budget(schedule, 8000)
