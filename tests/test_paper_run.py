from __future__ import annotations

import numpy as np

from rr_gid_cn.paper_run import run_paper_replication
from rr_gid_cn.s1_gate import prepare_s1_oracle
from rr_gid_cn.synthetic_oracle import all_pairs, make_frozen_mixture, reference_scale


def _tiny_prepared():
    mixture = make_frozen_mixture(seed=2026, alpha=1.0)
    scale = reference_scale(mixture, n=300, seed=7)
    panels = all_pairs()[:12]
    prepared = prepare_s1_oracle(
        mixture, scale, panels, seed=2026,
        reference_size=256, large_reference_size=256,
        information_samples=16, conditional_samples=8,
    )
    return mixture, scale, panels, prepared


def test_t6_paired_target_and_budget():
    mixture, scale, panels, prepared = _tiny_prepared()
    rows = run_paper_replication(
        mixture, scale, panels, 256, 202609256, prepared,
        scoring_steps=1,
        score_qmc_order=4,
        information_qmc_order=4,
        information_outer_rows=8,
        policies=("Uniform SQD", "A-OSQD", "RR-GID"),
        seed_manifest_entry={
            "replication": 0,
            "replication_seed": 202609256,
            "target_draw_seed": 202609259,
            "pilot_or_design_seed": 202609267,
            "score_seed_root": 202609260,
            "information_seed_root": 202616256,
        },
        validation_size=64,
        mlp_steps=5,
    )
    hashes = {row["target_draw_sha256"] for row in rows}
    assert len(hashes) == 1
    for row in rows:
        assert row["allocated_observations"] <= row["budget"]
        assert row["primary_metric"] == "risk_ratio_raw"


def test_t7_end_to_end_smoke():
    mixture, scale, panels, prepared = _tiny_prepared()
    rows = run_paper_replication(
        mixture, scale, panels, 256, 11, prepared,
        scoring_steps=1,
        score_qmc_order=4,
        information_qmc_order=4,
        information_outer_rows=8,
        policies=("Uniform SQD", "A-OSQD", "Discriminative Score OED", "RR-GID"),
        validation_size=32,
        mlp_steps=5,
    )
    assert {row["method"] for row in rows} == {
        "Uniform SQD", "A-OSQD", "Discriminative Score OED", "RR-GID",
    }
    assert all(np.isfinite(row["risk_ratio"]) for row in rows)
