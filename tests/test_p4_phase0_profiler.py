import json
from pathlib import Path

from rr_gid_cn.s1_gate import prepare_s1_oracle, run_replication
from rr_gid_cn.synthetic_oracle import make_frozen_mixture, reference_scale


def test_phase0_oracle_row_has_profiler_fields():
    mix = make_frozen_mixture()
    scale = reference_scale(mix, 200, 1)
    panels = tuple((i, i + 6) for i in range(6))
    prepared = prepare_s1_oracle(
        mix, scale, panels, seed=7, reference_size=400,
        information_samples=16, conditional_samples=4, large_reference_size=800,
    )
    rows = run_replication(
        mix, scale, panels, 120, 11, prepared=prepared,
        lu=8, h_tilted=8, h_cond=4, kl_samples=200,
        scoring_steps=1, policies=["oracle RR-GID"],
    )
    assert len(rows) == 1
    row = rows[0]
    runtime = row["runtime"]
    required = {
        "time_target_sampling", "time_pilot_build", "time_pilot_solve",
        "time_design_information", "time_fw", "time_mu", "time_H",
        "time_score", "time_linear_solve", "time_kl", "time_total",
        "time_attributed", "attributed_fraction", "device", "memory",
        "active_panels", "conditional_acceptance",
    }
    assert required <= set(runtime)
    assert runtime["attributed_fraction"] > 0.0
    assert row["workload"]["conditional_requested"] >= 120 * 8
    assert row["workload"]["conditional_proposals"] >= row["workload"]["conditional_requested"]
    assert runtime["conditional_acceptance"]["n"] >= 1
    assert row["target_draw_seed"] == 14


def test_phase0_config_is_not_formal():
    text = Path("configs/p4_phase0_local.yaml").read_text(encoding="utf-8")
    assert "not_formal: true" in text
    assert "phase0_profile" in text
    assert "formal" not in json.dumps({"experiment_mode": "phase0_profile"})
