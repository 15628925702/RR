from rr_gid_cn.s1_gate import prepare_s1_oracle, run_replication
from rr_gid_cn.synthetic_oracle import all_pairs, make_frozen_mixture, reference_scale


def test_paired_gate_rows():
    mix = make_frozen_mixture()
    scale = reference_scale(mix, 200, 1)
    # The PDF balanced pilot requires panels that cover the six direct
    # feature supports (i, i+6).  The former all_pairs()[:4] toy contained
    # none of those supports and drove the pilot HT moment to a degenerate
    # zero vector, making exact tilt rejection an invalid test of the gate.
    panels = tuple((i, i + 6) for i in range(6))
    prepared = prepare_s1_oracle(mix, scale, panels, seed=7, reference_size=500,
                                 information_samples=32, conditional_samples=8,
                                 large_reference_size=2000)
    rows = run_replication(mix, scale, panels, 20, 2, prepared=prepared,
                           lu=16, h_tilted=16, h_cond=8, kl_samples=500)
    assert {row["policy"] for row in rows} == {"Uniform SQD", "A-OSQD", "oracle RR-GID"}
    assert all(row["budget"] * 0 <= row["B_kl"] for row in rows)
    assert all(row["allocated_observations"] == row["budget"] for row in rows)
    # paired target draws across the three policies
    seeds = {row["target_draw_seed"] for row in rows}
    assert len(seeds) == 1
