from rr_gid_cn.s1_gate import prepare_s1_oracle, run_replication
from rr_gid_cn.synthetic_oracle import all_pairs, make_frozen_mixture, reference_scale


def test_paired_gate_rows():
    mix = make_frozen_mixture()
    scale = reference_scale(mix, 200, 1)
    panels = all_pairs()[:4]
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
