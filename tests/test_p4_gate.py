from rr_gid_cn.s1_gate import run_replication
from rr_gid_cn.synthetic_oracle import all_pairs, make_frozen_mixture, reference_scale


def test_paired_gate_rows():
    mix = make_frozen_mixture()
    scale = reference_scale(mix, 200, 1)
    rows = run_replication(mix, scale, all_pairs()[:4], 20, 2, reference_size=100)
    assert {row["policy"] for row in rows} == {"Uniform SQD", "A-OSQD", "oracle RR-GID"}
    assert all(row["budget"] * 0 <= row["B_kl"] for row in rows)

