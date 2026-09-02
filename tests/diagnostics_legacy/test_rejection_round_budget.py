from rr_gid_cn.synthetic_oracle import _rejection_round_budget


def test_rejection_round_budget_keeps_floor_when_nothing_accepted():
    assert _rejection_round_budget(
        round_index=100,
        remaining_samples=8000,
        proposals_total=10_000_000,
        accepted_raw_total=0,
    ) == 20000
    assert _rejection_round_budget(
        round_index=20000,
        remaining_samples=8000,
        proposals_total=80_000_000,
        accepted_raw_total=0,
    ) == 20000


def test_rejection_round_budget_extends_with_observed_acceptance():
    budget = _rejection_round_budget(
        round_index=20000,
        remaining_samples=415_000,
        proposals_total=80_000_000_000,
        accepted_raw_total=80_000,
        safety=8.0,
        hard_cap=200_000,
    )
    assert budget > 20000
    assert budget <= 200_000


def test_rejection_round_budget_hard_cap():
    budget = _rejection_round_budget(
        round_index=20000,
        remaining_samples=10_000_000,
        proposals_total=1_000,
        accepted_raw_total=1,
        hard_cap=50_000,
    )
    assert budget == 50_000
