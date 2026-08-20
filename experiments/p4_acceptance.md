# P4 Acceptance Record

- Stage: P4 Synthetic S1 statistical-optimality oracle gate
- Predecessor commit: `da0d180f31c27b174dd562459bfb05ceaea94578`
- Frozen settings: `alpha=1`, exact conditional-score evaluation interface, Uniform SQD/A-OSQD/oracle RR-GID, paired target draws.
- Local smoke: 3 budgets x 3 replications x 3 policies; formal replication target retained as `200` in config.
- Tests: P4 gate rows preserve all three policies and shared target seed.
- Explicitly not implemented here: Discriminative Score OED, VAEAC, final four-policy Fig.1.

P4 formal acceptance: **IN PROGRESS / NOT PASSED**. The implementation now uses a nonzero ESS-calibrated beta direction, panel-specific conditional information, real policy allocations, and a conditional-score final estimator. The required 200 replications, full S1 budget sweep, and J ablation remain outstanding; no formal result is claimed.

Formal configuration is recorded in `configs/p4_formal.yaml`; paired target draws are generated only by `scripts/p4_formal_manifest.py` and are not sampled independently per policy.

Latest numerical audit: nonzero beta and nonzero KL are now observed, but low-budget toy runs show unstable beta estimates under weak information conditioning. P4 remains **NOT PASSED** pending conditioning diagnostics, stable safe allocation, and formal budget-scale replication.

The formal runner is resumable at `scripts/p4_formal_run.py` and writes paired rows to budget-sharded JSONL files; no row is counted until its policy, budget, replication, and diagnostics are present.

Formal integrity run: `scripts/p4_validate.py` reports 3000 rows (5 budgets x 200 replications x 3 policies), zero budget/KL/paired-draw failures, and exact replication counts.

Formal statistical gate: **FAILED**. `results/p4_formal_summary.json` shows mean `B*KL` increasing with budget rather than approaching the frozen theoretical constant, and oracle RR-GID design ratios far from one. This identifies a remaining final-estimator/information-scaling defect. P5 is blocked until the defect is repaired and the full formal sweep is rerun.

The frozen 120-panel oracle preparation is cached in `experiments/p4_prepared_oracle.pkl` and must be hash-checked before formal jobs consume it.
