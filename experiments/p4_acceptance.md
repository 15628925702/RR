# P4 Acceptance Record

- Stage: P4 Synthetic S1 statistical-optimality oracle gate
- Predecessor commit: `da0d180f31c27b174dd562459bfb05ceaea94578`
- Frozen settings: `alpha=1`, exact conditional-score evaluation interface, Uniform SQD/A-OSQD/oracle RR-GID, paired target draws.
- Local smoke: 3 budgets x 3 replications x 3 policies; formal replication target retained as `200` in config.
- Tests: P4 gate rows preserve all three policies and shared target seed.
- Explicitly not implemented here: Discriminative Score OED, VAEAC, final four-policy Fig.1.

P4 formal acceptance: **NOT PASSED**. This commit contains only a local implementation/smoke gate; the required 200 replications, nonzero frozen beta direction, exact panel-specific information, allocation effect, and J ablation remain outstanding.
