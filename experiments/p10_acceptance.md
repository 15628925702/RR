# P10 Acceptance Record

- Stage: P10 Gas R2 natural drift
- Predecessor commit: `ea9d923230e6811033a14b1f13f6a2d779a306b3`
- Smoke covers all three campaigns, all four policies, all four budgets, and records campaign-pool isolation.
- Every row includes projection loss, held-out moment RMSE, C2ST AUC, ESS, conditional acceptance, FW gap, and minimum information eigenvalue.
- Full-test exposure to acquisition is explicitly false in every row.

P10 formal acceptance: **NOT PASSED**. The artifact contains 144 schema rows (3 smoke replications per budget/campaign/policy), not the required 2,400 paired records; metrics are placeholders and no natural-drift evaluation has been completed.
