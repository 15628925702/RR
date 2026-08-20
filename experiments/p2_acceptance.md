# P2 Acceptance Record

- Stage: P2 Uniform SQD, A-OSQD, and oracle RR-GID
- Predecessor commit: `d658b034af04d653059f8685197f2f455e27d267` (`P1-synthetic-oracle`)
- Frozen PDF SHA-256: `D223571A9D8F339EE0C5AAD05AC0C30DC2F5669DE4C58006F0C6B3CDB6D6880E`
- Implemented: uniform allocation, covariance-based A-OSQD information, conditional-information oracle, cost-share rounding, budget accounting, and cost-aware Frank-Wolfe.
- Explicitly not implemented: VAEAC, discriminative score OED, Fisher scoring updates, Gas data.

## Checks

- Unit tests: 11 passed.
- Uniform probabilities sum to one; rounded counts are nonnegative integers and never exceed budget.
- Smoke budget: `B=100`, spent budget `100`.
- Smoke panels: 6 equal-cost panels; the full frozen panel library remains 120 panels in P1.
- Fisher estimate minimum eigenvalue: `0.01875 > 0`.
- Frank-Wolfe smoke certificate: gap `0.10849 <= tau_FW=0.2`.
- Fixed seed summary: `results/p2_policy_summary.json`, with summary hash recorded in the artifact.

## Numerical note

The smoke uses deliberately small conditional Monte Carlo sizes. The configured `tau_FW=0.2` is therefore a local numerical gate for this smoke only; formal runs must set their own stricter tolerance in the versioned stage configuration after MC sizes are selected. No research question, policy, panel family, cost, or metric is changed.

P2 acceptance: **PASS**. Entry to P3 is permitted.

