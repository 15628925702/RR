# P1 Acceptance Record

- Stage: P1 Synthetic oracle pipeline
- Predecessor commit: `2c5ba1f099f041a7d026064fce21f39514e14d20` (`P0-init`)
- Frozen PDF SHA-256: `D223571A9D8F339EE0C5AAD05AC0C30DC2F5669DE4C58006F0C6B3CDB6D6880E`
- Scope: nonlinear four-component latent GMM, reversible sinh warp, exact Gaussian-mixture conditional sampler, frozen 12-dimensional feature map, and oracle Fisher/KL primitives.
- Explicitly not implemented: VAEAC, allocation policies, Frank-Wolfe, Fisher updates, Gas data.

## Checks

- Unit tests: 8 passed.
- Warp/inverse error: `<1e-10` in the deterministic test.
- Panel library: 120 coordinate pairs.
- Feature support/dimension: 12 features; outputs bounded in `[-1,1]`.
- Fisher PSD: empirical covariance minimum eigenvalue is checked against `-1e-10`.
- Conditional oracle: analytic latent Gaussian-mixture conditional moments are compared with independent conditional samples in `scripts/p1_smoke.py`.
- Reproducibility: fixed mixture seed and fixed sampling seed produce byte-identical arrays.
- Smoke metrics: maximum conditional mean z-score `1.73 < 5.0`; relative conditional covariance Frobenius error `0.0109 < 0.15`; minimum Fisher eigenvalue `0.0228`.
- Resolved config: `configs/p1_smoke.yaml`; its SHA-256 is recorded with the stage commit and summary artifact.

## Engineering resolutions derived from the PDF

- Reference scales are estimated from an independent Q0 pool, exactly as specified; the pool size and seed are configuration fields. The P1 smoke uses 6,000 samples and seed 2026 solely for local numerical validation. Formal experiment pool sizes remain controlled by their stage configurations.
- Conditional moment tolerances use a five-standard-error mean check and 15% relative Frobenius covariance error for the 20,000-sample smoke. These are numerical acceptance tolerances, not changes to the model, estimator, or experiment budget.

P1 acceptance: **PASS** for the implemented oracle and local smoke gate. No PDF-defined research choice was changed.
