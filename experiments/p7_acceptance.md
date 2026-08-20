# P7 Acceptance Record

- Stage: P7 Synthetic S2 nonlinearity and generator reuse
- Predecessor commit: `563146bb0f67581eab50a085f89dae5c096c9bf3`
- Frozen alpha sweep: `{0, 0.5, 1.0, 1.5}`; reuse campaigns: `{1, 5, 20, 50}`; budget `8000`.
- Smoke verifies a single generator hash across all reuse campaigns, RR-GID no retraining, and discriminative retraining per campaign.

P7 formal acceptance: **NOT PASSED**. The current artifact is a reuse schema smoke with fixed placeholder metrics; no actual alpha sweep or generator-reuse experiment has been completed.
