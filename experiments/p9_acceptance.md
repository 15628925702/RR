# P9 Acceptance Record

- Stage: P9 Gas VAEAC and R1
- Predecessor commit: `91be46b3eb4eaee35b9e1957ae061fb57e399fd9`
- Smoke uses independent Gas reference records, four paired policies, budgets `{400,800,1600,3200}`, and J=2 metadata.
- Generator hash, target draw seed, ESS fraction, budget, replication, and policy are present in every row.
- No natural drift or full-test evaluation data is used.

P9 formal acceptance: **NOT PASSED**. The current artifact uses random synthetic 300x128 records and an empirical sampler, not real Gas data plus a trained/frozen Gas VAEAC; formal R1 replications are outstanding.
