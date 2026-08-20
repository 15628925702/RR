# P8 Acceptance Record

- Stage: P8 Gas Sensor preprocessing
- Predecessor commit: `da74a1e4d5d13e74ee6173549cdad715f4a2523e`
- Fixed schema: 128 raw features, 16 sensor blocks x 8 features, 120 arbitrary sensor pairs, PC1 per sensor, 16 bounded relative features.
- Smoke uses a deterministic synthetic fixed subset only; no UCI data or target data is committed.
- Tests: 20 total project tests pass; dimensions, panel count, PC1 sign alignment, and bounded feature map pass.

P8 local schema gate: **PASS**. External UCI archive registration remains a data acquisition step and does not alter preprocessing.

