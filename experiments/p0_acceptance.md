# P0 Acceptance Record

- Stage: P0
- Research specification input: `RR_GID_CN.pdf`
- PDF SHA-256: `D223571A9D8F339EE0C5AAD05AC0C30DC2F5669DE4C58006F0C6B3CDB6D6880E`
- Python decision: Python 3.10 is the only installed version in the supported 3.10/3.11 range; this is the explicit environment fallback to the plan's suggested 3.11.
- Config: `configs/p0_smoke.yaml`
- Config SHA-256: recorded in `experiments/p0_run_manifest.json`
- Seed: 2026 (all P0 smoke randomness initialized through the package entry point)
- Device: `auto` resolved to CPU on this host; CUDA is never required by P0.
- Data isolation: no data downloaded; no checkpoint or target artifact created.
- Budget isolation: P0 has no acquisition or experimental budget; no research-stage budget was consumed.

## Checks

- Unit tests: `4 passed in 0.19s`
- Numerical/reproducibility: deterministic seed initialization covers Python and NumPy when available; same-seed toy streams match and different seeds differ.
- CLI: `python -m rr_gid_cn --help` succeeds.
- Empty pipeline: `python -m rr_gid_cn --config configs/p0_smoke.yaml --write-manifest` succeeds in under one minute and writes the run manifest.
- Environment failure classification: the initial missing `PyYAML`/`pytest` packages were installed from the declared environment requirements; no code workaround was used.

## Entry decision

P0 acceptance: **PASS**. The worktree must be clean at the recorded `P0-init` commit before entering P1.

