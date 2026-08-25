# P4 Active-Panel GPU Cache Diagnostic

Status: diagnostic only; this record is not a P4 S1 acceptance record.

The exact-adaptive conditional mean path was optimized without changing the
PDF estimator or the nested-QMC certificate. Panel-specific conditional
Gaussian tensors are cached per CUDA device, QMC normal prefixes use a bounded
LRU cache, and the integration is processed in observation chunks with
component-wise streaming accumulation. The normalized numerator/denominator
ratio, Sobol seed, mixture posterior, and adaptive tolerance are unchanged.

## Fixed-seed verification

- Server: `/root/RR_GID_CN/current`, eight A800 GPUs, one process per GPU.
- Seed rule: `202600000 + 2000*1000 + replication`, replications `0..7`.
- Budget: `B=2000`; policies: oracle RR-GID only.
- Completed diagnostic configuration: `h_tilted=128`, `h_cond=32`,
  `start_order=12`, `max_order=16`, `atol=1e-4`, `rtol=1e-3`.
- All eight JSON artifacts were produced without OOM or exception.
- Per-rep elapsed time: 27.32--28.48 s; mean 28.17 s.
- `B·KL_raw`: 12.968--51.193; mean 38.398.
- No negative-KL or budget/shape failure occurred.

The values are a completion and numerical-stability check only. Their spread
and mean do not satisfy the high-precision S1 theory gate and must not be used
as formal P4 results.

## High-resolution risk check

The same implementation at `h_tilted=1024`, `max_order=18` used about 10.9 GB
GPU memory after chunking and no longer hit OOM, but a single `B=2000` rep did
not finish within three minutes. A medium check (`h_tilted=256`,
`max_order=16`) completed in 39.09 s with `B·KL_raw=43.8965`. This establishes
that the remaining obstacle is computation time at the formal resolution, not
an unresolved memory or numerical exception.

## Hashes

Server SHA-256 at the diagnostic run:

- `src/rr_gid_cn/synthetic_oracle.py`: `5ba5da11053344df31bb110a27ddff7f4fa10f746eaff4a7552fd1eb24517c`
- `scripts/p4_single_probe.py`: `980c86df8c092761f9c6cd1ff8453ca49b3d75f1b33e0932850a4a4f7792d255`
- `configs/p4_formal.yaml`: `19639a3e2dd4946799e692ae84bd7adc9326f0a8e98ae2574b1e9f4e5b68348e`
- `experiments/p4_prepared_oracle_hp.pkl`: `8db4a1a2952636ea7d89bdc6c6ba599ddc2b0348b5e639fd10c3e30e0c342304`

The source change is committed separately from formal P4 results; old results
remain untouched.

## Next gate

Before restarting the five-budget formal P4, run a paired high-resolution
calibration with a documented runtime budget and verify the nested-QMC error
certificate and `B·KL` theory gate. This diagnostic alone does not authorize
P5.
