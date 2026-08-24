# P4 S1 LU/H convergence diagnostic record

Status: **NOT PASSED; formal five-budget run remains blocked.**

The diagnostic used the frozen prepared Synthetic artifact (`reference=50,000`,
`reference_large=200,000`) and paired target seeds
`202600000 + B*1000 + replication`. No diagnostic output was merged into any
formal P4 JSONL.

## LU paired probe

Run `results/p4_lu_diag_paired_run1/` used `B={200,400,800}`, LU factors
`{0.5,1,2}`, and six paired seeds, with `h_tilted=512`, `h_cond=16`. It was
stopped after 34 rows because the fixed-H configuration is known not to meet
the PDF information-error condition; its partial summary is retained. For
the completed cells:

| B | LU/B | n | mean B*K(LU) | SE |
|---:|---:|---:|---:|---:|
| 200 | 0.5 | 6 | 22.13 | 4.14 |
| 200 | 1.0 | 6 | 22.57 | 3.41 |
| 200 | 2.0 | 6 | 21.86 | 3.22 |
| 400 | 0.5 | 6 | 39.42 | 5.18 |
| 400 | 1.0 | 5 | 38.82 | 7.29 |
| 400 | 2.0 | 5 | 34.99 | 6.11 |

The LU effect is non-monotone and smaller than seed variance. Therefore
`LU=B` is not a demonstrated convergence choice, although it is a valid
candidate budget-growing route under the PDF.

## H-growth probe

`results/p4_h_growth_run1/` completed two paired seeds for
`(h_tilted,h_cond)=(256,8)` and `(1024,32)`, using `LU=B` and direct tilted
moment diagnostics. Means were 34.78 vs 33.59 at B=200 and 34.03 vs 27.09 at
B=400; the latter CI is very wide. A repeat in
`results/p4_h_growth_run2/` was stopped after 13 rows because the final high-H
cell was disproportionately slow. Its partial summary is retained.

The target horizontal line is `1/2 Phi = 31.8684677083`. These small samples
show a possible H improvement but do not establish a stable mean/CI or raw
Bregman non-negativity. `s1_gate.run_replication` now records both
`kl_raw/B_kl_raw` and compatibility-clipped `kl/B_kl`; acceptance must inspect
the raw fields and may not silently treat clipping as a proof.

## Blocking findings

1. `final_rr_estimator` uses finite conditional completions, so this route is
   budget-growing LU, not an exact conditional expectation. A formal S1 gate
   must demonstrate `LU(B)->infinity` with paired statistics.
2. The update-stage `I_hat` currently has fixed Monte Carlo sizes in the formal
   config. PDF Algorithm 2 requires its operator error to vanish; fixed
   `h_tilted=256,h_cond=8` is not an asymptotic implementation. H-growth or a
   verified exact information implementation is required.
3. The prepared/reference moment pools are finite and must be shown to have
   negligible score/moment bias at the formal budgets.

Until these three checks pass with the required paired CI, P4 formal
replications and all dependent P5-P12 stages remain blocked.
