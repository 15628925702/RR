# P4 Acceptance Record

- Stage: P4 Synthetic S1 statistical-optimality oracle gate
- Predecessor commits: `f303a24` (exact TiltCond + HT pilot) → `9d3213b` (stable PDF-compliant estimator) → `8949c23` (diagnostics/options) → `e80c7b8` (test/smoke adapt)
- Frozen settings: `alpha=1`, exact conditional-score evaluation interface, Uniform SQD / A-OSQD / oracle RR-GID, paired target draws, `J=2`, `b_B=ceil(10 B^{1/3})`, 200 replications per budget.
- Formal config: `configs/p4_formal.yaml` (pilot norm cap 2.0, `lu=128`, `h_tilted=128`, `h_cond=32`, large reference pool 200k).
- Formal integrity run: `scripts/p4_validate.py` reports **3000 rows (5 budgets x 200 reps x 3 policies), zero budget/paired-draw/KL failures, exact replication counts.**

## Fixes over the previous failed gate

The previous formal gate FAILED with `B*KL` growing with budget (151 -> 1795)
and oracle RR-GID design ratios far from one. Root-cause chain:

1. Pilot HT noise was amplified through the tanh saturation region, so the
   pilot beta diverged to the theta boundary.
2. That made the accept-reject conditional sampler essentially non-terminating
   (the reported throughput bottleneck) and the scoring updates diverged.

Repairs (all within the frozen PDF design):
- `solve_pilot_beta` caps `||beta||` (Theta boundary; PDF leaves it open).
- `final_rr_estimator` now follows PDF Alg.2 step 6: H is re-estimated by
  cross-completion `I_S(beta^j)` at each scoring step (previously the fixed
  beta_true oracle info, a PDF deviation); U uses exact-conditional-oracle
  observed scores via importance weighting on Q0 conditionals (PDF Sec. 4.1
  permits importance proposals; cross-completion removes self-normalized bias);
  projected full step `beta + H^{-1} U`.
- `mu_beta` is estimated on a 200k Q0 pool (the 50k pool carried ~0.01
  systematic bias amplified by `M(p)^{-1}` into large `B*KL` shifts).
- KL uses a large direct accept-reject sample of the target mean.
- A-OSQD baseline fixed: panel information is now the (S,S) block of the full
  reference precision (16x16), matching the PDF A-optimal formulation; the
  previous 2x2 covariance version collapsed to one panel (Phi ~ 1e8).
- `scripts/p4_formal_run.py`: resumable, `--rep-range` sharding, `--scoring-steps`
  J-ablation with J-suffixed outputs.

## Theory constants (oracle info at beta_true)

Uniform SQD 1/2 Phi = 64.8 ; A-OSQD = 95.8 ; oracle RR-GID = 31.9.

## Formal results (`results/p4_formal_summary.json`)

| B     | oracle RR-GID | Uniform SQD | A-OSQD   |
|-------|---------------|-------------|----------|
| 2000  | 47.5 ± 2.5    | 69.1 ± 2.3  | 120.8 ± 4.3 |
| 4000  | 58.3 ± 3.8    | 86.3 ± 2.9  | 171.4 ± 8.5 |
| 8000  | 74.4 ± 3.8    | 116.1 ± 4.2 | 278.3 ± 14.3 |
| 16000 | 110.7 ± 8.3   | 161.4 ± 5.1 | 460.1 ± 29.2 |
| 32000 | 236.0 ± 49.9  | 246.9 ± 7.7 | 890.3 ± 71.7 |

J ablation (B=8000): J=0 -> 1010.9 (all three policies identical, pilot-only
as expected); J=1 -> oracle 198 / Uniform 220 / A-OSQD 475; J=2 -> main table.
Monotone improvement with J and correct ordering.

## Gate assessment

**Formal integrity: PASSED** (3000 rows, zero budget/pairing/KL failures).
**Policy ordering and J trend: PASSED** (oracle RR-GID lowest at B<=16000,
J improves monotonically, J=0 degenerates to pilot-only).
**Statistical convergence gate: NOT PASSED / open.** `B*KL` still grows with
budget instead of plateauing at `1/2 Phi` (oracle RR-GID 47.5 -> 236.0 vs
theory 31.9). This is diagnosed as an intrinsic finite-sample property of the
PDF synthetic setting: after reference scaling the Fisher matrix has a weak
direction (lambda_min ~ 0.008; M(p) min eig ~ 0.0014), the pilot HT moment
noise is amplified there, and the pilot `e_beta` carries a Jensen bias that the
J=2 scoring reduces but does not remove. The resulting `B * (bias^T F bias)/2`
term grows with B. This is not an implementation deviation from PDF Algorithm 2
(the estimator now follows Alg. 2 step 6 exactly); it is a finite-B higher-order
correction that the asymptotic theory in PDF Sec. 5 does not capture.

Explicitly not implemented here: Discriminative Score OED, VAEAC, final
four-policy Fig.1. P5 remains blocked on the statistical convergence gate per
PROJECT_PLAN until a decision on the weak-direction finite-sample correction is
recorded; the implementation and full paired dataset are ready to rerun the gate
at any approved configuration.
