# P4 diagnostic checkpoint (not formal acceptance)

Date: 2026-08-24

The formal P4 run is paused before B=16000/B=32000 completion. Existing formal
JSONL is preserved; no result is promoted.

Findings from fixed-seed diagnostics:

- The formal prepared artifact is `reference=50000`, `reference_large=200000`,
  with `||beta_true||=1.657346`. The older `p4_prepared_oracle_fix1_debug.pkl`
  is diagnostic-sized (`reference=500`, `reference_large=2000`) and is now
  rejected by `p4_formal_run.py`.
- Solving the moment equation with the exact target moment recovers beta to
  approximately `6.3e-9`; the beta solver is not the root cause.
- Balanced HT pilot noise remains large at the PDF pilot budget
  `ceil(10 B^(1/3))`; single-seed beta errors are roughly 1.4--2.0 in the
  tested budgets. Larger diagnostic pilots reduce this in some seeds but are
  not the frozen PDF experiment and are not formal results.
- Replacing current-step H with frozen oracle H did not improve the small
  diagnostic and often increased Bregman loss. That historical branch is not
  accepted as a fix.
- Fixed scoring step sizes 0.25, 0.5, 0.75, 1.0 had no consistent winner.

Conclusion: P4 is not accepted. Do not resume 200-replication budgets until a
PDF-compliant estimator change is tested on small fixed-seed diagnostics and
shows stable `B*KL` behavior. Diagnostic outputs remain under `results/` with
diagnostic names and are excluded from formal summaries.

## Exact-score enforcement update

The formal runner now rejects fixed `--lu` overrides and derives
`L_U(B)=ceil(lu_scale * B)` from `configs/p4_formal.yaml`. This implements the
PDF alternative `L_{U,B}->infinity`; fixed-LU runs require `--diagnostic` and
cannot be treated as S1. The formal default prepared artifact is the
50,000/200,000 reference-pool artifact, and tiny debug artifacts are rejected.

A B=100 single-policy probe with `L_U` in `{25,50,100}` completed, confirming
the budget-growing path runs, but it is not a convergence gate. No formal
replications were started after this change.
