"""P5 formal runner: complete the four-policy S1 Fig.1.

Note: this is a scripts/ entry point. The project package is installed under
``src/``; running directly from a clone needs the repo root (or ``src/``) on
``PYTHONPATH``. We prepend it here so the same command works on any host.

P4 already ran Uniform SQD / A-OSQD / oracle RR-GID (results/p4_formal_<B>.jsonl,
frozen seed scheme ``seed = 202600000 + budget*1000 + replication``). P5 appends
the two acquisition policies the final Fig.1 needs that P4 did not run:

- **Discriminative Score OED** (PDF Sec. 6): mask-conditioned MLP retrained per
  campaign, tilted-reference Fisher ``F_hat``, Frank-Wolfe design.
- **learned RR-GID** (the formal generator-aware RR-GID): cross-completion panel
  information from the frozen P6 VAEAC generator, Frank-Wolfe design, then the
  shared PDF Algorithm 2 pilot + J-step Fisher scoring final estimator.

The seed scheme is identical to P4's, so every replication pairs the same target
draws across the final four policies. Rows are appended atomically and the runner
is resumable and shardable with ``--rep-range``.

Output: ``results/p5_four_<budget>.jsonl`` (rows contain policy / budget /
replication / kl / B_kl / design_ratio / diagnostics, same schema as P4).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

_here = Path(__file__).resolve().parent
_src = _here.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from rr_gid_cn.s1_gate import run_replication
from rr_gid_cn.synthetic_oracle import all_pairs, make_frozen_mixture, reference_scale, sample_full
from rr_gid_cn.vaeac import VAEACGenerator, load_vaeac_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--config", type=Path, default=Path("configs/p4_formal.yaml"))
    ap.add_argument("--max-replications", type=int, default=None)
    ap.add_argument("--rep-range", type=int, nargs=2, default=None)
    ap.add_argument("--gen-ckpt", type=Path, default=Path("experiments/p6_vaeac_synthetic.pt"))
    ap.add_argument("--mlp-steps", type=int, default=None)
    ap.add_argument("--mlp-hidden", type=int, default=None)
    ap.add_argument("--disc-reference-size", type=int, default=None)
    ap.add_argument("--gen-info-tilted", type=int, default=None)
    ap.add_argument("--gen-info-cond", type=int, default=None)
    args = ap.parse_args()

    with args.config.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)["p4"]
    budgets = (args.budget,) if args.budget is not None else tuple(cfg["budgets"])
    replications = args.max_replications or int(cfg["replications"])

    mixture = make_frozen_mixture(seed=2026, alpha=1.0)
    scale = reference_scale(mixture, 6000, 2026)
    panels = all_pairs()

    prepared_path = Path("experiments/p4_prepared_oracle.pkl")
    with prepared_path.open("rb") as stream:
        prepared = pickle.load(stream)
    if "reference_large" not in prepared:
        prepared["reference_large"] = sample_full(mixture, cfg["large_reference_size"], 2026 + 12345)

    model, ckpt = load_vaeac_checkpoint(args.gen_ckpt, device="cuda", expected_dim=16)
    gen = VAEACGenerator(model, ckpt.get("scale", scale),
                         alpha=float(ckpt.get("alpha", 1.0)), device="cuda")

    policies = ["Discriminative Score OED", "learned RR-GID"]
    kwargs = dict(
        lu=cfg["lu"], h_tilted=cfg["h_tilted"], h_cond=cfg["h_cond"],
        pilot_norm_cap=cfg["pilot_norm_cap"], kl_samples=cfg["kl_samples"],
        scoring_steps=int(cfg["scoring_steps"]), generator=gen,
        theta_norm_cap=cfg.get("theta_norm_cap"), theta_l1_cap=cfg.get("theta_l1_cap"),
        scoring_step_size=cfg.get("scoring_step_size", 1.0),
        gen_info_tilted=args.gen_info_tilted or int(cfg.get("gen_info_tilted", 256)),
        gen_info_cond=args.gen_info_cond or int(cfg.get("gen_info_cond", 32)),
        mlp_hidden=args.mlp_hidden or 64, mlp_steps=args.mlp_steps or 200,
        disc_reference_size=args.disc_reference_size,
    )

    out = Path("results")
    for budget in budgets:
        fp = out / f"p5_four_{budget}.jsonl"
        done: set[tuple[int, str]] = set()
        if fp.exists():
            for line in fp.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    done.add((row["replication"], row["policy"]))
        start, end = args.rep_range if args.rep_range else (0, replications)
        with fp.open("a", encoding="utf-8") as stream:
            for replication in range(start, min(end, replications)):
                if all((replication, pol) in done for pol in policies):
                    continue
                seed = 202600000 + budget * 1000 + replication
                rows = run_replication(mixture, scale, panels, budget, seed,
                                       prepared=prepared, policies=policies, **kwargs)
                for row in rows:
                    row["replication"] = replication
                    if (replication, row["policy"]) not in done:
                        stream.write(json.dumps(row, sort_keys=True) + "\n")
                        done.add((replication, row["policy"]))
                stream.flush()
                print(json.dumps({"budget": budget, "replication": replication,
                                  "rows": [r["policy"] for r in rows]}), flush=True)


if __name__ == "__main__":
    main()
