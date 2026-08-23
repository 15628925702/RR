"""Resumable P4 formal runner. Each completed replication is appended atomically."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import argparse

import yaml

from rr_gid_cn.s1_gate import prepare_s1_oracle, run_replication
from rr_gid_cn.synthetic_oracle import all_pairs, make_frozen_mixture, reference_scale, sample_full


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--config", type=Path, default=Path("configs/p4_formal.yaml"))
    ap.add_argument("--max-replications", type=int, default=None, help="limit replications per budget (for smoke runs)")
    ap.add_argument("--scoring-steps", type=int, default=None, help="J Fisher-scoring steps (default from config, 2; J ablation passes 0/1/2 explicitly)")
    ap.add_argument("--rep-range", type=int, nargs=2, default=None, help="[start, end) replication range for sharding a budget across processes")
    ap.add_argument("--prepared", type=Path, default=Path("experiments/p4_prepared_oracle_fix1.pkl"))
    ap.add_argument("--out-prefix", default="p4_exact_fix1")
    ap.add_argument("--reference-size", type=int, default=50000)
    ap.add_argument("--large-reference-size", type=int, default=None)
    ap.add_argument("--info-tilted", type=int, default=None)
    ap.add_argument("--info-cond", type=int, default=None)
    ap.add_argument("--scoring-step-size", type=float, default=None)
    args = ap.parse_args()
    with args.config.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    p4 = cfg["p4"]
    budgets = (args.budget,) if args.budget is not None else tuple(p4["budgets"])
    replications = args.max_replications or int(p4["replications"])
    steps = int(p4["scoring_steps"]) if args.scoring_steps is None else args.scoring_steps
    # Explicit --scoring-steps always writes a J-suffixed file so the J ablation
    # never collides with the default (J=2) main-configuration output.
    suffix = "" if args.scoring_steps is None else f"_J{steps}"
    out = Path("results")
    done: dict[int, set] = {b: set() for b in budgets}
    for b in budgets:
        fp = out / f"{args.out_prefix}_{b}{suffix}.jsonl"
        if fp.exists():
            for line in fp.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    done[b].add((row["replication"], row["policy"]))
    mixture = make_frozen_mixture(seed=2026, alpha=1.0)
    scale = reference_scale(mixture, 6000, 2026)
    panels = all_pairs()
    prepared_path = args.prepared
    if prepared_path.exists():
        with prepared_path.open("rb") as stream:
            prepared = pickle.load(stream)
        if "reference_large" not in prepared:
            prepared["reference_large"] = sample_full(mixture, p4["large_reference_size"], 2026 + 12345)
            with prepared_path.open("wb") as stream:
                pickle.dump(prepared, stream, protocol=5)
    else:
        prepared = prepare_s1_oracle(
            mixture, scale, panels, seed=2026,
            reference_size=args.reference_size,
            information_samples=args.info_tilted or p4["information_tilted_samples"],
            conditional_samples=args.info_cond or p4["information_conditional_samples"],
            large_reference_size=args.large_reference_size or p4["large_reference_size"],
        )
        with prepared_path.open("wb") as stream:
            pickle.dump(prepared, stream, protocol=5)
    kwargs = dict(lu=p4["lu"], h_tilted=p4["h_tilted"], h_cond=p4["h_cond"],
                  pilot_norm_cap=p4["pilot_norm_cap"], kl_samples=p4["kl_samples"],
                  scoring_steps=steps,
                  theta_norm_cap=p4.get("theta_norm_cap"),
                  scoring_step_size=p4.get("scoring_step_size", 1.0) if args.scoring_step_size is None else args.scoring_step_size)
    for budget in budgets:
        fp = out / f"{args.out_prefix}_{budget}{suffix}.jsonl"
        with fp.open("a", encoding="utf-8") as stream:
            start, end = args.rep_range if args.rep_range else (0, replications)
            for replication in range(start, min(end, replications)):
                if all((replication, pol) in done[budget] for pol in ("Uniform SQD", "A-OSQD", "oracle RR-GID")):
                    continue
                seed = 202600000 + budget * 1000 + replication
                rows = run_replication(mixture, scale, panels, budget, seed, prepared=prepared, **kwargs)
                for row in rows:
                    row["replication"] = replication
                    if (replication, row["policy"]) not in done[budget]:
                        stream.write(json.dumps(row, sort_keys=True) + "\n")
                        done[budget].add((replication, row["policy"]))
                stream.flush()
                print(json.dumps({"budget": budget, "replication": replication, "rows": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
