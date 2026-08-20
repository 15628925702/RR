"""Resumable P4 formal runner. Each completed replication is appended atomically."""

from __future__ import annotations

import json
from pathlib import Path

from rr_gid_cn.s1_gate import run_replication
from rr_gid_cn.synthetic_oracle import all_pairs, make_frozen_mixture, reference_scale


def main() -> None:
    out = Path("results/p4_formal_rows.jsonl")
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["budget"], row["replication"], row["policy"]))
    mixture = make_frozen_mixture(seed=2026, alpha=1.0)
    scale = reference_scale(mixture, 6000, 2026)
    panels = all_pairs()
    out.parent.mkdir(exist_ok=True)
    with out.open("a", encoding="utf-8") as stream:
        for budget in (2000, 4000, 8000, 16000, 32000):
            for replication in range(200):
                seed = 202600000 + budget * 1000 + replication
                rows = run_replication(mixture, scale, panels, budget, seed, reference_size=50000, information_samples=1000, conditional_samples=64)
                for row in rows:
                    row["replication"] = replication
                    if (budget, replication, row["policy"]) not in done:
                        stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                print(json.dumps({"budget": budget, "replication": replication, "rows": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
