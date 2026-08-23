"""P11 Table 1: R2 natural-drift metrics per campaign (PDF Table 1).

For each natural campaign (batch 7, batches 8-9, batch 10) report the four
policies' family-internal projection loss, held-out moment RMSE, C2ST AUC,
and importance ESS, averaged over budgets with SE.  Outputs both JSON (for
latex/word) and a CSV preview.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BUDGETS = (400, 800, 1600, 3200)
CAMPAIGNS = ("batch7", "batches89", "batch10")
POLICIES = ("Uniform SQD", "A-OSQD", "Discriminative Score OED", "RR-GID")
METRICS = ("projection_loss", "heldout_mean_rmse", "heldout_std_rmse", "c2st_auc", "ess")


def _load(campaign: str) -> dict[tuple[str, str], list[float]]:
    agg: dict[tuple[str, str], list[float]] = defaultdict(list)
    for b in BUDGETS:
        fp = Path("results") / f"p10_r2_{campaign}_{b}.jsonl"
        if not fp.exists():
            continue
        for line in fp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            for m in METRICS:
                agg[(r["policy"], m)].append(float(r[m]))
    return dict(agg)


def main() -> None:
    rows = []
    for camp in CAMPAIGNS:
        data = _load(camp)
        camp_row = {"campaign": camp}
        for pol in POLICIES:
            for m in METRICS:
                v = np.asarray(data.get((pol, m), []))
                if len(v):
                    camp_row[f"{pol}|{m}"] = [float(v.mean()), float(v.std() / np.sqrt(len(v)))]
                else:
                    camp_row[f"{pol}|{m}"] = None
        rows.append(camp_row)

    Path("results").mkdir(exist_ok=True)
    with open("results/table1_r2.json", "w", encoding="utf-8") as f:
        json.dump({"campaigns": rows, "metrics": METRICS, "policies": list(POLICIES)},
                  f, indent=2, sort_keys=True)
        f.write("\n")

    # CSV preview (mean only)
    with open("results/table1_r2.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["campaign", "policy", "proj_loss", "mean_rmse", "std_rmse", "c2st_auc", "ess"]
        w.writerow(header)
        for camp_row in rows:
            camp = camp_row["campaign"]
            for pol in POLICIES:
                vals = [camp, pol]
                for m in METRICS:
                    d = camp_row.get(f"{pol}|{m}")
                    vals.append("--" if d is None else f"{d[0]:.4f}")
                w.writerow(vals)
    print("saved results/table1_r2.json + results/table1_r2.csv")
    for camp_row in rows:
        print(camp_row["campaign"], {p: round(camp_row[f"{p}|c2st_auc"][0], 3) for p in POLICIES})


if __name__ == "__main__":
    main()
