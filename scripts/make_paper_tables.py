"""P11/P12: build paper-format data tables (LaTeX + CSV + JSON).

Tables produced (all from frozen formal results):
  Table 1 (S1): B*KL for the four policies across the five budgets.
  Table 2 (S2): design ratio and learned op-error across alpha.
  Table 3 (S2): generator-reuse frontier (regret + cumulative compute).
  Table 4 (R1): Gas semi-synthetic B*KL across the four budgets.
  Table 5 (R2): per-campaign metrics (projection loss, moment RMSE, C2ST AUC,
                ESS) for the four policies (paper Table 1).
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

S1_BUDGETS = (2000, 4000, 8000, 16000, 32000)
GAS_BUDGETS = (400, 800, 1600, 3200)
S1_POLICIES = ("Uniform SQD", "A-OSQD", "Discriminative Score OED", "learned RR-GID")
R1_POLICIES = ("Uniform SQD", "A-OSQD", "Discriminative Score OED", "RR-GID")
CAMPAIGNS = ("batch7", "batches89", "batch10")

OUT = Path("paper_tables")


def _mean_se(values) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


def _load_jsonl(fp: Path) -> list[dict]:
    if not fp.exists():
        return []
    return [json.loads(l) for l in fp.read_text(encoding="utf-8").splitlines() if l.strip()]


def _fmt(v: float, se: float, nd: int = 1) -> str:
    return f"{v:.{nd}f}$\\pm${se:.{nd}f}"


def _round_rec(rows: list[dict], nd: int = 3) -> list[dict]:
    """Round float cells to ``nd`` decimals for clean CSV/LaTeX output."""
    out = []
    for r in rows:
        nr = {}
        for k, v in r.items():
            if isinstance(v, float):
                nr[k] = round(v, nd)
            else:
                nr[k] = v
        out.append(nr)
    return out


# ---------------------------------------------------------------------------
def table_s1() -> dict:
    """Table 1: S1 B*KL (mean +- SE) per policy per budget."""
    rows = []
    for pol in S1_POLICIES:
        for b in S1_BUDGETS:
            vals = []
            for fp in (Path("results") / f"p4_formal_{b}.jsonl",
                       Path("results") / f"p5_four_{b}.jsonl"):
                for r in _load_jsonl(fp):
                    if r["policy"] == pol:
                        vals.append(r["B_kl"])
            if vals:
                m, se = _mean_se(vals)
                rows.append({"policy": pol, "budget": b, "B_kl_mean": m, "B_kl_se": se, "n": len(vals)})
    return {"name": "Table 1 (S1)", "rows": rows}


# B*KL theoretical oracle constants (1/2 * Phi(p*)) for reference (PDF Sec. 5).
THEORY_BKL = {2000: 31.9, 4000: 63.8, 8000: 127.6, 16000: 255.2, 32000: 510.4}


def table_s2_alpha() -> dict:
    """Table 2: S2 design ratio + op error across alpha."""
    s = json.load(open("results/p7_s2_summary.json", encoding="utf-8"))
    rows = []
    for r in s["alpha_sweep"]:
        rows.append({
            "alpha": r["alpha"],
            "design_ratio_Uniform": r["design_ratio_Uniform SQD"],
            "design_ratio_AOSQD": r["design_ratio_A-OSQD"],
            "design_ratio_Disc": r["design_ratio_Discriminative Score OED"],
            "design_ratio_RRGID": r["design_ratio_learned RR-GID"],
            "op_error_learned": r["max_operator_error_learned"],
        })
    return {"name": "Table 2 (S2 alpha)", "rows": rows}


def table_s2_reuse() -> dict:
    """Table 3: S2 reuse frontier."""
    s = json.load(open("results/p7_s2_summary.json", encoding="utf-8"))
    rows = []
    for r in sorted(s["reuse"], key=lambda x: x["campaigns"]):
        rows.append({
            "campaigns": r["campaigns"],
            "regret_RRGID": r["mean_design_regret_RR_GID"],
            "regret_Disc": r["mean_design_regret_Discriminative"],
            "regret_Uniform": r["mean_design_regret_Uniform"],
            "regret_AOSQD": r["mean_design_regret_A_OSQD"],
            "compute_RRGID_s": r["cumulative_compute_RR_GID_s"],
            "compute_Disc_s": r["cumulative_compute_Discriminative_s"],
        })
    return {"name": "Table 3 (S2 reuse)", "rows": rows}


def table_r1() -> dict:
    """Table 4: R1 B*KL per policy per Gas budget."""
    rows = []
    for pol in R1_POLICIES:
        for b in GAS_BUDGETS:
            vals = [r["B_kl"] for r in _load_jsonl(Path("results") / f"p9_r1_{b}.jsonl")
                    if r["policy"] == pol]
            if vals:
                m, se = _mean_se(vals)
                rows.append({"policy": pol, "budget": b, "B_kl_mean": m, "B_kl_se": se, "n": len(vals)})
    return {"name": "Table 4 (R1)", "rows": rows}


def table_r2() -> dict:
    """Table 5: R2 per-campaign metrics for the four policies (paper Table 1)."""
    rows = []
    for camp in CAMPAIGNS:
        for pol in R1_POLICIES:
            agg: dict[str, list[float]] = defaultdict(list)
            for b in GAS_BUDGETS:
                for r in _load_jsonl(Path("results") / f"p10_r2_{camp}_{b}.jsonl"):
                    if r["policy"] == pol:
                        for k in ("projection_loss", "heldout_mean_rmse", "c2st_auc", "ess"):
                            agg[k].append(r[k])
            row = {"campaign": camp, "policy": pol}
            for k in ("projection_loss", "heldout_mean_rmse", "c2st_auc", "ess"):
                if agg[k]:
                    m, se = _mean_se(agg[k])
                    row[f"{k}_mean"] = m
                    row[f"{k}_se"] = se
            rows.append(row)
    return {"name": "Table 5 (R2)", "rows": rows}


# ---------------------------------------------------------------------------
def write_csv(tables: list[dict]) -> None:
    for t in tables:
        name = t["name"].replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
        rows = t["rows"]
        if not rows:
            continue
        keys = list(rows[0].keys())
        with open(OUT / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote {OUT}/{name}.csv ({len(rows)} rows)")


def write_latex(tables: list[dict]) -> None:
    for t in tables:
        name = t["name"].replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
        rows = t["rows"]
        if not rows:
            continue
        keys = list(rows[0].keys())
        # numeric columns right-aligned, text columns left-aligned
        aligns = "".join("r" if isinstance(rows[0].get(k), (int, float)) else "l" for k in keys)
        lines = ["\\begin{table}[t]", "\\centering", "\\small",
                 "\\caption{" + t["name"] + ".}", "\\label{tab:" + name + "}"]
        lines.append("\\begin{tabular}{" + aligns + "}")
        lines.append("\\toprule")
        lines.append(" & ".join(k.replace("_", "\\_") for k in keys) + " \\\\")
        lines.append("\\midrule")
        for r in rows:
            cells = []
            for k in keys:
                v = r[k]
                if isinstance(v, float):
                    cells.append(f"{v:.3f}")
                elif isinstance(v, int):
                    cells.append(str(v))
                else:
                    cells.append(str(v).replace("_", "\\_"))
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        with open(OUT / f"{name}.tex", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"  wrote {OUT}/{name}.tex")


def write_json(tables: list[dict]) -> None:
    payload = {t["name"]: t["rows"] for t in tables}
    with open(OUT / "paper_tables.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"  wrote {OUT}/paper_tables.json")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    tables = [table_s1(), table_s2_alpha(), table_s2_reuse(), table_r1(), table_r2()]
    # round float cells for clean paper output
    tables = [{"name": t["name"], "rows": _round_rec(t["rows"])} for t in tables]
    write_csv(tables)
    write_latex(tables)
    write_json(tables)
    print("done.")


if __name__ == "__main__":
    main()
