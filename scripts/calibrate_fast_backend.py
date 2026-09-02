"""E4: calibrate cached QMC against the existing uncached QMC integrand."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rr_gid_cn.conditional_backend import (
    build_conditional_feature_basis,
    build_panel_information_basis,
    evaluate_conditional_basis,
    uncached_qmc_mean,
)
from rr_gid_cn.paper_run import run_paper_replication
from rr_gid_cn.synthetic_oracle import all_pairs, make_frozen_mixture, reference_scale, sample_full


def _cpu(monkeypatch_cuda: bool = True):
    if not monkeypatch_cuda:
        return
    import rr_gid_cn.synthetic_oracle as oracle
    oracle._cuda_device = lambda: None  # noqa: ARG005


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/paper/oracle_calibration.yaml"))
    ap.add_argument("--prepared", type=Path, default=Path("experiments/paper/oracle_artifact.pkl"))
    ap.add_argument("--queries", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("results/paper/oracle_calibration/report.json"))
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cal = cfg.get("calibration", {})
    n_queries = int(args.queries or cal.get("query_count", 40))
    mixture = make_frozen_mixture(seed=2026, alpha=1.0)
    scale = reference_scale(mixture, n=6000, seed=2026)
    panels = all_pairs()
    rng = np.random.default_rng(20260902)
    full = sample_full(mixture, max(n_queries * 2, 64), seed=9)
    _cpu()
    rel = []
    abs_coord = []
    order = int(cfg.get("score_qmc_order", 8))
    for i in range(n_queries):
        panel = panels[int(rng.integers(len(panels)))]
        row = full[i % len(full)][list(panel)]
        beta = rng.normal(0.0, 0.35, size=12)
        beta = beta * min(1.0, 2.0 / max(np.linalg.norm(beta), 1e-8))
        basis = build_conditional_feature_basis(
            mixture, np.atleast_2d(row), panel, order=order, seed=1000 + i,
            scale=scale, dtype="float64",
        )
        cached = evaluate_conditional_basis(beta, basis)[0]
        gold = uncached_qmc_mean(mixture, beta, np.atleast_2d(row), panel, order, 1000 + i, scale)[0]
        denom = np.maximum(np.abs(gold), 1e-8)
        rel.append(float(np.max(np.abs(cached - gold) / denom)))
        abs_coord.append(float(np.max(np.abs(cached - gold))))
    rel = np.asarray(rel)
    abs_coord = np.asarray(abs_coord)
    query_ok = (
        float(np.median(rel)) <= float(cal.get("median_relative_error", 1e-3))
        and float(np.quantile(rel, 0.95)) <= float(cal.get("p95_relative_error", 1e-2))
        and float(abs_coord.max()) <= float(cal.get("max_abs_coordinate_error", 2e-2))
    )
    info_ok = True
    e2e = None
    if args.prepared.exists():
        prepared = pickle.loads(args.prepared.read_bytes())
        subset = panels[: int(cal.get("information_panels", 8))]
        basis = build_panel_information_basis(
            mixture, prepared["reference"], subset, scale=scale,
            outer_rows=32, order=int(cfg.get("information_qmc_order", 6)),
            seed=7, dtype="float64",
        )
        for _ in range(int(cal.get("information_betas", 3))):
            beta = rng.normal(0.0, 0.2, size=12)
            infos = basis.information(beta, subset)
            if any(np.linalg.eigvalsh(mat).min() < -1e-8 for mat in infos.values()):
                info_ok = False
        rows = run_paper_replication(
            mixture, scale, panels, 8000, 202609000, prepared,
            scoring_steps=2, score_qmc_order=order,
            information_qmc_order=int(cfg.get("information_qmc_order", 6)),
            information_outer_rows=32,
            policies=("Uniform SQD", "RR-GID"),
            seed_manifest_entry={"replication": 0, "replication_seed": 202609000,
                                 "target_draw_seed": 202609003, "pilot_or_design_seed": 202609011,
                                 "score_seed_root": 202609004, "information_seed_root": 202616000},
        )
        e2e = {"n": len(rows), "risk": [row["risk_ratio"] for row in rows]}
    report = {
        "query_ok": bool(query_ok),
        "information_ok": bool(info_ok),
        "median_relative_error": float(np.median(rel)),
        "p95_relative_error": float(np.quantile(rel, 0.95)),
        "max_abs_coordinate_error": float(abs_coord.max()),
        "n_queries": int(n_queries),
        "end_to_end": e2e,
        "passed": bool(query_ok and info_ok),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
