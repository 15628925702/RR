"""Build and validate the P4 Phase-2 single-Q0 gold oracle (never Phase 3)."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import yaml

from rr_gid_cn.oracle_measure import (
    ConditionalQMC,
    FullLawQMC,
    InformationQMC,
    OracleMeasure,
    array_sha256,
    canonical_sha256,
    cholesky_objective,
    frank_wolfe_gold,
    mixture_sha256,
)
from rr_gid_cn.p4_integrity import sha256_file
from rr_gid_cn.synthetic_oracle import (
    all_pairs,
    feature_map,
    make_frozen_mixture,
    reference_scale,
    tilted_conditional_feature_mean_batch,
)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, encoding="utf-8").strip()


def _configs(payload: dict):
    return (
        FullLawQMC(**payload["full_law"]),
        ConditionalQMC(**payload["conditional"]),
        InformationQMC(**payload["information"]),
    )


def _oracle(payload: dict, mixture, scale) -> OracleMeasure:
    full, conditional, information = _configs(payload)
    return OracleMeasure(mixture, scale, full, conditional, information)


def _jsonable_diagnostics(diags):
    rows = []
    for panel_diags in diags:
        rows.append(
            {
                "converged": bool(all(item["converged"] for item in panel_diags)),
                "final_order_max": int(max(item["final_order"] for item in panel_diags)),
                "final_order_min": int(min(item["final_order"] for item in panel_diags)),
                "max_abs_delta": float(max(item["max_abs_delta"] for item in panel_diags)),
                "scramble_se_max": float(max(item["scramble_se"] for item in panel_diags)),
                "scrambles": int(panel_diags[0]["scrambles"]),
            }
        )
    return rows


def build_one(
    name: str,
    payload: dict,
    mixture,
    scale: np.ndarray,
    beta_true: np.ndarray,
    panels,
    out_dir: Path,
    provenance: dict,
):
    started = perf_counter()
    oracle = _oracle(payload, mixture, scale)
    moments = oracle.moments(beta_true)
    panel_results = []
    for index, panel in enumerate(panels):
        print(f"[{name}] panel {index + 1}/{len(panels)} {panel}", flush=True)
        panel_results.append(oracle.panel_information(beta_true, panel))
    information_raw = np.stack([item["raw"] for item in panel_results])
    information_sym = np.stack([item["sym"] for item in panel_results])
    information_projected = np.stack([item["projected"] for item in panel_results])
    p_star, fw_gap, fw_iterations, phi, linear = frank_wolfe_gold(
        moments["F_projected"],
        information_projected,
        np.full(len(panels), 1.0 / len(panels)),
        tolerance=float(payload["frank_wolfe"]["tolerance"]),
        max_iter=int(payload["frank_wolfe"]["max_iter"]),
    )
    verified_phi, verified_linear = cholesky_objective(
        moments["F_projected"], information_projected, p_star
    )
    if not np.isclose(phi, verified_phi, rtol=1e-12, atol=1e-12):
        raise RuntimeError("gold objective was not reproducible by Cholesky solve")
    npz_path = out_dir / f"oracle_{name}.npz"
    np.savez_compressed(
        npz_path,
        scale=scale,
        beta_true=beta_true,
        F_raw=moments["F_raw"],
        F_sym=moments["F_sym"],
        F_projected=moments["F_projected"],
        information_raw=information_raw,
        information_sym=information_sym,
        information_projected=information_projected,
        information_psd_correction_norm=np.asarray(
            [item["psd_correction_norm"] for item in panel_results]
        ),
        information_outer_scramble_se_max=np.asarray(
            [item["outer_scramble_se_max"] for item in panel_results]
        ),
        score_mean=np.stack([item["score_mean"] for item in panel_results]),
        score_mean_scramble_se=np.stack(
            [item["score_mean_scramble_se"] for item in panel_results]
        ),
        p_star=p_star,
    )
    definition = oracle.definition()
    metadata = {
        "schema_version": "p4-phase2-oracle-artifact-v1",
        "artifact_role": name,
        "not_formal": True,
        "phase": 2,
        **provenance,
        "config_hash": provenance["config_sha256"],
        "mixture_seed": int(payload["mixture_seed"]),
        "alpha": float(payload["alpha"]),
        "mixture_parameter_hash": mixture_sha256(mixture),
        "scale_hash": array_sha256(scale),
        "beta_true": beta_true.tolist(),
        "beta_true_hash": array_sha256(beta_true),
        "integration": definition,
        "integration_input_hashes": {
            "mixture": mixture_sha256(mixture),
            "scale": array_sha256(scale),
            "beta_true": array_sha256(beta_true),
        },
        "F": {
            "raw_hash": array_sha256(moments["F_raw"]),
            "sym_hash": array_sha256(moments["F_sym"]),
            "projected_hash": array_sha256(moments["F_projected"]),
            "psd_correction_norm": moments["F_psd_correction_norm"],
            "A": moments["A"],
            "mu_hash": array_sha256(moments["mu"]),
            "A_scramble_se": moments["A_scramble_se"],
            "mu_scramble_se_max": moments["mu_scramble_se_max"],
            "F_scramble_se_max": moments["F_scramble_se_max"],
            "replicate_A": moments["replicate_A"],
        },
        "panels": [list(panel) for panel in panels],
        "panel_information": {
            "raw_hash": array_sha256(information_raw),
            "sym_hash": array_sha256(information_sym),
            "projected_hash": array_sha256(information_projected),
            "psd_correction_norm": [
                float(item["psd_correction_norm"]) for item in panel_results
            ],
            "outer_scramble_se_max": [
                float(item["outer_scramble_se_max"]) for item in panel_results
            ],
            "conditional_convergence": _jsonable_diagnostics(
                [item["conditional_diagnostics"] for item in panel_results]
            ),
        },
        "p_star": p_star.tolist(),
        "p_star_hash": array_sha256(p_star),
        "fw_gap": float(fw_gap),
        "fw_iterations": int(fw_iterations),
        "objective_phi": float(phi),
        "theory_constant_half_phi": float(0.5 * phi),
        "gold_linear_algebra": {**linear, **verified_linear, "method": "cholesky_solve"},
        "independent_convergence_replicate_summary": None,
        "reference_integration_input_hashes": definition,
        "npz_sha256": sha256_file(npz_path),
        "elapsed_seconds": float(perf_counter() - started),
    }
    metadata_path = out_dir / f"oracle_{name}.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return oracle, metadata, panel_results, information_projected, p_star


def _whitened_operator_error(fisher: np.ndarray, estimated: np.ndarray, gold: np.ndarray) -> float:
    eigenvalues, eigenvectors = np.linalg.eigh((fisher + fisher.T) / 2)
    if eigenvalues.min() <= 0:
        raise np.linalg.LinAlgError("F is not positive definite for whitening")
    inverse_root = (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T
    return float(np.linalg.norm(inverse_root @ (estimated - gold) @ inverse_root, ord=2))


def current_beta_diagnostic(oracle, beta_true, panels, p_star, payload):
    cfg = payload["current_beta_information"]
    selected = np.argsort(p_star)[-int(cfg["selected_panels"]):][::-1]
    beta = float(cfg["beta_multiplier"]) * beta_true
    fisher = oracle.F(beta)
    rows = []
    for rank, index in enumerate(selected):
        panel = panels[int(index)]
        gold = oracle.panel_information(beta, panel)
        tilted = oracle.tilt_full(
            beta, int(cfg["estimated_tilted_samples"]), int(cfg["seed"]) + rank
        )
        observed = tilted[:, list(panel)]
        mu = oracle.mu(beta)
        first = tilted_conditional_feature_mean_batch(
            oracle.mixture,
            beta,
            observed,
            panel,
            int(cfg["estimated_conditional_samples"]),
            int(cfg["seed"]) + 1000 + 2 * rank,
            oracle.scale,
        ) - mu
        second = tilted_conditional_feature_mean_batch(
            oracle.mixture,
            beta,
            observed,
            panel,
            int(cfg["estimated_conditional_samples"]),
            int(cfg["seed"]) + 1001 + 2 * rank,
            oracle.scale,
        ) - mu
        first -= first.mean(axis=0)
        second -= second.mean(axis=0)
        raw = (first.T @ second + second.T @ first) / (2 * (len(first) - 1))
        sym = (raw + raw.T) / 2
        eigenvalues, eigenvectors = np.linalg.eigh(sym)
        projected = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
        rows.append(
            {
                "panel": list(panel),
                "panel_index": int(index),
                "raw": raw.tolist(),
                "sym": sym.tolist(),
                "projected": projected.tolist(),
                "psd_correction_norm": float(np.linalg.norm(projected - sym, ord="fro")),
                "gold_raw": gold["raw"].tolist(),
                "gold_sym": gold["sym"].tolist(),
                "gold_projected": gold["projected"].tolist(),
                "gold_psd_correction_norm": gold["psd_correction_norm"],
                "whitened_operator_error": _whitened_operator_error(
                    fisher, projected, gold["projected"]
                ),
            }
        )
    return {"beta": beta.tolist(), "rows": rows}


def score_centering_diagnostic(
    oracle, beta_true, panels, information, p_star, panel_results, half_phi, payload
):
    cfg = payload["score_centering"]
    fisher = oracle.F(beta_true)
    M = np.tensordot(p_star, information, axes=(0, 0))
    chol_M = np.linalg.cholesky((M + M.T) / 2)
    chol_F = np.linalg.cholesky((fisher + fisher.T) / 2)
    selected = np.argsort(p_star)[-int(cfg["selected_panels"]):][::-1]

    def metrics(delta):
        transformed = np.linalg.solve(chol_M.T, np.linalg.solve(chol_M, delta))
        induced = float(cfg["Bmax"]) * float(transformed @ fisher @ transformed)
        whitened = np.linalg.solve(chol_F, delta)
        return {
            "euclidean_norm": float(np.linalg.norm(delta)),
            "fisher_whitened_norm": float(np.linalg.norm(whitened)),
            "Bmax_induced_risk_bias": induced,
            "risk_fraction_of_half_phi": float(induced / half_phi),
        }

    rows = []
    for rank, index in enumerate(selected):
        panel = panels[int(index)]
        delta = panel_results[int(index)]["score_mean"]
        draws = oracle.tilt_full(
            beta_true, int(cfg["sampling_draws"]), int(cfg["sampling_seed"]) + rank
        )
        conditional = oracle.conditional_mean(
            beta_true,
            panel,
            draws[:, list(panel)],
            seed_offset=100_000 + rank,
        )
        scores = conditional - oracle.mu(beta_true)
        rows.append(
            {
                "panel": list(panel),
                "panel_index": int(index),
                **metrics(delta),
                "delta": delta.tolist(),
                "numerical_se": panel_results[int(index)][
                    "score_mean_scramble_se"
                ].tolist(),
                "sampling_se": (
                    scores.std(axis=0, ddof=1) / np.sqrt(len(scores))
                ).tolist(),
            }
        )
    allocation_delta = np.sum(
        p_star[:, None] * np.stack([item["score_mean"] for item in panel_results]),
        axis=0,
    )
    allocation = {
        **metrics(allocation_delta),
        "delta": allocation_delta.tolist(),
        "numerical_se": np.sqrt(
            np.sum(
                np.square(
                    p_star[:, None]
                    * np.stack(
                        [item["score_mean_scramble_se"] for item in panel_results]
                    )
                ),
                axis=0,
            )
        ).tolist(),
    }
    gate = max([allocation["risk_fraction_of_half_phi"]] + [
        row["risk_fraction_of_half_phi"] for row in rows
    ]) < float(cfg["risk_fraction_of_half_phi_max"])
    return {"selected_panels": rows, "allocation": allocation, "passed": bool(gate)}


def raw_kl_diagnostic(oracle, beta_true, payload):
    cfg = payload["raw_kl"]
    rng = np.random.default_rng(int(cfg["directions_seed"]))
    rows = []
    for norm in cfg["perturbation_norms"]:
        direction = rng.normal(size=len(beta_true))
        direction /= np.linalg.norm(direction)
        beta_hat = beta_true + float(norm) * direction
        rows.append({"perturbation_norm": float(norm), **oracle.kl(beta_true, beta_hat)})
    return {
        "rows": rows,
        "passed": bool(all(row["raw"] >= -row["numerical_tolerance"] for row in rows)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/p4_phase2_gold_20260826.yaml")
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite immutable output directory: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.mkdir()
    log_stream = (args.out / "run.log").open("w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_stream)
    sys.stderr = _Tee(sys.__stderr__, log_stream)
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))["p4_phase2"]
    if payload.get("phase") != 2 or not payload.get("not_formal"):
        raise ValueError("Phase-2 runner requires phase=2 and not_formal=true")
    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    source_paths = [
        Path("src/rr_gid_cn/oracle_measure.py"),
        Path("src/rr_gid_cn/synthetic_oracle.py"),
        Path("scripts/p4_phase2_gold.py"),
    ]
    provenance = {
        "code_commit": commit,
        "code_dirty": dirty,
        "source_sha256": {str(path).replace("\\", "/"): sha256_file(path) for path in source_paths},
        "config_sha256": sha256_file(args.config),
        "config_canonical_sha256": canonical_sha256(payload),
    }
    mixture = make_frozen_mixture(
        seed=int(payload["mixture_seed"]), alpha=float(payload["alpha"])
    )
    scale = reference_scale(
        mixture,
        n=int(payload["scale_sample_size"]),
        seed=int(payload["scale_seed"]),
    )
    primary_oracle = _oracle(payload, mixture, scale)
    beta_true = primary_oracle.calibrate_beta(
        seed=int(payload["beta_seed"]),
        target_ess_fraction=float(payload["beta_target_ess_fraction"]),
    )
    panels = all_pairs()
    primary = build_one(
        "primary", payload, mixture, scale, beta_true, panels, args.out, provenance
    )
    replicate_payload = deepcopy(payload)
    replicate_cfg = payload["independent_replicate"]
    replicate_payload["full_law"]["order"] += int(replicate_cfg["full_order_increment"])
    replicate_payload["full_law"]["seed"] += int(replicate_cfg["seed_offset"])
    replicate_payload["conditional"]["seed"] += int(replicate_cfg["seed_offset"])
    replicate_payload["information"]["outer_order"] += int(
        replicate_cfg["information_outer_order_increment"]
    )
    replicate_payload["information"]["seed"] += int(replicate_cfg["seed_offset"])
    replicate = build_one(
        "replicate", replicate_payload, mixture, scale, beta_true, panels, args.out, provenance
    )
    primary_oracle, primary_meta, primary_panels, primary_info, primary_p = primary
    _, replicate_meta, _, _, replicate_p = replicate
    relative_difference = abs(
        primary_meta["objective_phi"] - replicate_meta["objective_phi"]
    ) / replicate_meta["objective_phi"]
    top_k = int(payload["convergence_gate"]["top_panel_count"])
    primary_top = set(np.argsort(primary_p)[-top_k:])
    replicate_top = set(np.argsort(replicate_p)[-top_k:])
    top_overlap = len(primary_top & replicate_top) / top_k
    convergence = {
        "primary_phi": primary_meta["objective_phi"],
        "replicate_phi": replicate_meta["objective_phi"],
        "constant_relative_difference": float(relative_difference),
        "constant_relative_tolerance": float(
            payload["convergence_gate"]["constant_relative_tolerance"]
        ),
        "top_panel_count": top_k,
        "top_panel_overlap": float(top_overlap),
        "top_panel_overlap_min": float(
            payload["convergence_gate"]["top_panel_overlap_min"]
        ),
        "passed": bool(
            relative_difference
            < float(payload["convergence_gate"]["constant_relative_tolerance"])
            and top_overlap >= float(payload["convergence_gate"]["top_panel_overlap_min"])
        ),
    }
    primary_meta["independent_convergence_replicate_summary"] = convergence
    (args.out / "oracle_primary.json").write_text(
        json.dumps(primary_meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    diagnostics = {
        "schema_version": "p4-phase2-diagnostics-v1",
        "not_formal": True,
        "phase": 2,
        "convergence": convergence,
        "score_centering": score_centering_diagnostic(
            primary_oracle,
            beta_true,
            panels,
            primary_info,
            primary_p,
            primary_panels,
            primary_meta["theory_constant_half_phi"],
            payload,
        ),
        "current_beta_information": current_beta_diagnostic(
            primary_oracle, beta_true, panels, primary_p, payload
        ),
        "raw_kl": raw_kl_diagnostic(primary_oracle, beta_true, payload),
    }
    diagnostics["all_phase2_gates_passed"] = bool(
        diagnostics["convergence"]["passed"]
        and diagnostics["score_centering"]["passed"]
        and diagnostics["raw_kl"]["passed"]
        and primary_meta["gold_linear_algebra"]["lambda_min"] > 0
    )
    diagnostics_path = args.out / "diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
    )
    hash_manifest = {
        path.name: sha256_file(path)
        for path in sorted(args.out.iterdir())
        if path.is_file() and path.name != "sha256.json"
    }
    (args.out / "sha256.json").write_text(
        json.dumps(hash_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.out),
        "all_phase2_gates_passed": diagnostics["all_phase2_gates_passed"],
        "half_phi": primary_meta["theory_constant_half_phi"],
        "constant_relative_difference": relative_difference,
        "top_panel_overlap": top_overlap,
    }, sort_keys=True))
    if not diagnostics["all_phase2_gates_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
