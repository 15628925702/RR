"""Resume a verified Phase-2 artifact and recompute only score centering."""

from __future__ import annotations

import argparse
import gc
import json
import os
from datetime import datetime, timezone
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
    mixture_sha256,
)
from rr_gid_cn.p4_integrity import sha256_file
from rr_gid_cn.synthetic_oracle import (
    all_pairs,
    make_frozen_mixture,
    tilted_conditional_mean_exact,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MAX_SCORE_CENTERING_GATE = 0.01
CHECKPOINT_SCHEMA_VERSION = "p4-phase2-score-centering-checkpoint-v1"
SCRAMBLE_PARTIAL_SCHEMA_VERSION = "p4-phase2-scramble-partial-v1"
DOCUMENT_SCORE_ESTIMATOR = "same_q0_oracle_mu"
PANEL_ARRAY_FIELDS = (
    "delta",
    "outer_numerical_se",
    "conditional_numerical_se",
    "conditional_terminal_delta",
    "sampling_se",
    "row_final_order",
)
PANEL_SCALAR_FIELDS = (
    "conditional_query_scramble_se_max",
    "conditional_query_terminal_delta_max",
    "conditional_converged",
    "estimator",
    "max_final_order",
    "mean_final_order",
)
OUTER_SCRAMBLE_ARRAY_FIELDS = ("score_mean", "conditional_se", "terminal_delta")
OUTER_SCRAMBLE_SCALAR_FIELDS = (
    "mean_final_order",
    "max_final_order",
    "query_scramble_se_max",
    "query_terminal_delta_max",
    "converged",
)
SAMPLING_PARTIAL_ARRAY_FIELDS = ("row_means",)
SAMPLING_PARTIAL_SCALAR_FIELDS = ("mean_final_order", "max_final_order", "converged")


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


def _atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def checkpoint_identity(payload: dict) -> dict:
    base = payload["base_artifact"]
    return {
        "config_canonical_sha256": canonical_sha256(payload),
        "precision_id": payload["precision_id"],
        "base_artifact_sha256": {
            "config": base["config_sha256"],
            "metadata": base["metadata_sha256"],
            "npz": base["npz_sha256"],
        },
        "source_numerical_kernel_sha256": dict(sorted(payload["source_sha256"].items())),
    }


class PanelCheckpointStore:
    def __init__(self, output_dir: Path, identity: dict, *, resume: bool):
        self.output_dir = Path(output_dir)
        self.manifest_path = self.output_dir / "checkpoint_manifest.json"
        expected = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "identity": identity,
            "completed_panels": {},
        }
        if self.output_dir.exists():
            if not resume:
                raise FileExistsError(
                    f"refusing to overwrite existing output directory without --resume: "
                    f"{self.output_dir}"
                )
            if (self.output_dir / "diagnostics.json").exists() or (
                self.output_dir / "sha256.json"
            ).exists():
                raise FileExistsError(
                    f"refusing to resume finalized immutable output directory: {self.output_dir}"
                )
            self.manifest = self._read_manifest()
            if self.manifest.get("identity") != identity:
                raise ValueError("checkpoint config/base/source/hash identity mismatch")
            self._validate_all_panels()
        else:
            if resume:
                raise FileNotFoundError(
                    f"--resume requires an existing checkpoint output directory: "
                    f"{self.output_dir}"
                )
            self.output_dir.parent.mkdir(parents=True, exist_ok=True)
            self.output_dir.mkdir()
            self.manifest = expected
            _atomic_write_json(self.manifest_path, self.manifest)

    def _read_manifest(self) -> dict:
        if not self.manifest_path.is_file():
            raise ValueError("resume directory has no checkpoint manifest")
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("checkpoint manifest is corrupt") from exc
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
            or set(manifest) != {"schema_version", "identity", "completed_panels"}
            or not isinstance(manifest.get("completed_panels"), dict)
        ):
            raise ValueError("checkpoint manifest schema mismatch")
        return manifest

    def _validate_all_panels(self) -> None:
        for key in self.manifest["completed_panels"]:
            if not key.isdigit() or str(int(key)) != key:
                raise ValueError("checkpoint manifest has invalid panel index")
            self.load_panel(int(key))

    def load_panel(self, panel_index: int) -> dict | None:
        entry = self.manifest["completed_panels"].get(str(int(panel_index)))
        if entry is None:
            return None
        expected_entry_keys = {
            "panel_index",
            "file",
            "npz_sha256",
            "arrays",
            "fields",
        }
        if (
            not isinstance(entry, dict)
            or set(entry) != expected_entry_keys
            or entry["panel_index"] != int(panel_index)
            or entry["file"] != f"panel_{int(panel_index):03d}.npz"
            or not isinstance(entry["arrays"], dict)
            or set(entry["arrays"]) != set(PANEL_ARRAY_FIELDS)
            or not isinstance(entry["fields"], dict)
            or set(entry["fields"]) != set(PANEL_SCALAR_FIELDS)
        ):
            raise ValueError(f"checkpoint panel {panel_index} manifest schema mismatch")
        path = self.output_dir / entry["file"]
        if path.name != entry["file"] or not path.is_file():
            raise ValueError(f"checkpoint panel {panel_index} file is missing")
        if sha256_file(path) != entry["npz_sha256"]:
            raise ValueError(f"checkpoint panel {panel_index} SHA256 mismatch")
        try:
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != set(PANEL_ARRAY_FIELDS):
                    raise ValueError("array field set mismatch")
                arrays = {}
                for name in PANEL_ARRAY_FIELDS:
                    value = archive[name]
                    description = entry["arrays"][name]
                    if (
                        not isinstance(description, dict)
                        or set(description) != {"dtype", "shape", "sha256"}
                        or str(value.dtype) != description["dtype"]
                        or list(value.shape) != description["shape"]
                        or array_sha256(value) != description["sha256"]
                    ):
                        raise ValueError(f"array integrity mismatch: {name}")
                    arrays[name] = value.copy()
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError(f"checkpoint panel {panel_index} is corrupt") from exc
        fields = entry["fields"]
        if (
            fields["estimator"] != DOCUMENT_SCORE_ESTIMATOR
            or not isinstance(fields["conditional_converged"], bool)
            or not all(
                isinstance(fields[name], (int, float)) and np.isfinite(fields[name])
                for name in (
                    "conditional_query_scramble_se_max",
                    "conditional_query_terminal_delta_max",
                    "max_final_order",
                    "mean_final_order",
                )
            )
        ):
            raise ValueError(f"checkpoint panel {panel_index} field integrity mismatch")
        return {**arrays, **fields}

    def save_panel(self, panel_index: int, result: dict) -> None:
        key = str(int(panel_index))
        if key in self.manifest["completed_panels"]:
            raise ValueError(f"checkpoint panel {panel_index} already exists")
        arrays = {
            name: np.asarray(result[name])
            for name in PANEL_ARRAY_FIELDS
        }
        if any(value.dtype.hasobject for value in arrays.values()):
            raise TypeError("checkpoint arrays may not contain Python objects")
        fields = {name: result[name] for name in PANEL_SCALAR_FIELDS}
        if any(isinstance(value, np.generic) for value in fields.values()):
            fields = {
                name: value.item() if isinstance(value, np.generic) else value
                for name, value in fields.items()
            }
        filename = f"panel_{int(panel_index):03d}.npz"
        path = self.output_dir / filename
        _atomic_write_npz(path, arrays)
        entry = {
            "panel_index": int(panel_index),
            "file": filename,
            "npz_sha256": sha256_file(path),
            "arrays": {
                name: {
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "sha256": array_sha256(value),
                }
                for name, value in arrays.items()
            },
            "fields": fields,
        }
        updated = {
            **self.manifest,
            "completed_panels": {
                **self.manifest["completed_panels"],
                key: entry,
            },
        }
        _atomic_write_json(self.manifest_path, updated)
        self.manifest = updated
        self.clear_scramble_progress(panel_index)

    def _partials_dir(self) -> Path:
        return self.output_dir / "scramble_partials"

    def _partial_manifest_path(self, panel_index: int) -> Path:
        return self._partials_dir() / f"panel_{int(panel_index):03d}.json"

    def _empty_partial_manifest(self, panel_index: int) -> dict:
        return {
            "schema_version": SCRAMBLE_PARTIAL_SCHEMA_VERSION,
            "panel_index": int(panel_index),
            "identity": self.manifest["identity"],
            "outer_completed": {},
            "sampling": None,
        }

    def write_heartbeat(self, **payload) -> None:
        self._partials_dir().mkdir(parents=True, exist_ok=True)
        _atomic_write_json(
            self._partials_dir() / "heartbeat.json",
            {
                **payload,
                "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
        )

    def load_scramble_progress(self, panel_index: int) -> dict:
        path = self._partial_manifest_path(panel_index)
        if not path.is_file():
            return {"outer": {}, "sampling": None}
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"scramble partial manifest for panel {panel_index} is corrupt"
            ) from exc
        expected_keys = {
            "schema_version",
            "panel_index",
            "identity",
            "outer_completed",
            "sampling",
        }
        if (
            not isinstance(manifest, dict)
            or set(manifest) != expected_keys
            or manifest.get("schema_version") != SCRAMBLE_PARTIAL_SCHEMA_VERSION
            or manifest.get("panel_index") != int(panel_index)
            or manifest.get("identity") != self.manifest["identity"]
            or not isinstance(manifest.get("outer_completed"), dict)
        ):
            raise ValueError(
                f"scramble partial manifest for panel {panel_index} schema mismatch"
            )
        outer = {}
        for key, entry in manifest["outer_completed"].items():
            if not key.isdigit() or str(int(key)) != key:
                raise ValueError("checkpoint scramble key is invalid")
            outer[int(key)] = self._load_partial_entry(
                entry,
                OUTER_SCRAMBLE_ARRAY_FIELDS,
                OUTER_SCRAMBLE_SCALAR_FIELDS,
            )
        sampling = None
        if manifest["sampling"] is not None:
            sampling = self._load_partial_entry(
                manifest["sampling"],
                SAMPLING_PARTIAL_ARRAY_FIELDS,
                SAMPLING_PARTIAL_SCALAR_FIELDS,
            )
        return {"outer": outer, "sampling": sampling}

    def _load_partial_entry(self, entry, array_fields, scalar_fields) -> dict:
        expected_entry_keys = {"file", "npz_sha256", "arrays", "fields"}
        if (
            not isinstance(entry, dict)
            or set(entry) != expected_entry_keys
            or not isinstance(entry.get("file"), str)
            or not isinstance(entry.get("arrays"), dict)
            or set(entry["arrays"]) != set(array_fields)
            or not isinstance(entry.get("fields"), dict)
            or set(entry["fields"]) != set(scalar_fields)
        ):
            raise ValueError("scramble partial entry schema mismatch")
        path = self._partials_dir() / entry["file"]
        if path.name != entry["file"] or not path.is_file():
            raise ValueError("scramble partial file is missing")
        if sha256_file(path) != entry["npz_sha256"]:
            raise ValueError("scramble partial SHA256 mismatch")
        try:
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != set(array_fields):
                    raise ValueError("array field set mismatch")
                arrays = {}
                for name in array_fields:
                    value = archive[name]
                    description = entry["arrays"][name]
                    if (
                        not isinstance(description, dict)
                        or set(description) != {"dtype", "shape", "sha256"}
                        or str(value.dtype) != description["dtype"]
                        or list(value.shape) != description["shape"]
                        or array_sha256(value) != description["sha256"]
                    ):
                        raise ValueError(f"array integrity mismatch: {name}")
                    arrays[name] = value.copy()
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError("scramble partial file is corrupt") from exc
        fields = _jsonable_fields(entry["fields"])
        if "converged" in scalar_fields and not isinstance(fields["converged"], bool):
            raise ValueError("scramble partial converged flag is invalid")
        return {**arrays, **fields}

    def save_outer_scramble(self, panel_index: int, scramble: int, summary: dict) -> None:
        arrays = {name: np.asarray(summary[name]) for name in OUTER_SCRAMBLE_ARRAY_FIELDS}
        fields = {name: summary[name] for name in OUTER_SCRAMBLE_SCALAR_FIELDS}
        filename = f"panel_{int(panel_index):03d}_outer_{int(scramble):02d}.npz"
        self._save_partial_piece(
            panel_index,
            kind="outer",
            key=str(int(scramble)),
            filename=filename,
            arrays=arrays,
            fields=fields,
        )
        self.write_heartbeat(
            panel_index=int(panel_index),
            stage="outer_scramble",
            scramble=int(scramble) + 1,
        )

    def save_sampling_partial(self, panel_index: int, summary: dict) -> None:
        arrays = {name: np.asarray(summary[name]) for name in SAMPLING_PARTIAL_ARRAY_FIELDS}
        fields = {name: summary[name] for name in SAMPLING_PARTIAL_SCALAR_FIELDS}
        filename = f"panel_{int(panel_index):03d}_sampling.npz"
        self._save_partial_piece(
            panel_index,
            kind="sampling",
            key=None,
            filename=filename,
            arrays=arrays,
            fields=fields,
        )
        self.write_heartbeat(panel_index=int(panel_index), stage="sampling")

    def _save_partial_piece(
        self,
        panel_index: int,
        *,
        kind: str,
        key: str | None,
        filename: str,
        arrays: dict[str, np.ndarray],
        fields: dict,
    ) -> None:
        if any(value.dtype.hasobject for value in arrays.values()):
            raise TypeError("checkpoint arrays may not contain Python objects")
        fields = _jsonable_fields(fields)
        self._partials_dir().mkdir(parents=True, exist_ok=True)
        manifest_path = self._partial_manifest_path(panel_index)
        if manifest_path.is_file():
            progress_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if progress_manifest.get("identity") != self.manifest["identity"]:
                raise ValueError("scramble partial identity mismatch")
        else:
            progress_manifest = self._empty_partial_manifest(panel_index)
        if kind == "outer":
            if key in progress_manifest["outer_completed"]:
                raise ValueError(
                    f"checkpoint panel {panel_index} scramble {key} already exists"
                )
        elif progress_manifest["sampling"] is not None:
            raise ValueError(f"checkpoint panel {panel_index} sampling already exists")
        path = self._partials_dir() / filename
        _atomic_write_npz(path, arrays)
        entry = {
            "file": filename,
            "npz_sha256": sha256_file(path),
            "arrays": {
                name: {
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "sha256": array_sha256(value),
                }
                for name, value in arrays.items()
            },
            "fields": fields,
        }
        if kind == "outer":
            progress_manifest["outer_completed"] = {
                **progress_manifest["outer_completed"],
                key: entry,
            }
        else:
            progress_manifest["sampling"] = entry
        _atomic_write_json(manifest_path, progress_manifest)

    def clear_scramble_progress(self, panel_index: int) -> None:
        manifest_path = self._partial_manifest_path(panel_index)
        if not manifest_path.is_file():
            return
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            names = [entry.get("file") for entry in manifest.get("outer_completed", {}).values()]
            if isinstance(manifest.get("sampling"), dict):
                names.append(manifest["sampling"].get("file"))
        except (OSError, UnicodeError, json.JSONDecodeError, AttributeError):
            names = []
        for name in names:
            if not name:
                continue
            path = self._partials_dir() / name
            if path.is_file():
                path.unlink()
        manifest_path.unlink()


def _jsonable_fields(fields: dict) -> dict:
    converted = {}
    for name, value in fields.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bool) or value is None:
            converted[name] = value
        elif isinstance(value, int):
            converted[name] = int(value)
        elif isinstance(value, float):
            converted[name] = float(value)
        else:
            raise TypeError(f"checkpoint field {name} is not JSON-safe")
    return converted


def _release_cuda_workspace() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def validate_high_precision_config(payload: dict) -> None:
    if payload.get("phase") != 2 or payload.get("not_formal") is not True:
        raise ValueError("high-precision score centering requires phase=2 and not_formal=true")
    if payload.get("schema_version") != "p4-phase2-score-centering-config-v1":
        raise ValueError("unsupported high-precision score-centering config schema")
    gate = float(payload["risk_fraction_of_half_phi_max"])
    if gate <= 0 or gate > MAX_SCORE_CENTERING_GATE:
        raise ValueError("score-centering risk gate may not exceed 0.01")
    if int(payload["Bmax"]) != 32000:
        raise ValueError("Phase-2 score-centering certificate requires Bmax=32000")
    outer = payload["outer_qmc"]
    conditional = payload["conditional_qmc"]
    if int(outer["order"]) <= 6 or int(outer["scrambles"]) <= 3:
        raise ValueError("high-precision outer QMC must exceed the failed order/scrambles")
    if int(conditional["max_order"]) <= 10 or int(conditional["scrambles"]) <= 4:
        raise ValueError("high-precision conditional QMC must exceed the failed order/scrambles")
    if int(conditional["start_order"]) >= int(conditional["max_order"]):
        raise ValueError("adaptive conditional QMC requires start_order < max_order")


def active_panel_indices(
    probabilities: np.ndarray,
    numerical_probability_threshold: float,
    required_top_count: int,
    proposed_indices=None,
):
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1 or np.any(probabilities < 0):
        raise ValueError("p_star must be a nonnegative vector")
    if not np.isclose(probabilities.sum(), 1.0, rtol=0, atol=1e-10):
        raise ValueError("p_star must sum to one")
    threshold = float(numerical_probability_threshold)
    if threshold < 0:
        raise ValueError("numerical probability threshold must be nonnegative")
    required = set(np.flatnonzero(probabilities > threshold).tolist())
    top = set(np.argsort(probabilities)[-int(required_top_count):].tolist())
    required |= top
    active = sorted(required if proposed_indices is None else set(map(int, proposed_indices)))
    missing = sorted(required - set(active))
    if missing:
        raise ValueError(f"active panels omit non-ignored p_star mass: {missing}")
    ignored = np.ones(len(probabilities), dtype=bool)
    ignored[active] = False
    truncated_mass = float(probabilities[ignored].sum())
    return np.asarray(active, dtype=int), {
        "numerical_probability_threshold": threshold,
        "active_panel_count": len(active),
        "active_indices": active,
        "covered_probability_mass": float(probabilities[active].sum()),
        "truncated_mass": truncated_mass,
        "truncated_mass_upper_bound": truncated_mass,
        "required_top_indices": sorted(top),
    }


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_verified_base_artifact(payload: dict, repository_root=REPOSITORY_ROOT):
    root = Path(repository_root).resolve()
    base = payload["base_artifact"]
    directory = _resolve(root, base["directory"])
    config_path = _resolve(root, base["config"])
    metadata_path = directory / base["metadata"]
    npz_path = directory / base["npz"]
    if sha256_file(config_path) != base["config_sha256"]:
        raise ValueError("base config SHA256 mismatch")
    if sha256_file(metadata_path) != base["metadata_sha256"]:
        raise ValueError("base metadata SHA256 mismatch")
    if sha256_file(npz_path) != base["npz_sha256"]:
        raise ValueError("base npz SHA256 mismatch")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("phase") != 2 or metadata.get("not_formal") is not True:
        raise ValueError("base artifact is not a Phase-2 not_formal artifact")
    if metadata["config_sha256"] != base["config_sha256"]:
        raise ValueError("base artifact/config hash mismatch")
    if metadata["npz_sha256"] != base["npz_sha256"]:
        raise ValueError("base metadata/npz hash mismatch")
    for relative, expected in payload["source_sha256"].items():
        actual = sha256_file(root / relative)
        if actual != expected:
            raise ValueError(f"source SHA256 mismatch: {relative}")
    gold_source = payload.get("gold_source_sha256", payload["source_sha256"])
    for relative, expected in gold_source.items():
        recorded = metadata["source_sha256"].get(relative)
        if recorded is not None and recorded != expected:
            raise ValueError(f"gold source SHA256 mismatch: {relative}")
    arrays = np.load(npz_path, allow_pickle=False)
    checks = {
        "scale": metadata["scale_hash"],
        "beta_true": metadata["beta_true_hash"],
        "F_projected": metadata["F"]["projected_hash"],
        "information_projected": metadata["panel_information"]["projected_hash"],
        "p_star": metadata["p_star_hash"],
    }
    for name, expected in checks.items():
        if array_sha256(arrays[name]) != expected:
            raise ValueError(f"base artifact array hash mismatch: {name}")
    return metadata, arrays, {
        "config_path": config_path,
        "metadata_path": metadata_path,
        "npz_path": npz_path,
    }


def _conditional_summary(
    oracle,
    beta,
    panel,
    observed,
    weights,
    cfg,
    seed,
    *,
    return_row_means=False,
):
    observed = np.asarray(observed)
    weights = np.asarray(weights, dtype=float)
    mu = np.asarray(oracle.mu(beta), dtype=float)
    if mu.shape != (len(beta),):
        raise ValueError(f"oracle mu must have shape {(len(beta),)}, got {mu.shape}")
    values, diagnostics = tilted_conditional_mean_exact(
        oracle.mixture,
        beta,
        observed,
        panel,
        seed=int(seed),
        scale=oracle.scale,
        feature_fn=oracle.feature_fn,
        start_order=int(cfg["start_order"]),
        max_order=int(cfg["max_order"]),
        atol=float(cfg["terminal_delta_atol"]),
        rtol=float(cfg["terminal_delta_rtol"]),
        scrambles=int(cfg["scrambles"]),
        scramble_se_atol=float(cfg["scramble_se_atol"]),
        scramble_se_rtol=float(cfg["scramble_se_rtol"]),
        return_diagnostics=True,
    )
    score_mean = weights @ values - mu * weights.sum()
    row_final_order = np.asarray(diagnostics["row_final_order"], dtype=int)
    scramble_se = float(diagnostics["scramble_se"] or 0.0)
    terminal_delta = float(diagnostics["max_abs_delta"] or 0.0)
    return {
        "score_mean": score_mean,
        "conditional_se": np.full(len(beta), scramble_se),
        "terminal_delta": np.full(len(beta), terminal_delta),
        "query_scramble_se_max": scramble_se,
        "query_terminal_delta_max": terminal_delta,
        "converged": bool(diagnostics["converged"]),
        "row_means": values if return_row_means else None,
        "row_final_order": row_final_order,
        "max_final_order": int(row_final_order.max()) if len(row_final_order) else int(cfg["max_order"]),
        "mean_final_order": float(row_final_order.mean()) if len(row_final_order) else float(cfg["max_order"]),
        "n_active_by_order": diagnostics.get("n_active_by_order", {}),
        "estimator": DOCUMENT_SCORE_ESTIMATOR,
        "reference_mean": mu,
    }


def _risk_metrics(delta, fisher, information, p_star, Bmax, half_phi):
    M = np.tensordot(p_star, information, axes=(0, 0))
    chol_M = np.linalg.cholesky((M + M.T) / 2)
    chol_F = np.linalg.cholesky((fisher + fisher.T) / 2)
    transformed = np.linalg.solve(chol_M.T, np.linalg.solve(chol_M, delta))
    induced = float(Bmax) * float(transformed @ fisher @ transformed)
    return {
        "euclidean_norm": float(np.linalg.norm(delta)),
        "fisher_whitened_norm": float(np.linalg.norm(np.linalg.solve(chol_F, delta))),
        "Bmax_induced_risk_bias": induced,
        "risk_fraction_of_half_phi": float(induced / half_phi),
    }


def _weighted_allocation_vectors(panel_results, active, p_star):
    """p*-weighted active-panel mean; inactive mass is bounded separately."""
    p_star = np.asarray(p_star, dtype=float)
    active = [int(index) for index in active]
    missing = [index for index in active if int(index) not in panel_results]
    if missing:
        raise ValueError(f"allocation requires all active panels: {missing}")
    weights = p_star[active][:, None]

    def stacked(name):
        return np.stack([np.asarray(panel_results[int(index)][name]) for index in active])

    def rss(name):
        return np.sqrt(np.sum(np.square(weights * stacked(name)), axis=0))

    return {
        "delta": np.sum(weights * stacked("delta"), axis=0),
        "outer_numerical_se": rss("outer_numerical_se"),
        "conditional_numerical_se": rss("conditional_numerical_se"),
        "sampling_se": rss("sampling_se"),
    }


def _panel_score_mean(
    oracle,
    beta,
    panel,
    cfg,
    panel_rank,
    *,
    checkpoint_store: PanelCheckpointStore | None = None,
    panel_index: int | None = None,
):
    outer = cfg["outer_qmc"]
    n_scrambles = int(outer["scrambles"])
    progress = {"outer": {}, "sampling": None}
    if checkpoint_store is not None and panel_index is not None:
        progress = checkpoint_store.load_scramble_progress(int(panel_index))
        if progress["outer"] or progress["sampling"] is not None:
            print(
                f"[hp-centering] index panel={panel} resume_partial "
                f"outer={len(progress['outer'])}/{n_scrambles} "
                f"sampling={progress['sampling'] is not None}",
                flush=True,
            )
    outer_values = []
    conditional_se_values = []
    terminal_values = []
    final_order_means = []
    query_se_max = 0.0
    query_delta_max = 0.0
    all_converged = True
    max_final_order = 0
    for scramble in range(n_scrambles):
        if scramble in progress["outer"]:
            summary = progress["outer"][scramble]
            resumed = " resumed"
        else:
            seed = int(outer["seed"]) + 1_000_003 * scramble + 8191 * panel_rank
            x, _phi, weights = oracle._outer_qmc(int(outer["order"]), seed, beta)
            summary = _conditional_summary(
                oracle,
                beta,
                panel,
                x[:, list(panel)],
                weights,
                cfg["conditional_qmc"],
                int(cfg["conditional_qmc"]["seed"])
                + 10_000_019 * scramble
                + 8191 * panel_rank,
            )
            if checkpoint_store is not None and panel_index is not None:
                checkpoint_store.save_outer_scramble(int(panel_index), scramble, summary)
            _release_cuda_workspace()
            resumed = ""
        outer_values.append(summary["score_mean"])
        conditional_se_values.append(summary["conditional_se"])
        terminal_values.append(summary["terminal_delta"])
        final_order_means.append(summary["mean_final_order"])
        query_se_max = max(query_se_max, summary["query_scramble_se_max"])
        query_delta_max = max(query_delta_max, summary["query_terminal_delta_max"])
        max_final_order = max(max_final_order, summary["max_final_order"])
        all_converged &= bool(summary["converged"])
        active = summary.get("n_active_by_order")
        extra = f" n_active_by_order={active}" if active else ""
        print(
            f"[hp-centering] index panel={panel} outer_scramble={scramble + 1}/"
            f"{n_scrambles} mean_final_order={summary['mean_final_order']:.2f} "
            f"max_final_order={summary['max_final_order']} "
            f"converged={summary['converged']}{resumed}{extra}",
            flush=True,
        )
        if not summary["converged"]:
            print(
                f"[hp-centering] T-04 did not fully converge at panel={panel} "
                f"outer_scramble={scramble + 1}; per doc, certifying risk-relevant "
                f"weighted mean (not per-query), so continuing all scrambles/sampling",
                flush=True,
            )
    outer_values = np.stack(outer_values)
    n_outer = len(outer_values)
    if n_outer >= 2:
        outer_se = outer_values.std(axis=0, ddof=1) / np.sqrt(n_outer)
    else:
        outer_se = np.full(outer_values.shape[1], np.nan)
    stacked_conditional_se = np.stack(conditional_se_values)
    conditional_se = np.sqrt(np.square(stacked_conditional_se).sum(axis=0)) / n_outer

    # Doc-aligned: do NOT abort on per-query non-convergence. The Phase-2
    # score-centering gate is the risk-relevant weighted mean (1% induced bias),
    # not requiring every query to converge. Always run all scrambles + sampling
    # so the point estimate carries a real numerical/sampling SE. Per-query
    # convergence is still recorded as a diagnostic (conditional_converged).
    sampling = cfg["sampling"]
    if progress["sampling"] is not None:
        sample_summary = progress["sampling"]
        print(
            f"[hp-centering] index panel={panel} sampling resumed",
            flush=True,
        )
    else:
        draws = oracle.tilt_full(
            beta, int(sampling["draws"]), int(sampling["seed"]) + panel_rank
        )
        sample_summary = _conditional_summary(
            oracle,
            beta,
            panel,
            draws[:, list(panel)],
            np.full(len(draws), 1.0 / len(draws)),
            cfg["conditional_qmc"],
            int(cfg["conditional_qmc"]["seed"]) + 500_000_003 + 8191 * panel_rank,
            return_row_means=True,
        )
        if checkpoint_store is not None and panel_index is not None:
            checkpoint_store.save_sampling_partial(int(panel_index), sample_summary)
        _release_cuda_workspace()
    sample_scores = sample_summary["row_means"] - oracle.mu(beta)
    sampling_se = sample_scores.std(axis=0, ddof=1) / np.sqrt(len(sample_scores))
    final_order_means.append(sample_summary["mean_final_order"])
    max_final_order = max(max_final_order, sample_summary["max_final_order"])
    return {
        "delta": outer_values.mean(axis=0),
        "outer_numerical_se": outer_se,
        "conditional_numerical_se": conditional_se,
        "conditional_terminal_delta": np.max(np.stack(terminal_values), axis=0),
        "sampling_se": sampling_se,
        "row_final_order": np.asarray(final_order_means, dtype=float),
        "conditional_query_scramble_se_max": query_se_max,
        "conditional_query_terminal_delta_max": query_delta_max,
        "conditional_converged": bool(all_converged and sample_summary["converged"]),
        "estimator": DOCUMENT_SCORE_ESTIMATOR,
        "max_final_order": int(max_final_order),
        "mean_final_order": float(np.mean(final_order_means)),
    }


def _collect_panel_results(
    oracle,
    beta,
    panels,
    compute_indices,
    payload,
    checkpoint_store: PanelCheckpointStore | None,
    rank_map: dict[int, int],
):
    panel_results = {}
    compute_indices = [int(i) for i in compute_indices]
    for display_i, index in enumerate(compute_indices):
        rank = int(rank_map[int(index)])
        result = (
            checkpoint_store.load_panel(int(index))
            if checkpoint_store is not None
            else None
        )
        if result is not None:
            print(
                f"[hp-centering] panel {display_i + 1}/{len(compute_indices)} index={index} "
                f"panel={panels[index]} resumed"
            )
        else:
            print(
                f"[hp-centering] panel {display_i + 1}/{len(compute_indices)} index={index} "
                f"panel={panels[index]}"
            )
            result = _panel_score_mean(
                oracle,
                beta,
                panels[int(index)],
                payload,
                rank,
                checkpoint_store=checkpoint_store,
                panel_index=int(index),
            )
            if checkpoint_store is not None:
                checkpoint_store.save_panel(int(index), result)
        panel_results[int(index)] = result
    return panel_results


def run_score_centering(
    payload: dict,
    metadata: dict,
    arrays,
    checkpoint_store: PanelCheckpointStore | None = None,
) -> dict:
    mixture = make_frozen_mixture(
        seed=int(metadata["mixture_seed"]), alpha=float(metadata["alpha"])
    )
    if mixture_sha256(mixture) != metadata["mixture_parameter_hash"]:
        raise ValueError("rebuilt mixture does not match base artifact")
    integration = metadata["integration"]
    oracle = OracleMeasure(
        mixture,
        arrays["scale"],
        FullLawQMC(**integration["full"]),
        ConditionalQMC(**integration["conditional"]),
        InformationQMC(**integration["information"]),
    )
    beta = arrays["beta_true"]
    p_star = arrays["p_star"]
    information = arrays["information_projected"]
    fisher = arrays["F_projected"]
    panels = all_pairs()
    active_cfg = payload["active_panels"]
    active, coverage = active_panel_indices(
        p_star,
        float(active_cfg["numerical_probability_threshold"]),
        int(active_cfg["required_top_count"]),
    )
    selected = np.argsort(p_star)[-int(active_cfg["required_top_count"]):][::-1]
    probe = payload.get("probe_indices")
    is_probe = probe is not None
    if is_probe:
        probe_indices = np.asarray(sorted({int(i) for i in probe}), dtype=int)
        missing = sorted(set(probe_indices.tolist()) - set(map(int, active)))
        if missing:
            raise ValueError(f"probe indices are not active panels: {missing}")
        compute_indices = probe_indices
    else:
        compute_indices = active
    panel_results = _collect_panel_results(
        oracle,
        beta,
        panels,
        compute_indices,
        payload,
        checkpoint_store,
        rank_map={int(i): r for r, i in enumerate(active)},
    )

    half_phi = float(metadata["theory_constant_half_phi"])
    Bmax = int(payload["Bmax"])
    rows = []
    for index in selected:
        if int(index) not in panel_results:
            continue
        result = panel_results[int(index)]
        numerical_se = np.hypot(
            result["outer_numerical_se"], result["conditional_numerical_se"]
        )
        rows.append({
            "panel": list(panels[int(index)]),
            "panel_index": int(index),
            "p_star": float(p_star[index]),
            "estimator": result["estimator"],
            "max_final_order": result["max_final_order"],
            "mean_final_order": result["mean_final_order"],
            "delta": result["delta"].tolist(),
            "outer_numerical_se": result["outer_numerical_se"].tolist(),
            "conditional_numerical_se": result["conditional_numerical_se"].tolist(),
            "combined_numerical_se": numerical_se.tolist(),
            "conditional_terminal_delta": result["conditional_terminal_delta"].tolist(),
            "sampling_se": result["sampling_se"].tolist(),
            "conditional_query_scramble_se_max": result["conditional_query_scramble_se_max"],
            "conditional_query_terminal_delta_max": result["conditional_query_terminal_delta_max"],
            "conditional_converged": result["conditional_converged"],
            **_risk_metrics(result["delta"], fisher, information, p_star, Bmax, half_phi),
            "numerical_se_risk": _risk_metrics(
                numerical_se, fisher, information, p_star, Bmax, half_phi
            ),
        })

    computed = [int(i) for i in panel_results]
    M = np.tensordot(p_star, information, axes=(0, 0))
    chol_M = np.linalg.cholesky((M + M.T) / 2)
    Minv = np.linalg.solve(chol_M.T, np.linalg.solve(chol_M, np.eye(M.shape[0])))
    risk_matrix = Minv @ fisher @ Minv
    trunc_radius = 2.0 * coverage["truncated_mass_upper_bound"] * np.sqrt(len(beta))
    probe_rows = []
    for index in computed:
        result = panel_results[int(index)]
        numerical_se = np.hypot(
            result["outer_numerical_se"], result["conditional_numerical_se"]
        )
        probe_rows.append({
            "panel": list(panels[int(index)]),
            "panel_index": int(index),
            "p_star": float(p_star[index]),
            **_risk_metrics(result["delta"], fisher, information, p_star, Bmax, half_phi),
            "conditional_converged": result["conditional_converged"],
        })
    if is_probe:
        allocation = None
        point_gate = bool(probe_rows) and max(
            row["risk_fraction_of_half_phi"] for row in probe_rows
        ) < float(payload["risk_fraction_of_half_phi_max"])
        numerical_se_risk = None
        # Doc-aligned Phase-2 gate: the risk-relevant weighted mean must induce
        # < 1% of half-Phi. Per-query convergence is recorded (conditional_gate)
        # only as a diagnostic, not a hard certificate requirement.
        conditional_gate = all(
            panel_results[int(index)]["conditional_converged"] for index in computed
        )
        certificate = bool(point_gate)
    else:
        vectors = _weighted_allocation_vectors(panel_results, active, p_star)
        active_delta = vectors["delta"]
        outer_se = vectors["outer_numerical_se"]
        conditional_se = vectors["conditional_numerical_se"]
        sampling_se = vectors["sampling_se"]
        combined_se = np.hypot(outer_se, conditional_se)
        allocation_metrics = _risk_metrics(
            active_delta, fisher, information, p_star, Bmax, half_phi
        )
        active_quadratic_root = np.sqrt(allocation_metrics["Bmax_induced_risk_bias"] / Bmax)
        trunc_quadratic_root = np.sqrt(np.linalg.eigvalsh(risk_matrix).max()) * trunc_radius
        risk_upper = Bmax * (active_quadratic_root + trunc_quadratic_root) ** 2
        allocation = {
            "delta": active_delta.tolist(),
            "outer_numerical_se": outer_se.tolist(),
            "conditional_numerical_se": conditional_se.tolist(),
            "combined_numerical_se": combined_se.tolist(),
            "sampling_se": sampling_se.tolist(),
            **allocation_metrics,
            "truncation_euclidean_norm_upper_bound": trunc_radius,
            "Bmax_induced_risk_with_truncation_upper_bound": float(risk_upper),
            "risk_fraction_with_truncation_upper_bound": float(risk_upper / half_phi),
            "numerical_se_risk": _risk_metrics(
                combined_se, fisher, information, p_star, Bmax, half_phi
            ),
        }
        point_gate = max(
            [allocation["risk_fraction_with_truncation_upper_bound"]]
            + [row["risk_fraction_of_half_phi"] for row in rows]
        ) < float(payload["risk_fraction_of_half_phi_max"])
        numerical_se_risk = max(
            [allocation["numerical_se_risk"]["risk_fraction_of_half_phi"]]
            + [row["numerical_se_risk"]["risk_fraction_of_half_phi"] for row in rows]
        )
        conditional_gate = all(
            panel_results[int(index)]["conditional_converged"] for index in active
        )
        # Doc-aligned Phase-2 gate = 1% induced-bias on the risk-relevant
        # weighted mean. conditional_gate kept as a diagnostic only.
        certificate = bool(point_gate)
    return {
        "schema_version": "p4-phase2-score-centering-diagnostics-v1",
        "phase": 2,
        "not_formal": True,
        "probe": bool(is_probe),
        "precision_id": payload["precision_id"],
        "estimator": DOCUMENT_SCORE_ESTIMATOR,
        "Bmax": Bmax,
        "risk_fraction_of_half_phi_max": float(payload["risk_fraction_of_half_phi_max"]),
        "numerical_se_risk_fraction_observed": None if numerical_se_risk is None else float(numerical_se_risk),
        "base_artifact": {
            "artifact_sha256": payload["base_artifact"]["npz_sha256"],
            "config_sha256": payload["base_artifact"]["config_sha256"],
            "theory_constant_half_phi": half_phi,
        },
        "qmc_configuration_used": {
            "outer_qmc": payload["outer_qmc"],
            "conditional_qmc": payload["conditional_qmc"],
            "sampling": payload["sampling"],
        },
        "coverage": coverage,
        "selected_panels": rows,
        "probe_panels": probe_rows if is_probe else None,
        "active_panel_diagnostics": {
            str(index): {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in panel_results[int(index)].items()
            }
            for index in computed
        },
        "allocation": allocation,
        "point_estimate_gate_passed": bool(point_gate),
        "conditional_qmc_gate_passed": bool(conditional_gate),
        "probe_gate_passed": bool(is_probe and point_gate),
        "score_centering_passed": bool(certificate),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/p4_phase2_gold_doc_adaptive_20260827.yaml")
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume only from a matching, non-finalized per-panel checkpoint directory",
    )
    parser.add_argument(
        "--only-indices",
        type=int,
        nargs="+",
        help="probe subset of active panels; not a Phase-2 certificate",
    )
    args = parser.parse_args()
    root = REPOSITORY_ROOT
    config_path = _resolve(root, args.config)
    out = _resolve(root, args.out)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))[
        "p4_phase2_score_centering"
    ]
    validate_high_precision_config(payload)
    if args.only_indices:
        payload["probe_indices"] = list(args.only_indices)
    metadata, arrays, verified_paths = load_verified_base_artifact(payload, root)
    checkpoint_store = PanelCheckpointStore(
        out, checkpoint_identity(payload), resume=args.resume
    )
    log_stream = (out / "run.log").open(
        "a" if args.resume else "w", encoding="utf-8"
    )
    sys.stdout = _Tee(sys.__stdout__, log_stream)
    sys.stderr = _Tee(sys.__stderr__, log_stream)
    started = perf_counter()
    diagnostics = run_score_centering(
        payload, metadata, arrays, checkpoint_store=checkpoint_store
    )
    diagnostics["provenance"] = {
        "config_sha256": sha256_file(config_path),
        "config_canonical_sha256": canonical_sha256(payload),
        "source_sha256": {
            "scripts/p4_phase2_score_centering.py": sha256_file(Path(__file__)),
            **payload["source_sha256"],
        },
        "base_paths": {key: str(value.relative_to(root)).replace("\\", "/")
                       for key, value in verified_paths.items()},
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "code_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).strip()),
        "device": "cuda:0",
        "elapsed_seconds": float(perf_counter() - started),
    }
    diagnostics_path = out / "diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
    )
    hash_manifest = {
        path.name: sha256_file(path)
        for path in sorted(out.iterdir())
        if path.is_file() and path.name != "sha256.json"
    }
    (out / "sha256.json").write_text(
        json.dumps(hash_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    summary = {
        "output": str(out),
        "probe": bool(diagnostics.get("probe")),
        "score_centering_passed": diagnostics["score_centering_passed"],
        "probe_gate_passed": diagnostics.get("probe_gate_passed"),
    }
    if diagnostics.get("allocation"):
        summary["allocation_risk_fraction"] = diagnostics["allocation"][
            "risk_fraction_with_truncation_upper_bound"
        ]
    if diagnostics["selected_panels"]:
        summary["max_selected_risk_fraction"] = max(
            row["risk_fraction_of_half_phi"] for row in diagnostics["selected_panels"]
        )
    elif diagnostics.get("probe_panels"):
        summary["max_probe_risk_fraction"] = max(
            row["risk_fraction_of_half_phi"] for row in diagnostics["probe_panels"]
        )
    print(json.dumps(summary, sort_keys=True))
    if diagnostics.get("probe"):
        if not diagnostics.get("probe_gate_passed"):
            raise SystemExit(2)
    elif not diagnostics["score_centering_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
