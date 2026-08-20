"""Versioned configuration loading and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Paths:
    root: Path
    data: Path
    results: Path
    figures: Path
    experiments: Path


@dataclass(frozen=True)
class Config:
    seed: int
    device: str
    dtype: str
    paths: Paths


def _read_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML config requires PyYAML") from exc
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a mapping")
    return value


def load_config(path: str | Path) -> Config:
    config_path = Path(path).resolve()
    raw = _read_mapping(config_path)
    required = {"seed", "device", "dtype", "paths"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"missing configuration keys: {sorted(missing)}")
    seed = raw["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    device = str(raw["device"]).lower()
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    dtype = str(raw["dtype"]).lower()
    if dtype not in {"float32", "float64"}:
        raise ValueError("dtype must be float32 or float64")
    path_values = raw["paths"]
    if not isinstance(path_values, dict):
        raise ValueError("paths must be a mapping")
    root = Path(path_values.get("root", "."))
    if not root.is_absolute():
        root = (config_path.parent / root).resolve()
    resolved = {name: (root / Path(path_values.get(name, name))).resolve() for name in ("data", "results", "figures", "experiments")}
    return Config(seed=seed, device=device, dtype=dtype, paths=Paths(root=root, **resolved))


def config_to_dict(config: Config) -> dict[str, Any]:
    return {"seed": config.seed, "device": config.device, "dtype": config.dtype, "paths": {"root": str(config.paths.root), "data": str(config.paths.data), "results": str(config.paths.results), "figures": str(config.paths.figures), "experiments": str(config.paths.experiments)}}

