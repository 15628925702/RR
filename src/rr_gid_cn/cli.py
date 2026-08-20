"""Command-line entry point for the P0 smoke pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import config_to_dict, load_config
from .device import select_device
from .seed import initialize_seed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RR-GID_CN reproducible pipeline")
    parser.add_argument("--config", type=Path, help="YAML or JSON configuration")
    parser.add_argument("--write-manifest", action="store_true", help="write a P0 run manifest")
    args = parser.parse_args(argv)
    if args.config is None:
        parser.print_help()
        return 0
    config = load_config(args.config)
    initialize_seed(config.seed)
    device = select_device(config.device)
    config_payload = json.dumps(config_to_dict(config), sort_keys=True).encode("utf-8")
    manifest = {"stage": "P0", "timestamp_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version, "platform": platform.platform(), "seed": config.seed, "device": device, "dtype": config.dtype, "config": config_to_dict(config), "config_sha256": hashlib.sha256(config_payload).hexdigest()}
    if args.write_manifest:
        config.paths.experiments.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8")
        manifest["manifest_sha256"] = hashlib.sha256(payload).hexdigest()
        (config.paths.experiments / "p0_run_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "P0", "device": device, "seed": config.seed, "manifest_written": args.write_manifest}, sort_keys=True))
    return 0
