"""Run Gas preprocessing fixed-subset smoke."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from rr_gid_cn.gas_preprocess import fit_scaler_pca, panel_library, transform_features


def main() -> None:
    rng = np.random.default_rng(2026)
    raw = rng.normal(size=(200, 128))
    train, validation = raw[:160], raw[160:]
    mean, std, pcs = fit_scaler_pca(train)
    phi = transform_features(validation, mean, std, pcs)
    summary = {"stage": "P8", "reference_train": 160, "reference_validation": 40, "raw_dimension": 128, "sensor_blocks": 16, "panel_count": len(panel_library()), "phi_shape": list(phi.shape), "phi_min": float(phi.min()), "phi_max": float(phi.max())}
    payload = json.dumps(summary, sort_keys=True, indent=2) + "\n"
    Path("results").mkdir(exist_ok=True)
    Path("results/p8_gas_preprocess_summary.json").write_text(payload, encoding="utf-8")
    summary["summary_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    Path("results/p8_gas_preprocess_summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

