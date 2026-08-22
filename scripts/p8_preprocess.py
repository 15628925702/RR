"""P8 Gas Sensor preprocessing: load UCI batches, split, standardize, PC1 feature map.

PDF Sec. 8.1: batches 1-6 reference (80/20 train/validation), target campaigns
batch 7 / batches 8-9 / batch 10. 128 raw features are standardized with the
reference-train mean/std; each 16-sensor block gets its PC1; the relative-shift
feature map is phi_j = tanh(z_j), phi_{8+j} = tanh(z_j z_{j+8}), r = 16.
"""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np

DATA_DIR = Path("data/gas")
OUT_DIR = Path("data/gas/processed")


def load_batch(fn: Path, n_features: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Parse a UCI Gas batch file (SVMLight rows with a leading 'gas;conc' token)."""
    rows, classes = [], []
    for line in open(fn, encoding="utf-8"):
        parts = line.strip().split()
        first = parts[0].split(";")
        cls = float(first[1])
        vec = np.zeros(n_features, dtype=float)
        for tok in parts[1:]:
            idx, val = tok.split(":")
            vec[int(idx) - 1] = float(val)
        rows.append(vec)
        classes.append(cls)
    return np.asarray(rows), np.asarray(classes)


def load_all_batches(data_dir: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    batches = {}
    for b in range(1, 11):
        fn = data_dir / f"batch{b}.dat"
        batches[b] = load_batch(fn)
    return batches


def panel_library(n_sensors: int = 16):
    return tuple(combinations(range(n_sensors), 2))


def fit_scaler_pca(reference_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(reference_train, dtype=float)
    mean = x.mean(0)
    std = x.std(0, ddof=1)
    std = np.where(std < 1e-12, 1.0, std)
    z = (x - mean) / std
    pcs = []
    for sensor in range(16):
        block = z[:, sensor * 8 : (sensor + 1) * 8]
        _, _, vh = np.linalg.svd(block, full_matrices=False)
        vector = vh[0]
        if vector[np.argmax(np.abs(vector))] < 0:
            vector = -vector
        pcs.append(vector)
    return mean, std, np.asarray(pcs)


def transform_features(x: np.ndarray, mean: np.ndarray, std: np.ndarray, pcs: np.ndarray) -> np.ndarray:
    z = (np.asarray(x) - mean) / std
    scores = np.column_stack([z[:, i * 8 : (i + 1) * 8] @ pcs[i] for i in range(16)])
    return np.concatenate([np.tanh(scores[:, :8]), np.tanh(scores[:, :8] * scores[:, 8:])], axis=1)


def main() -> None:
    batches = load_all_batches(DATA_DIR)
    total = sum(x.shape[0] for x, _ in batches.values())
    # Reference = batches 1-6; targets = batch 7, batches 8-9, batch 10.
    ref_x = np.concatenate([batches[b][0] for b in range(1, 7)], axis=0)
    target_campaigns = {
        "batch7": batches[7][0],
        "batches89": np.concatenate([batches[8][0], batches[9][0]], axis=0),
        "batch10": batches[10][0],
    }
    rng = np.random.default_rng(2026)
    n = len(ref_x)
    perm = rng.permutation(n)
    n_train = int(round(0.8 * n))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    ref_train = ref_x[train_idx]
    ref_val = ref_x[val_idx]
    mean, std, pcs = fit_scaler_pca(ref_train)
    phi_train = transform_features(ref_train, mean, std, pcs)
    phi_val = transform_features(ref_val, mean, std, pcs)
    phi_targets = {k: transform_features(v, mean, std, pcs) for k, v in target_campaigns.items()}
    panels = panel_library()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / "gas_processed.npz",
        ref_train=ref_train, ref_val=ref_val,
        phi_train=phi_train, phi_val=phi_val,
        train_idx=train_idx, val_idx=val_idx,
        mean=mean, std=std, pcs=pcs,
        **{f"phi_{k}": v for k, v in phi_targets.items()},
    )
    # Data card
    card = {
        "source": "UCI Gas Sensor Array Drift at Different Concentrations (dataset 270)",
        "n_batches": 10, "n_total": total,
        "reference_batches": list(range(1, 7)), "reference_rows": int(n),
        "target_campaigns": {k: int(v.shape[0]) for k, v in target_campaigns.items()},
        "ref_train_rows": int(len(train_idx)), "ref_val_rows": int(len(val_idx)),
        "n_features_raw": 128, "n_sensors": 16, "features_per_sensor": 8,
        "feature_dim": 16, "n_panels": len(panels),
        "phi_in_unit_range": bool(np.all(np.abs(phi_train) <= 1.0) and np.all(np.abs(phi_val) <= 1.0)),
        "ref_train_mean_abs": float(np.abs(phi_train.mean(0)).max()),
        "ref_train_std_range": [float(phi_train.std(0).min()), float(phi_train.std(0).max())],
        "split_seed": 2026, "split": "80/20 train/val on reference batches 1-6",
    }
    card["sha256"] = hashlib.sha256(json.dumps(card, sort_keys=True).encode()).hexdigest()
    (OUT_DIR / "gas_data_card.json").write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(card, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
