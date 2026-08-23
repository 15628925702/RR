"""Train and evaluate the P9 Gas 128-dim VAEAC generator (PDF Sec. 8.1)."""
import time
import numpy as np
import torch

from rr_gid_cn.gas_preprocess import panel_library
from rr_gid_cn.vaeac import VAEAC, VAEACGenerator, train_vaeac


def main() -> None:
    data = np.load("data/gas/processed/gas_processed.npz")
    ref_train = data["ref_train"]
    ref_val = data["ref_val"]
    sensor_pairs = panel_library()
    # 120 sensor-pair panels -> 120 feature-coordinate panels (2 sensors x 8 feats)
    coord_panels = tuple(
        tuple(j for s in pair for j in range(s * 8, (s + 1) * 8)) for pair in sensor_pairs
    )
    print(f"ref_train {ref_train.shape} ref_val {ref_val.shape} panels {len(coord_panels)}", flush=True)
    t0 = time.time()
    # Gas has no sinh warp: alpha=0 -> inverse_warp is identity, the VAE learns the
    # standardized 128-dim sensor features (reference-train mean/std already applied).
    model = VAEAC(dim=128, latent=64, hidden=256, seed=0)
    model, hist = train_vaeac(model, ref_train, coord_panels, scale=None, alpha=0.0, epochs=120,
                              batch=512, lr=1e-3, device="cuda", seed=0, beta=0.5)
    print(f"trained in {time.time()-t0:.0f}s; loss {hist[0]:.2f} -> {hist[-1]:.2f}", flush=True)
    gen = VAEACGenerator(model, np.ones(128), alpha=0.0)
    samples = gen.sample_full(5000, 1)
    rm = ref_train.mean(0)
    rs = ref_train.std(0)
    print("sample mean err (std units):", round(float(np.abs((samples.mean(0) - rm) / np.maximum(rs, 1e-9)).max()), 3), flush=True)
    print("sample std err (std units):", round(float(np.abs((samples.std(0) - rs) / np.maximum(rs, 1e-9)).max()), 3), flush=True)
    # conditional quality on a sensor-pair panel: mean of missing coords vs true
    test_x = ref_val[:10]
    feat_idx = list(coord_panels[0])  # sensors 0,1 -> feature block 0..15
    errs = []
    for i in range(10):
        obs = test_x[i, feat_idx]
        cond = gen.sample_conditional(obs, tuple(feat_idx), 2000, 42 + i)
        for c in range(128):
            if c in feat_idx:
                continue
            errs.append(abs(float(cond[:, c].mean()) - float(test_x[i, c])) / float(max(rs[c], 1e-9)))
    print("conditional mean abs err (median, std units):", round(float(np.median(errs)), 3), flush=True)
    torch.save({"model": model.state_dict(), "data_mean": model.data_mean,
                "data_std": model.data_std, "alpha": 0.0,
                "reference_n": len(ref_train), "validation_n": len(ref_val)},
               "experiments/p9_gas_vaeac.pt")
    print("saved experiments/p9_gas_vaeac.pt", flush=True)


if __name__ == "__main__":
    main()
