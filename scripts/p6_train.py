"""Train and evaluate the P6 synthetic VAEAC generator on GPU."""
import time
import numpy as np
import torch

from rr_gid_cn.synthetic_oracle import all_pairs, inverse_warp, make_frozen_mixture, reference_scale, sample_full
from rr_gid_cn.vaeac import VAEAC, VAEACGenerator, train_vaeac


def main() -> None:
    mix = make_frozen_mixture(seed=2026, alpha=1.0)
    scale = reference_scale(mix, 6000, 2026)
    reference = sample_full(mix, 20000, 2026)
    panels = all_pairs()
    t0 = time.time()
    # The synthetic reference is a four-mode warped GMM; a learned mixture
    # prior is required for VAEAC's unconditional sampler to retain modes.
    model = VAEAC(dim=16, latent=128, hidden=512, seed=0, k_prior=4, gmm_prior=True)
    model, hist = train_vaeac(model, reference, panels, scale, alpha=1.0, epochs=250,
                              batch=512, lr=1e-3, device="cuda", seed=0, beta=0.05)
    print(f"trained in {time.time()-t0:.0f}s; loss {hist[0]:.2f} -> {hist[-1]:.2f}", flush=True)
    gen = VAEACGenerator(model, scale, alpha=1.0)
    samples = gen.sample_full(20000, 1)
    rm = reference.mean(0)
    rs = reference.std(0)
    print("sample mean max err:", round(float(np.abs(samples.mean(0) - rm).max()), 3), flush=True)
    print("sample std max err:", round(float(np.abs(samples.std(0) - rs).max()), 3), flush=True)
    z_train = inverse_warp(reference, 1.0)
    z_sample = inverse_warp(samples, 1.0)
    print("Z sample std max:", round(float(z_sample.std(0).max()), 2),
          "vs train", round(float(z_train.std(0).max()), 2), flush=True)
    # conditional quality: conditional mean of missing coords vs true values
    test_x = reference[1000:1010]
    panel = (0, 6)
    errs = []
    for i in range(10):
        obs = test_x[i, list(panel)]
        cond = gen.sample_conditional(obs, panel, 2000, 42 + i)
        true = test_x[i]
        for c in range(16):
            if c in panel:
                continue
            errs.append(abs(float(cond[:, c].mean()) - float(true[c])))
    print("cond mean abs err: median", round(float(np.median(errs)), 3),
          "mean", round(float(np.mean(errs)), 3), flush=True)
    ess = gen.importance_ess(np.zeros(12), samples)
    print("importance ESS/N at beta=0:", round(ess, 3), flush=True)
    torch.save({"model": model.state_dict(), "z_std": model.z_std, "scale": scale, "alpha": 1.0},
               "experiments/p6_vaeac_synthetic.pt")
    print("model saved", flush=True)


if __name__ == "__main__":
    main()
