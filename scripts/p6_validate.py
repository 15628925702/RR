import numpy as np
import torch
from rr_gid_cn.synthetic_oracle import make_frozen_mixture, sample_full, sample_conditional_batch
from rr_gid_cn.vaeac import VAEAC, VAEACGenerator

mix = make_frozen_mixture(seed=2026, alpha=1.0)
val = sample_full(mix, 2000, 2027)
ck = torch.load("experiments/p6_vaeac_synthetic.pt", map_location="cpu", weights_only=False)
model = VAEAC(dim=16, latent=16, hidden=256, seed=0)
model.load_state_dict(ck["model"])
model.data_mean, model.data_std = ck["data_mean"], ck["data_std"]
gen = VAEACGenerator(model, ck["scale"], device="cpu")
scales = val.std(0)
for panel in [(0, 1), (0, 6), (3, 9), (10, 15)]:
    obs = val[:16, list(panel)]
    got = gen.sample_conditional_batch(obs, panel, 400, 11)
    truth = sample_conditional_batch(mix, obs, panel, 400, 12)
    mean_rmse = np.sqrt(np.mean(((got.mean(1) - truth.mean(1)) / scales) ** 2))
    std_rmse = np.sqrt(np.mean(((got.std(1) - truth.std(1)) / scales) ** 2))
    print(panel, "mean_z_rmse", round(float(mean_rmse), 4), "std_z_rmse", round(float(std_rmse), 4))
full = gen.sample_full(10000, 13)
print("full_mean_z", float(np.max(np.abs((full.mean(0) - val.mean(0)) / scales))))
print("full_std_z", float(np.max(np.abs((full.std(0) - val.std(0)) / scales))))
