"""Train and validate the canonical P6 Synthetic VAEAC generator."""
import time
import argparse
from pathlib import Path
import numpy as np
import torch

from rr_gid_cn.synthetic_oracle import all_pairs, make_frozen_mixture, reference_scale, sample_conditional_batch, sample_full
from rr_gid_cn.vaeac import VAEAC, VAEACGenerator, train_vaeac


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=Path('experiments/p6_vaeac_synthetic_a800.pt'))
    ap.add_argument('--log', type=Path, default=Path('experiments/p6_vaeac_a800_train.json'))
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    mix = make_frozen_mixture(seed=2026, alpha=1.0)
    scale = reference_scale(mix, 6000, 2026)
    reference = sample_full(mix, 50000, 2026)
    validation = sample_full(mix, 10000, 2027)
    panels = all_pairs()
    t0 = time.time()
    model = VAEAC(dim=16, latent=16, hidden=256, seed=args.seed)
    model, hist = train_vaeac(model, reference, panels, scale, alpha=1.0,
                              epochs=args.epochs, batch=512, lr=1e-3,
                              device=args.device, seed=args.seed, beta=1.0)
    print(f"trained in {time.time()-t0:.0f}s; loss {hist[0]:.2f} -> {hist[-1]:.2f}", flush=True)
    gen = VAEACGenerator(model, scale, alpha=1.0)
    samples = gen.sample_full(20000, 1)
    print("sample mean max err:", round(float(np.abs(samples.mean(0) - reference.mean(0)).max()), 3), flush=True)
    print("sample std max err:", round(float(np.abs(samples.std(0) - reference.std(0)).max()), 3), flush=True)
    panel = (0, 6)
    cond = gen.sample_conditional_batch(validation[:8, list(panel)], panel, 1000, 42)
    truth = sample_conditional_batch(mix, validation[:8, list(panel)], panel, 1000, 43)
    mean_rmse = float(np.sqrt(np.mean((cond.mean(1) - truth.mean(1)) ** 2)))
    std_rmse = float(np.sqrt(np.mean((cond.std(1) - truth.std(1)) ** 2)))
    print("cond mean RMSE", round(mean_rmse, 3), "cond std RMSE", round(std_rmse, 3), flush=True)
    print("importance ESS/N at beta=0:", round(gen.importance_ess(np.zeros(12), samples), 3), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "data_mean": model.data_mean,
                "data_std": model.data_std, "scale": scale, "alpha": 1.0,
                "reference_n": len(reference), "validation_n": len(validation)},
               args.output)
    record = {"output": str(args.output), "device": args.device, "epochs": args.epochs,
              "train_seconds": time.time()-t0, "loss_first": hist[0], "loss_last": hist[-1],
              "full_mean_max_error": float(np.abs(samples.mean(0) - reference.mean(0)).max()),
              "full_std_max_error": float(np.abs(samples.std(0) - reference.std(0)).max()),
              "conditional_mean_rmse": mean_rmse, "conditional_std_rmse": std_rmse}
    args.log.write_text(__import__('json').dumps(record, indent=2) + '\n', encoding='utf-8')
    print("model saved", args.output, flush=True)


if __name__ == "__main__":
    main()
