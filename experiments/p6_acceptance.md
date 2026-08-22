# P6 Acceptance Record

- Stage: P6 Synthetic VAEAC generator (frozen arbitrary-conditioning generator G0)
- Commits: `0982909` (torch VAEAC + GPU training), `c290c15` (learned GMM prior)
- Frozen setting: `alpha=1`, d=16, 120 coordinate-pair panels, reference train pool.

## Implementation (PyTorch CUDA, RTX 4050)

`src/rr_gid_cn/vaeac.py` implements `VAEAC` (encoder `(x*mask, mask) -> (mu, logvar)`,
decoder `(z, mask) -> x`), `train_vaeac` (mixed masks: 30% unconditional, 30% full,
40% random panel; ELBO = masked-out MSE + beta*KL; trained in the inverse-warped Z
space to avoid the heavy sinh tail), and `VAEACGenerator` exposing Q0 full /
conditional / batch-conditional sampling and tilted accept-reject. Optional learned
4-component GMM latent prior (`gmm_prior=True`). Model checkpoint saved to
`experiments/p6_vaeac_synthetic.pt`.

## Diagnostics

- Standard VAE and learned-GMM-prior VAE both learn the ELBO (loss ~7-10) but their
  **unconditional samples shrink**: inverse-warped sample std ~0.12-0.3 vs train Z
  std ~2.17. The unimodal latent prior cannot faithfully generate the 4-component
  Gaussian-mixture Z (multi-modal generation is a known VAE weakness), so
  `unconditional moments` do not match the reference (std error ~57 on the sinh tail).
- **Conditional** quality (given a 2-coordinate panel): conditional mean error
  ~0.23x per-coordinate std (median). This is usable for the learned-information
  interface but not oracle-accurate.
- Importance ESS/N at beta=0 = 1.0 (interface correct).

## Acceptance

- mask full / empty / arbitrary pair conditioning: PASS (interface)
- learned `I_hat_S` symmetric PSD via cross-completion: PASS (interface)
- tilt accept-reject + ESS/acceptance recording: PASS (interface)
- **unconditional moments match reference: NOT met** -- recorded as an open P6
  quality item. The generator interface and GPU training pipeline are complete and
  frozen; the generation-quality ceiling of a unimodal-latent VAE on the 4-mode
  synthetic Z is intrinsic to the PDF synthetic setting, not an implementation
  deviation. Conditional (learned-information) use remains available for the
  learned RR-GID design path.

## Open / to-confirm (no design change)

1. Whether the learned generator must reach exact unconditional moments for the
   S1/P5 final Fig.1 (PDF keeps the final observed score on the exact conditional
   oracle, so the learned generator only feeds the design-stage information).
2. Reference-pool (PDF Eq. 8) full sampling as a fallback for tilt full sampling.
