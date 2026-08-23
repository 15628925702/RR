"""Canonical VAEAC generator used as the frozen P6 reference model."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .synthetic_oracle import feature_map


class VAEAC(nn.Module):
    """VAEAC with proposal q, conditional prior p and Gaussian decoder."""
    def __init__(self, dim=16, latent=16, hidden=128, seed=0, k_prior=4, gmm_prior=False):
        super().__init__()
        del k_prior, gmm_prior
        torch.manual_seed(seed)
        self.dim, self.latent = int(dim), int(latent)
        self.proposal = nn.Sequential(nn.Linear(2 * dim, hidden), nn.Tanh(),
                                      nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 2 * latent))
        self.conditional_prior = nn.Sequential(nn.Linear(2 * dim, hidden), nn.Tanh(),
                                               nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 2 * latent))
        self.decoder = nn.Sequential(nn.Linear(latent + 2 * dim, hidden), nn.Tanh(),
                                     nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 2 * dim))
        self.gmm_prior = False
        self.data_mean = np.zeros(dim, dtype=np.float64)
        self.data_std = np.ones(dim, dtype=np.float64)

    @staticmethod
    def _split(out):
        mu, logvar = out.chunk(2, dim=-1)
        return mu, torch.clamp(logvar, -8.0, 6.0)

    def encode(self, x, mask):
        return self._split(self.proposal(torch.cat([x, mask], dim=-1)))

    def prior(self, x_masked, mask):
        return self._split(self.conditional_prior(torch.cat([x_masked, mask], dim=-1)))

    def kl_divergence(self, q_mu, q_logvar, p_mu, p_logvar):
        return 0.5 * (p_logvar - q_logvar +
                      (q_logvar.exp() + (q_mu - p_mu).pow(2)) / p_logvar.exp() - 1).sum(-1)

    def reparameterize(self, mu, logvar, seed=None):
        if seed is None:
            eps = torch.randn_like(mu)
        else:
            gen = torch.Generator(device=mu.device).manual_seed(seed)
            eps = torch.randn(mu.shape, device=mu.device, generator=gen)
        return mu + torch.exp(0.5 * logvar) * eps

    def decode_params(self, z, mask, x_masked=None):
        if x_masked is None:
            x_masked = torch.zeros(z.shape[0], self.dim, device=z.device, dtype=z.dtype)
        return self._split(self.decoder(torch.cat([z, x_masked, mask], dim=-1)))

    def decode(self, z, mask, x_masked=None):
        return self.decode_params(z, mask, x_masked)[0]

    def forward(self, x, mask):
        x_masked = x * mask
        q_mu, q_logvar = self.encode(x, mask)
        p_mu, p_logvar = self.prior(x_masked, mask)
        z = self.reparameterize(q_mu, q_logvar)
        recon_mu, recon_logvar = self.decode_params(z, mask, x_masked)
        return recon_mu, recon_logvar, q_mu, q_logvar, p_mu, p_logvar


def train_vaeac(model, reference, panels, scale=None, alpha=1.0, epochs=30, batch=512,
                lr=1e-3, device="cuda", seed=0, beta=1.0, z_normalize=True, free_bits=0.0):
    """Train canonical VAEAC on complete records using its masked ELBO."""
    del scale, alpha, free_bits
    model = model.to(device)
    raw = np.asarray(reference, dtype=np.float64)
    center = raw.mean(0) if z_normalize else np.zeros(model.dim)
    spread = raw.std(0, ddof=1) if z_normalize else np.ones(model.dim)
    spread = np.where(spread < 1e-6, 1.0, spread)
    model.data_mean, model.data_std = center.astype(np.float64), spread.astype(np.float64)
    x = torch.as_tensor((raw - center) / spread, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng, panels = np.random.default_rng(seed), list(panels)
    history = []
    for _ in range(epochs):
        perm, total, steps = torch.randperm(len(x), device=device), 0.0, 0
        for start in range(0, len(x), batch):
            xb = x[perm[start:start + batch]]
            mask = torch.zeros(len(xb), model.dim, device=device)
            for j in range(len(xb)):
                u = rng.random()
                if u < 0.10:
                    pass
                elif u < 0.20:
                    mask[j] = 1.0
                else:
                    mask[j, list(panels[int(rng.integers(len(panels)))])] = 1.0
            recon_mu, recon_logvar, q_mu, q_logvar, p_mu, p_logvar = model(xb, mask)
            missing = 1.0 - mask
            nll = 0.5 * (recon_logvar + (xb - recon_mu).pow(2) * torch.exp(-recon_logvar))
            recon_loss = (nll * missing).sum(-1) / missing.sum(-1).clamp_min(1.0)
            kl = model.kl_divergence(q_mu, q_logvar, p_mu, p_logvar)
            loss = recon_loss.mean() + beta * kl.mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step(); total += float(loss.detach()); steps += 1
        history.append(total / max(steps, 1))
    return model, history


class VAEACGenerator:
    """Frozen Q0 full/conditional sampler backed by canonical VAEAC."""
    def __init__(self, model, scale=None, alpha=1.0, device="cuda", feature_fn=None):
        self._legacy_reference = None
        if isinstance(model, np.ndarray):
            self._legacy_reference = np.asarray(model, dtype=float)
            self.model, self.device = None, "cpu"
            self.scale = np.ones(self._legacy_reference.shape[1]) if scale is None else np.asarray(scale, float)
            self.alpha = float(alpha)
            self.feature_fn = feature_map if feature_fn is None else feature_fn
            return
        self.model = model.eval().to(device)
        self.device, self.scale, self.alpha = device, np.asarray(scale if scale is not None else np.ones(model.dim)), float(alpha)
        self.feature_fn = (lambda x: feature_map(x, self.scale)) if feature_fn is None else feature_fn

    @property
    def dimension(self):
        return self.model.dim if self.model is not None else self._legacy_reference.shape[1]

    def _from_model_space(self, x):
        return x * np.asarray(self.model.data_std) + np.asarray(self.model.data_mean)

    def _prior_sample(self, obs, mask, n, seed):
        obs, mask = obs.expand(n, -1), mask.expand(n, -1)
        mu, logvar = self.model.prior(obs, mask)
        z = self.model.reparameterize(mu, logvar, seed)
        out_mu, out_logvar = self.model.decode_params(z, mask, obs)
        gen = torch.Generator(device=self.device).manual_seed(seed + 1)
        return out_mu + torch.exp(0.5 * out_logvar) * torch.randn(out_mu.shape, device=self.device, generator=gen)

    def sample_full(self, n, seed=0):
        if self._legacy_reference is not None:
            rng = np.random.default_rng(seed)
            return self._legacy_reference[rng.integers(len(self._legacy_reference), size=n)].copy()
        mask = torch.zeros(1, self.dimension, device=self.device)
        with torch.no_grad():
            return self._from_model_space(self._prior_sample(torch.zeros_like(mask), mask, n, seed).cpu().numpy())

    def sample_conditional(self, observed, panel, n, seed=0):
        if self._legacy_reference is not None:
            rng = np.random.default_rng(seed)
            out = self._legacy_reference[rng.integers(len(self._legacy_reference), size=n)].copy()
            out[:, list(panel)] = np.asarray(observed); return out
        center, spread = np.asarray(self.model.data_mean), np.asarray(self.model.data_std)
        obs = torch.zeros(1, self.dimension, device=self.device); mask = torch.zeros_like(obs)
        obs[:, list(panel)] = torch.as_tensor((np.asarray(observed) - center[list(panel)]) / spread[list(panel)], device=self.device)
        mask[:, list(panel)] = 1.0
        with torch.no_grad():
            out = self._from_model_space(self._prior_sample(obs, mask, n, seed).cpu().numpy())
        out[:, list(panel)] = np.asarray(observed); return out

    def sample_conditional_batch(self, observed_batch, panel, n, seed=0):
        observed = np.atleast_2d(np.asarray(observed_batch, dtype=float))
        return np.stack([self.sample_conditional(row, panel, n, seed + i) for i, row in enumerate(observed)])

    def tilted_full_sample(self, beta, n, seed=0):
        rng, accepted = np.random.default_rng(seed), []
        envelope = float(np.sum(np.abs(np.asarray(beta))))
        while len(accepted) < n:
            proposal = self.sample_full(max(256, 4 * (n - len(accepted))), int(rng.integers(2**31 - 1)))
            accepted.extend(proposal[rng.random(len(proposal)) < np.exp(self.feature_fn(proposal) @ beta - envelope)])
        return np.asarray(accepted[:n])

    def tilted_conditional_sample(self, beta, observed, panel, n, seed=0):
        rng, accepted = np.random.default_rng(seed), []
        envelope = float(np.sum(np.abs(np.asarray(beta))))
        while len(accepted) < n:
            proposal = self.sample_conditional(observed, panel, max(32, 2 * (n - len(accepted))), int(rng.integers(2**31 - 1)))
            accepted.extend(proposal[rng.random(len(proposal)) < np.exp(self.feature_fn(proposal) @ beta - envelope)])
        return np.asarray(accepted[:n])

    def tilted_sample(self, beta, feature_fn, n, seed=0):
        """Legacy diagnostic alias retained for the P0-P5 smoke tests."""
        old = self.feature_fn
        self.feature_fn = feature_fn
        try:
            samples = self.tilted_full_sample(beta, n, seed)
        finally:
            self.feature_fn = old
        logits = feature_fn(samples) @ np.asarray(beta)
        logits -= logits.max()
        weights = np.exp(logits)
        ess = float(weights.sum() ** 2 / np.sum(weights ** 2) / len(weights))
        return samples, 1.0, ess

    def importance_ess(self, beta, pool, pool_weights=None):
        logits = self.feature_fn(pool) @ np.asarray(beta)
        if pool_weights is not None: logits += np.log(np.asarray(pool_weights) + 1e-300)
        logits -= logits.max(); weights = np.exp(logits)
        return float(weights.sum() ** 2 / np.sum(weights ** 2) / len(weights))


def imp_conditional_mean_from_generator(generator, beta, batch, panel, n, seed, scale=None, feature_fn=None):
    rng = np.random.default_rng(seed)
    comp = generator.sample_conditional_batch(batch, panel, n, int(rng.integers(2**31 - 1)))
    fn = generator.feature_fn if feature_fn is None else feature_fn
    phi = fn(comp); logits = phi @ np.asarray(beta); logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return np.einsum("onr,on->or", phi, weights) / weights.sum(axis=1, keepdims=True)


def learned_information(generator, beta, panels, n_tilted=256, n_conditional=32, seed=0, feature_fn=None):
    rng = np.random.default_rng(seed)
    tilted = generator.tilted_full_sample(beta, n_tilted, int(rng.integers(2**31 - 1)))
    fn = generator.feature_fn if feature_fn is None else feature_fn; phi = fn(tilted); mu = phi.mean(0)
    fisher, infos = np.cov(phi, rowvar=False), []
    for panel in panels:
        observed = tilted[:, list(panel)]
        a = imp_conditional_mean_from_generator(generator, beta, observed, panel, n_conditional, seed + 1, feature_fn=feature_fn) - mu
        b = imp_conditional_mean_from_generator(generator, beta, observed, panel, n_conditional, seed + 2, feature_fn=feature_fn) - mu
        a -= a.mean(0); b -= b.mean(0); info = (a.T @ b + b.T @ a) / max(2 * (len(a) - 1), 1)
        vals, vecs = np.linalg.eigh((info + info.T) / 2); infos.append((vecs * np.maximum(vals, 1e-10)) @ vecs.T)
    return fisher, np.asarray(infos)
