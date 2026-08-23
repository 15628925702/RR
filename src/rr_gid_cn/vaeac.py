"""Arbitrary-conditioning VAE generator (P6), PyTorch CUDA implementation.

Follows Ivanov et al., "Variational Autoencoder with Arbitrary Conditioning",
ICLR 2019, as the frozen reference generator G0 of the RR-GID PDF. The encoder
consumes ``(x * mask, mask)`` and the decoder consumes ``(z, mask)``; missing
coordinates are imputed and observed coordinates are overwritten at sampling
time. Training minimizes the ELBO with the reconstruction term evaluated on
masked-out coordinates only, using masks sampled from the candidate panel
family (the 120 equal-cost coordinate pairs).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .synthetic_oracle import feature_map, inverse_warp, warp


def _gaussian_log_prob(z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * (logvar + (z - mu) ** 2 / logvar.exp() + np.log(2.0 * np.pi))


class VAEAC(nn.Module):
    """Mask-conditioned VAE over ``dim`` continuous coordinates."""

    def __init__(self, dim: int = 16, latent: int = 8, hidden: int = 64, seed: int = 0,
                 k_prior: int = 4, gmm_prior: bool = False):
        super().__init__()
        torch.manual_seed(seed)
        self.dim = int(dim)
        self.latent = int(latent)
        self.encoder = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2 * latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent + dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, dim),
        )
        self.gmm_prior = bool(gmm_prior)
        if gmm_prior:
            # Learned Gaussian-mixture prior matching the multi-modal Z.
            self.log_pi = nn.Parameter(torch.zeros(k_prior))
            self.prior_mu = nn.Parameter(torch.randn(k_prior, latent) * 0.2)
            self.prior_logvar = nn.Parameter(torch.full((k_prior, latent), -0.5))

    def log_prior(self, z: torch.Tensor) -> torch.Tensor:
        log_comp = torch.log_softmax(self.log_pi, dim=0) + _gaussian_log_prob(
            z[:, None, :], self.prior_mu[None, :, :], self.prior_logvar[None, :, :]).sum(-1)
        return torch.logsumexp(log_comp, dim=-1)

    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor, free_bits: float = 0.0) -> torch.Tensor:
        if not self.gmm_prior:
            kl = -0.5 * (1.0 + logvar - mu ** 2 - logvar.exp())
            if free_bits > 0:
                kl = torch.maximum(kl - free_bits, torch.zeros_like(kl))
            return kl.sum(-1)
        z = self.reparameterize(mu, logvar)
        log_q = _gaussian_log_prob(z, mu, logvar).sum(-1)
        return log_q - self.log_prior(z)

    def encode(self, x_masked: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.cat([x_masked, mask], dim=-1)
        p = self.encoder(h)
        mu, logvar = p.chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor, seed: int | None = None) -> torch.Tensor:
        if seed is not None:
            generator = torch.Generator(device=mu.device).manual_seed(seed)
            eps = torch.randn(mu.shape, device=mu.device, generator=generator)
        else:
            eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps

    def decode(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z, mask], dim=-1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_masked = x * mask
        mu, logvar = self.encode(x_masked, mask)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, mask)
        return recon, mu, logvar


def train_vaeac(model: VAEAC, reference: np.ndarray, panels, scale: np.ndarray | None = None,
                alpha: float = 1.0, epochs: int = 30, batch: int = 512, lr: float = 1e-3,
                device: str = "cuda", seed: int = 0, beta: float = 1.0,
                z_normalize: bool = True, free_bits: float = 0.0) -> tuple[VAEAC, list[float]]:
    """Train ``model`` on reference samples with random panel masks (ELBO).

    The VAE is trained in the **inverse-warped latent space** ``Z = T_alpha^-1(X)``
    (light-tailed Gaussian mixture), optionally standardized to unit per-coordinate
    scale, and samples are re-warped to X. ``beta < 1`` down-weights the KL term
    (beta-VAE) so the decoder keeps more information about the multi-modal Z.
    ``scale`` is kept only for the feature-map interface.
    """
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    z = inverse_warp(np.asarray(reference, dtype=float), alpha)
    if z_normalize:
        z_std = z.std(axis=0, ddof=1)
        z_std = np.where(z_std < 1e-6, 1.0, z_std)
        z = z / z_std
        model.z_std = z_std.astype(np.float64)
    else:
        model.z_std = np.ones(model.dim, dtype=np.float64)
    if model.gmm_prior:
        # Initialize the learned prior from the actual latent mixture before
        # ELBO optimization; a random prior collapses the four synthetic modes.
        from sklearn.cluster import KMeans
        k = model.prior_mu.shape[0]
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(z)
        labels = km.labels_
        means = np.stack([z[labels == j].mean(0) for j in range(k)])
        vars_ = np.stack([z[labels == j].var(0) + 1e-3 for j in range(k)])
        counts = np.bincount(labels, minlength=k).astype(float)
        with torch.no_grad():
            model.prior_mu.copy_(torch.as_tensor(means, dtype=torch.float32, device=device))
            model.prior_logvar.copy_(torch.log(torch.as_tensor(vars_, dtype=torch.float32, device=device)))
            model.log_pi.copy_(torch.log(torch.as_tensor(counts / counts.sum(), dtype=torch.float32, device=device)))
    x = torch.as_tensor(z, dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    n = len(x)
    dim = model.dim
    panels = list(panels)
    history: list[float] = []
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        count = 0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            xb = x[idx]
            mask = torch.zeros(len(idx), dim, device=device)
            for j in range(len(idx)):
                r = rng.random()
                if r < 0.3:
                    pass  # unconditional mask (all zeros)
                elif r < 0.6:
                    mask[j] = 1.0  # full-observation mask
                else:
                    panel = panels[int(rng.integers(len(panels)))]
                    mask[j, list(panel)] = 1.0
            recon, mu, logvar = model(xb, mask)
            recon_loss = ((recon - xb) ** 2 * (1.0 - mask)).sum(-1).mean()
            kl = model.kl_divergence(mu, logvar, free_bits=free_bits).mean()
            loss = recon_loss + beta * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            count += 1
        history.append(total / count)
    return model, history


def _mask_from_panel(panel, dim: int, device) -> torch.Tensor:
    mask = torch.zeros(dim, device=device)
    mask[list(panel)] = 1.0
    return mask


class VAEACGenerator:
    """Frozen VAEAC generator exposing the Q0 full/conditional sampling interface.

    The network operates in the inverse-warped (Z) space; samples are re-warped
    to X with ``T_alpha`` so the generator models the original Q0 over X.
    """

    def __init__(self, model: VAEAC | np.ndarray, scale: np.ndarray | None = None, alpha: float = 1.0, device: str = "cuda",
                 feature_fn=None):
        # Compatibility adapter for the pre-checkpoint smoke API. Formal P6
        # always passes a VAEAC model and explicit reference scale.
        self._legacy_reference = None
        if isinstance(model, np.ndarray):
            self._legacy_reference = np.asarray(model, dtype=float)
            self.model = None
            self.scale = np.ones(self._legacy_reference.shape[1], dtype=float) if scale is None else np.asarray(scale, dtype=float)
            self.alpha, self.device = float(alpha), "cpu"
            self.z_std = np.ones(self._legacy_reference.shape[1], dtype=float)
            self.feature_fn = (lambda x: feature_map(x, self.scale)) if feature_fn is None else feature_fn
            return
        self.model = model.eval().to(device)
        self.scale = np.asarray(scale, dtype=float)
        self.alpha = float(alpha)
        self.device = device
        self.z_std = np.asarray(getattr(model, "z_std", np.ones(model.dim)), dtype=float)
        # Relative-shift feature map used by the tilted samplers (synthetic by default).
        self.feature_fn = (lambda x: feature_map(x, self.scale)) if feature_fn is None else feature_fn

    def _to_x(self, z_out: np.ndarray) -> np.ndarray:
        return warp(z_out * self.z_std, self.alpha)

    @property
    def dimension(self) -> int:
        return self.model.dim if self.model is not None else self._legacy_reference.shape[1]

    def _sample_prior(self, n: int, seed: int = 0) -> torch.Tensor:
        model = self.model
        if model.gmm_prior:
            g = torch.Generator(self.device).manual_seed(seed)
            pi = torch.softmax(model.log_pi.detach(), 0)
            comp = torch.multinomial(pi, n, replacement=True, generator=g)
            mu = model.prior_mu.detach()[comp]
            sd = torch.exp(0.5 * model.prior_logvar.detach()[comp])
            return mu + sd * torch.randn(n, model.latent, device=self.device, generator=g)
        g = torch.Generator(self.device).manual_seed(seed)
        return torch.randn(n, model.latent, device=self.device, generator=g)

    def sample_full(self, n: int, seed: int = 0) -> np.ndarray:
        """Unconditional Q0 samples via z ~ N(0, I), decode with the zero mask."""
        if self._legacy_reference is not None:
            rng = np.random.default_rng(seed)
            return self._legacy_reference[rng.integers(len(self._legacy_reference), size=n)].copy()
        model = self.model
        z = self._sample_prior(n, seed)
        mask = torch.zeros(n, model.dim, device=self.device)
        with torch.no_grad():
            out_z = model.decode(z, mask)
        return self._to_x(out_z.cpu().numpy())

    def sample_conditional(self, observed: np.ndarray, panel: tuple[int, ...], n: int, seed: int = 0) -> np.ndarray:
        """Conditional Q0 samples given the observed panel coordinates (X space)."""
        if self._legacy_reference is not None:
            rng = np.random.default_rng(seed)
            ref = self._legacy_reference
            mask = np.all(np.isclose(ref[:, list(panel)], np.asarray(observed)[None, :]), axis=1)
            pool = ref[mask] if np.any(mask) else ref
            out = pool[rng.integers(len(pool), size=n)].copy()
            out[:, list(panel)] = observed
            return out
        observed = np.asarray(observed)
        model = self.model
        dim = model.dim
        generator = torch.Generator(self.device).manual_seed(seed)
        z_s = inverse_warp(observed, self.alpha)
        x_norm = np.zeros(dim, dtype=float)
        x_norm[list(panel)] = z_s / self.z_std[list(panel)]
        x_t = torch.as_tensor(x_norm, dtype=torch.float32, device=self.device).unsqueeze(0).expand(n, -1)
        mask = _mask_from_panel(panel, dim, self.device).unsqueeze(0).expand(n, -1)
        with torch.no_grad():
            mu, logvar = model.encode(x_t * mask, mask)
            z = model.reparameterize(mu, logvar, seed=seed)
            out_z = model.decode(z, mask)
        out_np = self._to_x(out_z.cpu().numpy())
        out_np[:, list(panel)] = observed
        return out_np

    def sample_conditional_batch(self, observed_batch: np.ndarray, panel: tuple[int, ...], n: int, seed: int = 0) -> np.ndarray:
        """Vectorized conditional sampling for a batch of observations (X space)."""
        observed = np.atleast_2d(np.asarray(observed_batch, dtype=float))
        rows = observed.shape[0]
        model = self.model
        dim = model.dim
        z_s = inverse_warp(observed, self.alpha)
        x_norm = np.zeros((rows, dim), dtype=float)
        x_norm[:, list(panel)] = z_s / self.z_std[list(panel)]
        x_t = torch.as_tensor(x_norm, dtype=torch.float32, device=self.device).unsqueeze(1).expand(-1, n, -1)
        mask = _mask_from_panel(panel, dim, self.device).view(1, 1, dim).expand(rows, n, -1)
        with torch.no_grad():
            mu, logvar = model.encode(x_t * mask, mask)
            generator = torch.Generator(self.device).manual_seed(seed)
            eps = torch.randn(rows, n, model.latent, device=self.device, generator=generator)
            z = mu + torch.exp(0.5 * logvar) * eps
            out_z = model.decode(z, mask)
        out_np = self._to_x(out_z.cpu().numpy())
        out_np[:, :, list(panel)] = observed[:, None, :]
        return out_np

    def tilted_full_sample(self, beta: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
        """Exact accept-reject samples from Q_beta relative to the generator's Q0."""
        rng = np.random.default_rng(seed)
        envelope = float(np.sum(np.abs(np.asarray(beta))))
        accepted = []
        while len(accepted) < n:
            proposal = self.sample_full(max(256, min(4096, 4 * (n - len(accepted)))), int(rng.integers(2**31 - 1)))
            logits = self.feature_fn(proposal) @ beta
            keep = rng.random(len(proposal)) < np.exp(logits - envelope)
            accepted.extend(proposal[keep])
        return np.asarray(accepted[:n])

    def tilted_sample(self, beta, feature_fn, n, seed=0):
        """Legacy diagnostic alias returning samples, acceptance, and ESS."""
        old = self.feature_fn
        self.feature_fn = feature_fn
        try:
            rng = np.random.default_rng(seed)
            draws = []
            proposals = 0
            envelope = float(np.sum(np.abs(np.asarray(beta))))
            while len(draws) < n:
                p = self.sample_full(max(32, 2 * (n-len(draws))), int(rng.integers(2**31-1)))
                proposals += len(p)
                keep = rng.random(len(p)) < np.exp(feature_fn(p) @ np.asarray(beta) - envelope)
                draws.extend(p[keep])
            out = np.asarray(draws[:n])
            acceptance = 1.0 if np.allclose(beta, 0.0) else float(n / max(proposals, 1))
            return out, acceptance, self.importance_ess(np.asarray(beta), out)
        finally:
            self.feature_fn = old

    def tilted_conditional_sample(self, beta: np.ndarray, observed: np.ndarray, panel: tuple[int, ...], n: int, seed: int = 0) -> np.ndarray:
        """Exact accept-reject samples from Q_beta(·|X_S = x_s) via the generator."""
        rng = np.random.default_rng(seed)
        beta = np.asarray(beta, dtype=float)
        panel_set = set(panel)
        fixed = np.zeros(beta.shape[0], dtype=bool)
        fixed[:6] = [i in panel_set for i in range(6)]
        fixed[6:] = [i in panel_set and i + 6 in panel_set for i in range(6)]
        observed_full = np.zeros(self.dimension)
        observed_full[list(panel)] = np.asarray(observed)
        observed_features = self.feature_fn(observed_full[None, :])[0]
        envelope = float(np.dot(beta[fixed], observed_features[fixed]) + np.sum(np.abs(beta[~fixed])))
        accepted = []
        while len(accepted) < n:
            proposal = self.sample_conditional(observed, panel, max(32, 2 * (n - len(accepted))), int(rng.integers(2**31 - 1)))
            logits = self.feature_fn(proposal) @ beta
            keep = rng.random(len(proposal)) < np.exp(logits - envelope)
            accepted.extend(proposal[keep])
        return np.asarray(accepted[:n])

    def importance_ess(self, beta: np.ndarray, pool: np.ndarray, pool_weights: np.ndarray | None = None) -> float:
        """Self-normalized importance ESS/N for a tilt beta on a generator pool."""
        logits = self.feature_fn(pool) @ np.asarray(beta)
        if pool_weights is not None:
            logits = logits + np.log(np.asarray(pool_weights) + 1e-300)
        logits = logits - logits.max()
        w = np.exp(logits)
        ess = float(w.sum() ** 2 / np.sum(w**2) / len(w))
        return ess


def learned_information(generator: VAEACGenerator, beta: np.ndarray, panels,
                        n_tilted: int = 256, n_conditional: int = 32, seed: int = 0,
                        feature_fn=None) -> tuple[np.ndarray, np.ndarray]:
    """Cross-completion panel information (PDF Eq. 9) using the learned generator.

    Returns ``(fisher_hat, infos)`` where ``infos`` are PSD-projected panel
    information matrices estimated with importance-weighted conditional
    completions (matching the S1 gate's ``panel_information_cross``).
    """
    rng = np.random.default_rng(seed)
    tilted = generator.tilted_full_sample(beta, n_tilted, int(rng.integers(2**31 - 1)))
    fn = generator.feature_fn if feature_fn is None else feature_fn
    phi = fn(tilted)
    mu = phi.mean(0)
    fisher_hat = np.cov(phi, rowvar=False)
    infos = []
    for panel in panels:
        observed = tilted[:, list(panel)]
        a = imp_conditional_mean_from_generator(generator, beta, observed, panel, n_conditional, seed + 1, feature_fn=feature_fn) - mu
        b = imp_conditional_mean_from_generator(generator, beta, observed, panel, n_conditional, seed + 2, feature_fn=feature_fn) - mu
        a = a - a.mean(0)
        b = b - b.mean(0)
        info_hat = (a.T @ b + b.T @ a) / max(2 * (len(a) - 1), 1)
        vals, vecs = np.linalg.eigh((info_hat + info_hat.T) / 2)
        infos.append((vecs * np.maximum(vals, 1e-10)) @ vecs.T)
    return fisher_hat, np.asarray(infos)


def imp_conditional_mean_from_generator(generator, beta, batch, panel, n, seed, scale=None, feature_fn=None):
    """E_{Q_beta}[phi(X)|X_S] via importance weighting on generator conditionals."""
    rng = np.random.default_rng(seed)
    batch = np.atleast_2d(np.asarray(batch, dtype=float))
    comp = generator.sample_conditional_batch(batch, panel, n, int(rng.integers(2**31 - 1)))
    fn = generator.feature_fn if feature_fn is None else feature_fn
    phi = fn(comp)
    w = np.exp(phi @ np.asarray(beta))
    return np.einsum("onr,on->or", phi, w) / w.sum(axis=1, keepdims=True)
