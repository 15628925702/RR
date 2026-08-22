"""Discriminative Score OED baseline interface."""

from __future__ import annotations

import os

import numpy as np


def masked_input(x: np.ndarray, panel: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x)
    mask = np.zeros(x.shape[-1], dtype=float)
    mask[list(panel)] = 1.0
    masked = x * mask
    return np.concatenate([masked, np.broadcast_to(mask, x.shape[:-1] + mask.shape)], axis=-1), mask


def masked_pool(x: np.ndarray, mask_matrix: np.ndarray) -> np.ndarray:
    """Batch masked encodings ``[x * m_i, m_i]`` for a per-row mask matrix.

    ``x`` has shape ``(n, d)`` and ``mask_matrix`` has shape ``(n, d)``. Returns
    the ``(n, 2d)`` encoding used as MLP input (the PDF's uniform random mask
    draws over the candidate panel family).
    """
    return np.concatenate([x * mask_matrix, mask_matrix], axis=-1)


def weighted_mse(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    """Weighted MSE averaged over samples and output dims.

    ``weights`` are per-sample tilt weights; they are normalized to sum to n so the
    reported value stays on the same scale as a plain unweighted MSE.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n = weights.shape[0]
    w = weights / weights.sum() * n
    return float(np.mean(w[:, None] * (y_pred - y_true) ** 2))


class LinearScoreNetwork:
    """Small deterministic CPU baseline implementing masked score prediction."""

    def __init__(self, input_dim: int, output_dim: int):
        self.weights = np.zeros((input_dim, output_dim))

    def fit(self, x: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> None:
        gram = x.T @ x + ridge * np.eye(x.shape[1])
        self.weights = np.linalg.solve(gram, x.T @ y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x) @ self.weights


def _xavier(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    bound = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-bound, bound, size=(fan_in, fan_out))


class MaskedScoreMLP:
    """Mask-conditioned MLP for discriminative score OED.

    Two hidden tanh layers implemented in pure numpy (manual forward/backward).
    Inputs are the masked encodings produced by :func:`masked_input`
    (``[x*mask, mask]``); outputs are the r-dimensional centered score s_beta(X).
    Fit minimizes a per-sample tilt-weighted MSE with an L2 ridge penalty.

    ``predict``/``fit`` mirror :class:`LinearScoreNetwork`, so ``score_information``
    can consume either model type unchanged.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden: int = 64, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden = int(hidden)
        # Xavier-style (Glorot uniform) initialization.
        self.W1 = _xavier(rng, input_dim, hidden)
        self.b1 = np.zeros(hidden)
        self.W2 = _xavier(rng, hidden, hidden)
        self.b2 = np.zeros(hidden)
        self.W3 = _xavier(rng, hidden, output_dim)
        self.b3 = np.zeros(output_dim)
        self.loss_history: list[float] = []

    def predict(self, x: np.ndarray) -> np.ndarray:
        h1 = np.tanh(np.asarray(x) @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        return h2 @ self.W3 + self.b3

    def _fit_torch(self, x: np.ndarray, y: np.ndarray, w: np.ndarray,
                   steps: int, lr: float, ridge: float) -> None:
        """GPU (float64) weighted-MSE training matching the numpy path's math.

        Mirrors :meth:`fit` step-for-step (same loss, same per-step L2 ridge
        gradient ``2*ridge*W``, same linear learning-rate decay) so switching to
        the torch backend is numerically transparent on any CUDA machine.
        """
        import torch
        n = len(x)
        dev = "cuda"
        # Keep this worker's GPU footprint to half the device by default so the
        # host can co-locate another project on the same GPU (open config knob).
        fraction = float(os.environ.get("RR_GID_GPU_FRACTION", "0.5"))
        try:
            torch.cuda.set_per_process_memory_fraction(fraction, device=dev)
        except (RuntimeError, ValueError):
            pass
        Xt = torch.as_tensor(x, dtype=torch.float64, device=dev)
        Yt = torch.as_tensor(y, dtype=torch.float64, device=dev)
        wt = torch.as_tensor(w, dtype=torch.float64, device=dev)
        W1 = torch.tensor(self.W1, dtype=torch.float64, device=dev, requires_grad=True)
        b1 = torch.tensor(self.b1, dtype=torch.float64, device=dev, requires_grad=True)
        W2 = torch.tensor(self.W2, dtype=torch.float64, device=dev, requires_grad=True)
        b2 = torch.tensor(self.b2, dtype=torch.float64, device=dev, requires_grad=True)
        W3 = torch.tensor(self.W3, dtype=torch.float64, device=dev, requires_grad=True)
        b3 = torch.tensor(self.b3, dtype=torch.float64, device=dev, requires_grad=True)
        ridge_t = float(ridge)
        self.loss_history = []
        for step in range(int(steps)):
            h1 = torch.tanh(Xt @ W1 + b1)
            h2 = torch.tanh(h1 @ W2 + b2)
            out = h2 @ W3 + b3
            mse = torch.mean(wt[:, None] * (out - Yt) ** 2)
            self.loss_history.append(float(mse.item()))
            loss = mse + ridge_t * (W1.pow(2).sum() + W2.pow(2).sum() + W3.pow(2).sum())
            loss.backward()
            step_lr = lr * (1.0 - step / steps)
            with torch.no_grad():
                W1 -= step_lr * W1.grad
                b1 -= step_lr * b1.grad
                W2 -= step_lr * W2.grad
                b2 -= step_lr * b2.grad
                W3 -= step_lr * W3.grad
                b3 -= step_lr * b3.grad
                for p in (W1, b1, W2, b2, W3, b3):
                    p.grad = None
        self.W1 = W1.detach().cpu().numpy()
        self.b1 = b1.detach().cpu().numpy()
        self.W2 = W2.detach().cpu().numpy()
        self.b2 = b2.detach().cpu().numpy()
        self.W3 = W3.detach().cpu().numpy()
        self.b3 = b3.detach().cpu().numpy()

    def fit(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None,
            steps: int = 200, lr: float = 1e-2, ridge: float = 1e-4) -> MaskedScoreMLP:
        """Weighted-MSE gradient descent with linear learning-rate decay.

        ``weights`` are per-sample tilt weights, normalized to ``w/w.sum()*n``
        (all ones when ``None``). On a CUDA machine the heavy loop runs in
        float64 on the GPU (numerically identical math); otherwise it falls back
        to the pure-numpy path. Returns ``self``.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = x.shape[0]
        if weights is None:
            w = np.ones(n)
        else:
            w = np.asarray(weights, dtype=float)
            w_sum = w.sum()
            w = np.ones(n) if w_sum == 0 else w / w_sum * n
        try:
            import torch
            if torch.cuda.is_available() and n >= 500:
                self._fit_torch(x, y, w, int(steps), float(lr), float(ridge))
                return self
        except (ImportError, RuntimeError, ValueError):
            pass
        self.loss_history = []
        for step in range(int(steps)):
            h1 = np.tanh(x @ self.W1 + self.b1)
            h2 = np.tanh(h1 @ self.W2 + self.b2)
            out = h2 @ self.W3 + self.b3
            self.loss_history.append(weighted_mse(y, out, w))
            # dL/d out = 2 * w_i * r_ij / n  (L is mean over samples & output dims).
            d_out = 2.0 * (out - y) * w[:, None] / n
            d_W3 = h2.T @ d_out + 2.0 * ridge * self.W3
            d_b3 = d_out.sum(axis=0)
            d_a2 = (d_out @ self.W3.T) * (1.0 - h2 ** 2)
            d_W2 = h1.T @ d_a2 + 2.0 * ridge * self.W2
            d_b2 = d_a2.sum(axis=0)
            d_a1 = (d_a2 @ self.W2.T) * (1.0 - h1 ** 2)
            d_W1 = x.T @ d_a1 + 2.0 * ridge * self.W1
            d_b1 = d_a1.sum(axis=0)
            step_lr = lr * (1.0 - step / steps)
            self.W1 -= step_lr * d_W1
            self.b1 -= step_lr * d_b1
            self.W2 -= step_lr * d_W2
            self.b2 -= step_lr * d_b2
            self.W3 -= step_lr * d_W3
            self.b3 -= step_lr * d_b3
        return self


def score_information(model: LinearScoreNetwork | MaskedScoreMLP, validation: np.ndarray,
                      panels: tuple[tuple[int, ...], ...], weights: np.ndarray) -> np.ndarray:
    infos = []
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    for panel in panels:
        inputs, _ = masked_input(validation, panel)
        scores = model.predict(inputs)
        mean = weights @ scores
        centered = scores - mean
        infos.append((centered * weights[:, None]).T @ centered)
    return np.asarray(infos)
