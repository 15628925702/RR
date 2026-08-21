"""Discriminative Score OED baseline interface."""

from __future__ import annotations

import numpy as np


def masked_input(x: np.ndarray, panel: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x)
    mask = np.zeros(x.shape[-1], dtype=float)
    mask[list(panel)] = 1.0
    masked = x * mask
    return np.concatenate([masked, np.broadcast_to(mask, x.shape[:-1] + mask.shape)], axis=-1), mask


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

    def fit(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None,
            steps: int = 200, lr: float = 1e-2, ridge: float = 1e-4) -> MaskedScoreMLP:
        """Weighted-MSE gradient descent with linear learning-rate decay.

        ``weights`` are per-sample tilt weights, normalized to ``w/w.sum()*n``
        (all ones when ``None``). Returns ``self``.
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
