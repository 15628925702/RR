import numpy as np
import pytest

from rr_gid_cn.discriminative import (
    LinearScoreNetwork,
    MaskedScoreMLP,
    masked_input,
    score_information,
    weighted_mse,
)


def test_mask_no_leakage_and_information_shape():
    x = np.arange(20, dtype=float).reshape(2, 10)
    encoded, mask = masked_input(x, (1, 3))
    assert np.all(encoded[:, :10][:, [0, 2, 4]] == 0)
    assert np.array_equal(mask[[1, 3]], [1, 1])
    model = LinearScoreNetwork(20, 4)
    model.fit(encoded, np.ones((2, 4)))
    info = score_information(model, x, ((1, 3),), np.ones(2))
    assert info.shape == (1, 4, 4)


def test_mask_encoding_positions_and_no_leakage():
    x = np.arange(1.0, 9.0).reshape(1, 8)
    encoded, mask = masked_input(x, (2, 5))
    assert mask.shape == (8,)
    # binary mask has 1 exactly on panel coordinates, 0 elsewhere
    assert np.array_equal(mask, np.array([0, 0, 1, 0, 0, 1, 0, 0]))
    obs = encoded[:, :8]
    # observed coordinates preserved verbatim
    assert np.array_equal(obs[0, [2, 5]], x[0, [2, 5]])
    # unobserved coordinates zeroed (no leakage)
    assert np.all(obs[0, [0, 1, 3, 4, 6, 7]] == 0)
    # second half of the encoding carries the broadcast mask
    assert np.array_equal(encoded[:, 8:], np.broadcast_to(mask, encoded.shape[:1] + mask.shape))


def test_mlp_fits_nonlinear_target():
    rng = np.random.default_rng(11)
    d, out, n_train, n_val = 6, 3, 400, 200
    x = rng.normal(size=(n_train + n_val, d))
    v = rng.normal(size=(d, out)) / np.sqrt(d)
    y = np.tanh(x @ v)  # nonlinear target a linear model cannot fit well
    xtr, ytr = x[:n_train], y[:n_train]
    xval, yval = x[n_train:], y[n_train:]
    model = MaskedScoreMLP(d, out, hidden=32, seed=0)
    err_before = float(np.mean((model.predict(xval) - yval) ** 2))
    model.fit(xtr, ytr, steps=400, lr=0.05, ridge=1e-4)
    err_after = float(np.mean((model.predict(xval) - yval) ** 2))
    # training must cut the validation MSE substantially
    assert err_after < 0.25 * err_before
    # and the tracked weighted loss must strictly decrease
    assert model.loss_history[-1] < model.loss_history[0]


def test_mlp_fit_returns_self_and_accepts_weights():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(30, 4))
    y = rng.normal(size=(30, 2))
    model = MaskedScoreMLP(4, 2, hidden=8, seed=0)
    ret = model.fit(x, y, weights=np.exp(rng.normal(size=30)), steps=10, lr=1e-2)
    assert ret is model
    assert len(model.loss_history) == 10
    assert model.predict(x).shape == (30, 2)


def test_score_information_mlp_symmetry_psd():
    rng = np.random.default_rng(7)
    d, out, n = 8, 5, 60
    val = rng.normal(size=(n, d))
    panels = ((0, 1), (2, 3), (4, 5))
    model = MaskedScoreMLP(2 * d, out, hidden=16, seed=1)
    # cheap warm-up so predict is exercised on real masked inputs
    enc, _ = masked_input(val, panels[0])
    model.fit(enc, np.zeros((n, out)), steps=5, lr=1e-3, ridge=1e-4)
    weights = np.exp(rng.normal(size=n))  # positive tilt weights
    infos = score_information(model, val, panels, weights)
    assert infos.shape == (len(panels), out, out)
    for info in infos:
        assert np.allclose(info, info.T, atol=1e-10)  # symmetric
        assert np.linalg.eigvalsh(info).min() >= -1e-8  # approximately PSD


def test_weighted_mse_normalization_and_dominance():
    y_true = np.array([[0.0], [1.0], [1.0]])
    y_pred = np.array([[1.0], [1.0], [1.0]])
    uniform = weighted_mse(y_true, y_pred, np.ones(3))
    w = np.array([3.0, 1.0, 1.0])  # big weight on the big-error sample
    weighted = weighted_mse(y_true, y_pred, w)
    # weights normalize to sum n -> [1.8, 0.6, 0.6]; only sample 0 has error 1
    assert np.isclose(weighted, (1.8 * 1.0 + 0.6 * 0.0 + 0.6 * 0.0) / 3.0)
    # high weight on the large-error sample inflates the loss vs uniform
    assert weighted > uniform
    # scale invariance: c * weights yields the same normalized loss
    assert np.isclose(weighted_mse(y_true, y_pred, 10.0 * w), weighted)
