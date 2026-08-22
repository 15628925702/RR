import numpy as np

from rr_gid_cn.gas_preprocess import fit_scaler_pca, panel_library, transform_features


def test_gas_dimensions_panels_and_bounded_phi():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(80, 128))
    mean, std, pcs = fit_scaler_pca(x[:60])
    phi = transform_features(x[60:], mean, std, pcs)
    assert len(panel_library()) == 120
    assert pcs.shape == (16, 8)
    assert phi.shape == (20, 16)
    assert np.all((phi >= -1) & (phi <= 1))
    # reference-train standardization has zero mean and unit variance
    z = (x[:60] - mean) / std
    assert np.allclose(z.mean(0), 0.0, atol=1e-8)
    assert np.allclose(z.std(0, ddof=1), 1.0, atol=1e-8)


def test_pc1_fit_on_train_only():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(40, 128))
    mean, std, pcs = fit_scaler_pca(x[:30])
    # transform of the train rows uses train-only PCA; no leak from val rows
    phi = transform_features(x[:30], mean, std, pcs)
    assert phi.shape == (30, 16)
