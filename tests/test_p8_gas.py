import numpy as np

from rr_gid_cn.gas_preprocess import fit_scaler_pca, panel_library, transform_features


def test_gas_dimensions_panels_and_bounded_phi():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(80, 128))
    mean, pcs = fit_scaler_pca(x[:60])
    phi = transform_features(x[60:], mean, x[:60].std(0, ddof=1), pcs)
    assert len(panel_library()) == 120
    assert pcs.shape == (16, 8)
    assert phi.shape == (20, 16)
    assert np.all((phi >= -1) & (phi <= 1))

