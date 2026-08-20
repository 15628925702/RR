import numpy as np
import pytest

from rr_gid_cn.rrgid import Theta, balanced_counts, cross_completion_information, psd_project


def test_balanced_counts_and_projection():
    counts = balanced_counts(60, np.full(3, 1 / 3), np.ones(3, dtype=int))
    assert counts.sum() <= 60
    theta = Theta(np.full(2, -1.0), np.full(2, 1.0))
    assert np.all(theta.project(np.array([-2.0, 2.0])) == [-1.0, 1.0])


def test_psd_projection():
    projected = psd_project(np.array([[1.0, 2.0], [2.0, 1.0]]))
    assert np.linalg.eigvalsh(projected).min() >= 1e-10 - 1e-12


def test_cross_completion_is_symmetric_psd():
    from rr_gid_cn.synthetic_oracle import make_frozen_mixture, reference_scale, sample_full
    mix = make_frozen_mixture()
    scale = reference_scale(mix, 500, 7)
    tilted = sample_full(mix, 20, 8)
    info = cross_completion_information(mix, np.zeros(12), (0, 1), tilted, scale, 4, 9)
    assert np.allclose(info, info.T, atol=1e-10)
    assert np.linalg.eigvalsh(info).min() >= -1e-10

