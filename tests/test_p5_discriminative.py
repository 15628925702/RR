import numpy as np

from rr_gid_cn.discriminative import LinearScoreNetwork, masked_input, score_information


def test_mask_no_leakage_and_information_shape():
    x = np.arange(20, dtype=float).reshape(2, 10)
    encoded, mask = masked_input(x, (1, 3))
    assert np.all(encoded[:, :10][:, [0, 2, 4]] == 0)
    assert np.array_equal(mask[[1, 3]], [1, 1])
    model = LinearScoreNetwork(20, 4)
    model.fit(encoded, np.ones((2, 4)))
    info = score_information(model, x, ((1, 3),), np.ones(2))
    assert info.shape == (1, 4, 4)

