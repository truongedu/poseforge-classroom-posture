"""CPU-only tests for device-parity harness; real checkpoint/CUDA checked on laptop."""
import numpy as np
import pytest

from scripts.check_mlp_device_parity import make_features, load_features, compare


def test_synthetic_samples_follow_v2_feature_shape_and_repeatable():
    x = make_features(16, 302)
    assert x.shape == (16, 52)
    assert x.dtype == np.float32 and np.isfinite(x).all()
    assert np.array_equal(x, make_features(16, 302))
    assert set(np.unique(x.reshape(16, 13, 4)[:, :, 3])) <= {0., 1.}


def test_parity_detects_difference_and_top1_mismatch():
    class Fake:
        def __init__(self, probs): self.probs = probs
        def predict(self, features): return self.probs
    samples = make_features(2, 302)
    a = Fake(np.array([.8, .2], np.float32))
    same = Fake(np.array([.800001, .199999], np.float32))
    wrong = Fake(np.array([.2, .8], np.float32))
    err, mismatches, passed = compare(a, same, samples)
    assert passed and mismatches == 0 and err < 1e-4
    assert compare(a, wrong, samples)[1:] == (2, False)


def test_parity_load_features_checks_contract(tmp_path):
    p = tmp_path / 'features.npy'
    np.save(p, make_features(4, 1))
    assert load_features(p, 2).shape == (2, 52)
    np.save(p, np.zeros((4, 51), np.float32))
    with pytest.raises(ValueError, match='N x 52'):
        load_features(p, 2)
