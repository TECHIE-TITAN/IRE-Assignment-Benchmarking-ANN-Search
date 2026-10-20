import numpy as np
import pytest

from synthetic_data import DriftConfig, make_drift_dataset

# small config so the tests run fast
CFG = DriftConfig(d=16, n_clusters=8, n_base=500, n_new=300, n_queries=200, seed=7)


def test_same_seed_gives_identical_data():
    a = make_drift_dataset(CFG, 2.0)
    b = make_drift_dataset(CFG, 2.0)
    for key in ("base", "new", "queries", "shift"):
        assert np.array_equal(a[key], b[key])


def test_different_seed_gives_different_data():
    other = DriftConfig(d=16, n_clusters=8, n_base=500, n_new=300, n_queries=200, seed=8)
    assert not np.array_equal(make_drift_dataset(CFG, 0.0)["base"],
                              make_drift_dataset(other, 0.0)["base"])


def test_shapes_and_dtype():
    data = make_drift_dataset(CFG, 1.0)
    assert data["base"].shape == (500, 16)
    assert data["new"].shape == (300, 16)
    assert data["queries"].shape == (200, 16)
    assert all(data[k].dtype == np.float32 for k in ("base", "new", "queries"))


def test_base_does_not_depend_on_shift():
    assert np.array_equal(make_drift_dataset(CFG, 0.0)["base"],
                          make_drift_dataset(CFG, 4.0)["base"])


def test_shift_only_translates_new_vectors_and_queries():
    control = make_drift_dataset(CFG, 0.0)
    drifted = make_drift_dataset(CFG, 3.0)
    assert np.allclose(drifted["new"] - control["new"], drifted["shift"], atol=1e-4)
    assert np.allclose(drifted["queries"] - control["queries"], drifted["shift"], atol=1e-4)


def test_shift_size_is_in_cluster_radii():
    data = make_drift_dataset(CFG, 2.0)
    assert np.linalg.norm(data["shift"]) == pytest.approx(2.0 * CFG.radius)


def test_shift_direction_is_the_same_for_every_size():
    one = make_drift_dataset(CFG, 1.0)["shift"]
    three = make_drift_dataset(CFG, 3.0)["shift"]
    assert np.allclose(three, 3.0 * one)


def test_zero_shift_is_the_control():
    assert np.count_nonzero(make_drift_dataset(CFG, 0.0)["shift"]) == 0


def test_vector_limit_is_enforced():
    too_big = DriftConfig(n_base=400_000, n_new=200_000)
    with pytest.raises(ValueError):
        make_drift_dataset(too_big, 0.0)
