import numpy as np
import pandas as pd

from src.drift import compute_shifts, sensor_of_feature, stability_scores
from src.knn import FEATURES, TARGET
from src.sensors import sensor_features


def make_part(n_rows, seed, batch_id):
    rng = np.random.default_rng(seed)
    part = pd.DataFrame(rng.normal(size=(n_rows, 128)), columns=FEATURES)
    part[TARGET] = np.tile([1, 2], n_rows // 2)
    part["batch_id"] = batch_id
    part["concentration_ppmv"] = 100.0
    return part


def test_sensor_of_feature_matches_sensor_features():
    assert sensor_of_feature("f001") == 1
    assert sensor_of_feature("f009") == 2
    assert sensor_of_feature("f128") == 16

    for sensor in range(1, 17):
        assert all(
            sensor_of_feature(name) == sensor for name in sensor_features([sensor])
        )


def test_identical_data_has_zero_shift():
    reference = make_part(200, seed=0, batch_id=1)

    result = compute_shifts(reference, reference)

    assert np.allclose(result["mean_abs_shift"], 0.0, atol=1e-9)


def test_known_offset_gives_known_shift():
    reference = make_part(200, seed=0, batch_id=1)

    shifted = reference.copy()
    reference_std = reference.groupby(TARGET)[FEATURES].transform("std")
    shifted[FEATURES] = reference[FEATURES] + 2 * reference_std
    shifted["batch_id"] = 5

    result = compute_shifts(shifted, reference)

    assert np.allclose(result["mean_abs_shift"], 2.0)


def test_stability_scores_ignore_future_batches():
    train = make_part(200, seed=1, batch_id=1)
    validation = make_part(100, seed=2, batch_id=4)

    parts_a = {
        "train": train,
        "validation": validation,
        "future_test": make_part(100, seed=3, batch_id=7),
    }
    parts_b = {
        "train": train,
        "validation": validation,
        "future_test": make_part(100, seed=4, batch_id=9) + 50.0,
    }

    result_a = stability_scores(parts_a, (10, 300))
    result_b = stability_scores(parts_b, (10, 300))

    pd.testing.assert_frame_equal(result_a, result_b)
    assert len(result_a) == 16
    assert sorted(result_a["stability_rank"]) == list(range(1, 17))