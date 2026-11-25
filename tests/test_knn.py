import numpy as np
import pandas as pd

from src.knn import FEATURES, TARGET, make_knn, select_k
from src.sensors import sensor_features


def make_demo_parts():
    """Small random dataset with the same columns as the real one."""
    rng = np.random.default_rng(0)

    def make_part(n_rows):
        part = pd.DataFrame(
            rng.normal(size=(n_rows, 128)), columns=FEATURES
        )
        part[TARGET] = np.tile([1, 2], n_rows // 2)
        return part

    return {"train": make_part(60), "validation": make_part(30)}


def test_sensor_feature_counts():
    assert len(sensor_features([1, 2, 3, 4])) == 32
    assert len(sensor_features(range(1, 9))) == 64
    assert len(sensor_features(range(1, 17))) == 128


def test_all_sensors_cover_all_features():
    assert sensor_features(range(1, 17)) == FEATURES


def test_select_k_returns_table_and_valid_k():
    parts = make_demo_parts()
    k_values = [1, 3, 5]

    table, best_k = select_k(parts, FEATURES, k_values)

    assert best_k in k_values
    assert len(table) == len(k_values) * 2
    assert set(table["split"]) == {"train", "validation"}
    assert {"k", "balanced_accuracy", "macro_f1"}.issubset(table.columns)


def test_scaler_is_fitted_on_train_only():
    parts = make_demo_parts()
    train = parts["train"]

    model = make_knn(3)
    model.fit(train[FEATURES], train[TARGET])

    scaler = model.named_steps["standardscaler"]
    assert np.allclose(scaler.mean_, train[FEATURES].mean().to_numpy())

    # The validation part must not influence the scaler.
    validation = parts["validation"]
    assert not np.allclose(scaler.mean_, validation[FEATURES].mean().to_numpy())