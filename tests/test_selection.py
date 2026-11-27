import numpy as np
import pandas as pd

from src.knn import FEATURES, TARGET
from src.selection import (
    build_selection_table,
    combine_rankings,
    importance_scores,
    select_sensors,
)
from src.sensors import sensor_features


def make_signal_part(n_rows, seed, batch_id):
    """Random data where only the features of sensor 5 separate the gases."""
    rng = np.random.default_rng(seed)
    part = pd.DataFrame(rng.normal(size=(n_rows, 128)), columns=FEATURES)
    part[TARGET] = np.tile([1, 2], n_rows // 2)
    part.loc[part[TARGET] == 2, sensor_features([5])] += 5.0
    part["batch_id"] = batch_id
    part["concentration_ppmv"] = 100.0
    return part


def make_rankings():
    """Sensor 16 is the most important, sensor 1 the most stable."""
    sensors = np.arange(1, 17)
    importance = pd.DataFrame({
        "sensor_id": sensors,
        "importance_score": sensors / 100,
        "importance_std": 0.0,
        "importance_rank": 17 - sensors,
    })
    stability = pd.DataFrame({
        "sensor_id": sensors,
        "stability_score": sensors / 10,
        "stability_rank": sensors,
    })
    return combine_rankings(importance, stability)


def test_importance_finds_the_informative_sensor():
    parts = {
        "train": make_signal_part(200, seed=1, batch_id=1),
        "validation": make_signal_part(100, seed=2, batch_id=4),
    }

    result = importance_scores(parts, k=5, n_repeats=5, seed=0)

    assert len(result) == 16
    assert result.loc[result["importance_rank"] == 1, "sensor_id"].iloc[0] == 5


def test_importance_ignores_future_batches():
    train = make_signal_part(200, seed=1, batch_id=1)
    validation = make_signal_part(100, seed=2, batch_id=4)

    parts_a = {
        "train": train,
        "validation": validation,
        "future_test": make_signal_part(100, seed=3, batch_id=7),
    }
    parts_b = {
        "train": train,
        "validation": validation,
        "future_test": make_signal_part(100, seed=4, batch_id=9),
    }

    result_a = importance_scores(parts_a, k=5, n_repeats=3, seed=0)
    result_b = importance_scores(parts_b, k=5, n_repeats=3, seed=0)

    pd.testing.assert_frame_equal(result_a, result_b)


def test_select_sensors_follow_the_rankings():
    rankings = make_rankings()

    assert select_sensors("all", 16, rankings) == list(range(1, 17))
    assert select_sensors("importance", 4, rankings) == [13, 14, 15, 16]
    assert select_sensors("stability", 4, rankings) == [1, 2, 3, 4]
    # Hybrid ties go to the more important sensor.
    assert select_sensors("hybrid", 4, rankings) == [13, 14, 15, 16]


def test_random_selection_is_reproducible():
    rankings = make_rankings()

    first = select_sensors("random", 8, rankings, seed=11)
    second = select_sensors("random", 8, rankings, seed=11)

    assert first == second
    assert len(first) == 8
    assert len(set(first)) == 8
    assert all(1 <= sensor <= 16 for sensor in first)


def test_selection_table_layout():
    config = {
        "sensor_selection": {
            "budgets": [4, 8, 16],
            "strategies": ["all", "random", "importance", "stability", "hybrid"],
        },
        "recalibration": {"sampling_seeds": [11, 22]},
    }

    table = build_selection_table(make_rankings(), config)

    # all: 1 row; random: 2 budgets x 2 seeds; three other strategies: 2 rows each.
    assert len(table) == 1 + 4 + 6
    assert set(table.loc[table["budget"] == 16, "strategy"]) == {"all"}
    assert (table["n_features"] == 8 * table["budget"]).all()
    assert (table["features"].str.split(";").str.len() == table["n_features"]).all()