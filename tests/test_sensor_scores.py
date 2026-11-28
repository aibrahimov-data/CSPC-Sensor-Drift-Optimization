import pandas as pd

from src.sensor_scores import SENSOR_MODELS, build_sensor_scores


def make_rankings():
    sensors = list(range(1, 17))
    return pd.DataFrame({
        "sensor_id": sensors,
        "importance_score": 0.0,
        "importance_std": 0.0,
        "importance_rank": sensors,
        "stability_score": 1.0,
        "stability_rank": sensors,
        "hybrid_score": 1.0,
        "hybrid_rank": sensors,
    })


def make_selected():
    return pd.DataFrame([
        {"strategy": "importance", "budget": 4, "seed": None, "sensor_ids": "1;2;9;10"},
        {"strategy": "stability", "budget": 4, "seed": None, "sensor_ids": "4;5;6;10"},
        {"strategy": "random", "budget": 4, "seed": 11, "sensor_ids": "2;8;12;14"},
        {"strategy": "random", "budget": 4, "seed": 22, "sensor_ids": "4;6;10;11"},
        {"strategy": "importance", "budget": 8, "seed": None, "sensor_ids": "1;2;3;4;5;6;9;10"},
    ])


def test_each_sensor_model_has_four_sensors():
    assert sorted(SENSOR_MODELS) == list(range(1, 17))

    counts = pd.Series(SENSOR_MODELS).value_counts()
    assert set(counts.index) == {"TGS2600", "TGS2602", "TGS2610", "TGS2620"}
    assert (counts == 4).all()

    assert SENSOR_MODELS[1] == "TGS2602"
    assert SENSOR_MODELS[3] == "TGS2600"
    assert SENSOR_MODELS[5] == "TGS2610"
    assert SENSOR_MODELS[7] == "TGS2620"


def test_subset_membership_columns():
    scores = build_sensor_scores(make_rankings(), make_selected())

    assert len(scores) == 16

    sensor_10 = scores[scores["sensor_id"] == 10].iloc[0]
    assert sensor_10["selected_in_4"] == "importance;stability"
    assert sensor_10["random_seeds_4"] == "22"
    assert sensor_10["selected_in_8"] == "importance"

    sensor_2 = scores[scores["sensor_id"] == 2].iloc[0]
    assert sensor_2["selected_in_4"] == "importance"
    assert sensor_2["random_seeds_4"] == "11"

    sensor_16 = scores[scores["sensor_id"] == 16].iloc[0]
    assert sensor_16["selected_in_4"] == ""
    assert sensor_16["random_seeds_4"] == ""