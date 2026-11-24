from src.dummy import (
    create_demo_dataset,
    train_dummy,
    load_config,
)

from src.split import make_temporal_splits


def test_temporal_splitting():

    df = create_demo_dataset()

    config = load_config()

    splits = make_temporal_splits(df, config)

    assert len(splits["train"]) == 60
    assert len(splits["validation"]) == 40
    assert len(splits["source_train"]) == 100
    assert len(splits["future_test"]) == 100

    assert set(splits["train"]["batch_id"]) == {1, 2, 3}

    assert set(splits["validation"]["batch_id"]) == {4, 5}

    assert set(splits["future_test"]["batch_id"]) == {
        6, 7, 8, 9, 10
    }
    
    
def test_dummy_classifier():

    df = create_demo_dataset()

    config = load_config()

    splits = make_temporal_splits(df, config)

    model = train_dummy(
        splits["train"],
        splits["validation"],
    )

    feature_columns = [
        f"f{i:03d}" for i in range(1, 129)
    ]

    predictions = model.predict(
        splits["validation"][feature_columns]
    )

    assert len(predictions) == 40

    assert set(predictions) == {1}
    
    
import pytest


def test_missing_feature():

    df = create_demo_dataset()

    config = load_config()

    splits = make_temporal_splits(df, config)

    validation = splits["validation"].drop(
        columns=["f128"]
    )

    with pytest.raises(ValueError):

        train_dummy(
            splits["train"],
            validation,
        )