from pathlib import Path   # Artificial data is used in this model-demo
import argparse

import numpy as np
import pandas as pd
import yaml

from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)

from src.split import make_temporal_splits


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config():
    config_path = PROJECT_ROOT / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_demo_dataset():
    """
    Create artificial data to test the pipeline.
    These are NOT real sensor measurements.
    """

    rng = np.random.default_rng(42)
    rows = []

    for batch in range(1, 11):
        for gas in [1, 2]:
            for _ in range(10):

                row = {
                    "batch_id": batch,
                    "gas_id": gas,
                }

                for i in range(1, 129):
                    row[f"f{i:03d}"] = rng.normal()

                rows.append(row)

    return pd.DataFrame(rows)


def train_dummy(train_df, validation_df):

    feature_columns = [
        f"f{i:03d}" for i in range(1, 129)
    ]

    required_columns = feature_columns + ["gas_id"]

    for name, dataset in [
        ("training", train_df),
        ("validation", validation_df),
    ]:
        missing = set(required_columns) - set(dataset.columns)

        if missing:
            raise ValueError(
                f"Missing columns in {name}: {sorted(missing)}"
            )

    X_train = train_df[feature_columns]
    y_train = train_df["gas_id"]

    X_val = validation_df[feature_columns]
    y_val = validation_df["gas_id"]

    model = DummyClassifier(
        strategy="most_frequent"
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    balanced_accuracy = balanced_accuracy_score(
        y_val, predictions
    )

    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro",
        labels=[1, 2],
        zero_division=0,
    )

    cm = confusion_matrix(
        y_val,
        predictions,
        labels=[1, 2],
    )

    print("\n===== DUMMY CLASSIFIER =====")

    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(validation_df)}")

    print(f"\nBalanced Accuracy: {balanced_accuracy:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    return model


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run using artificial demonstration data",
    )

    args = parser.parse_args()

    config = load_config()

    if args.demo:
        print("Using ARTIFICIAL demo data")
        df = create_demo_dataset()

    else:
        data_path = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "ethanol_ethylene.csv"
        )

        if not data_path.exists():
            raise FileNotFoundError(
                "Processed dataset not found. "
                "Ask Ismayil to complete src/parse.py "
                "or use --demo."
            )

        df = pd.read_csv(data_path)

    splits = make_temporal_splits(df, config)

    train_dummy(
        splits["train"],
        splits["validation"],
    )


if __name__ == "__main__":
    main()