from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def make_temporal_splits(df: pd.DataFrame, config: dict) -> dict:
    """Separate early source batches from unseen future batches."""

    if "batch_id" not in df.columns:
        raise ValueError("Missing required column: batch_id")

    if not df.index.is_unique:
        raise ValueError("Dataset index must be unique")

    settings = config["splitting"]

    train_batches = set(settings["train_batches"])
    val_batches = set(settings["validation_batches"])
    source_batches = set(settings["source_batches"])
    future_batches = set(settings["test_batches"])

    # Prevent overlapping batch assignments.
    if train_batches & val_batches:
        raise ValueError("Training and validation batches overlap")

    if source_batches != train_batches | val_batches:
        raise ValueError("Source batches must equal train + validation")

    if source_batches & future_batches:
        raise ValueError("Source and future batches overlap")

    parts = {
        "train": df[df["batch_id"].isin(train_batches)].copy(),

        "validation": df[df["batch_id"].isin(val_batches)].copy(),

        "source_train": df[
            df["batch_id"].isin(source_batches)
        ].copy(),

        "future_test": df[
            df["batch_id"].isin(future_batches)
        ].copy(),
    }

    if any(part.empty for part in parts.values()):
        raise ValueError("At least one split is empty; check batch_id values")

    if len(parts["source_train"]) + len(parts["future_test"]) != len(df):
        raise ValueError("Some observations have an unconfigured batch")

    return parts


def make_recalibration_split(
    future_test: pd.DataFrame,
    target_batch: int,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create calibration candidates and a locked test set."""

    if not {"batch_id", "gas_id"}.issubset(future_test.columns):
        raise ValueError("Required columns: batch_id and gas_id")

    if target_batch not in config["recalibration"]["target_batches"]:
        raise ValueError("Batch is not configured for recalibration")

    target = future_test[
        future_test["batch_id"] == target_batch
    ].copy()

    counts = target["gas_id"].value_counts()

    if len(counts) != 2 or counts.min() < 2:
        raise ValueError("Need at least 2 observations from each gas")

    candidates, locked_test = train_test_split(
        target,
        test_size=config["recalibration"]["holdout_fraction"],
        random_state=config["recalibration"]["holdout_seed"],
        stratify=target["gas_id"],
    )

    if not candidates.index.is_unique or not locked_test.index.is_unique:
        raise ValueError("Dataset index must be unique")

    return candidates.copy(), locked_test.copy()


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[1]

    config = yaml.safe_load(
        (project_root / "config.yaml").read_text(encoding="utf-8")
    )

    if not isinstance(config, dict):
        raise ValueError("config.yaml is empty or invalid")

    # Small artificial dataset for testing.
    demo = pd.DataFrame([
        {"batch_id": batch, "gas_id": gas}
        for batch in range(1, 11)
        for gas in (1, 2)
        for _ in range(10)
    ])

    parts = make_temporal_splits(demo, config)

    for name, part in parts.items():
        print(f"{name}: {len(part)} rows")

    candidates, locked = make_recalibration_split(
        parts["future_test"], 7, config
    )

    print(f"batch 7 calibration candidates: {len(candidates)} rows")
    print(f"batch 7 locked holdout: {len(locked)} rows")