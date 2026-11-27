"""K-Nearest Neighbors classifier with k selection on source validation data."""

from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.split import make_temporal_splits

FEATURES = [f"f{i:03d}" for i in range(1, 129)]
TARGET = "gas_id"


def make_knn(k: int):
    """Scaler + KNN. The scaler is fitted only on the data passed to fit()."""
    return make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=k),
    )


def select_k(parts: dict, feature_cols: list, k_values: list):
    """Fit KNN on train (batches 1-3) for each k and score train + validation.

    Returns the full results table and the selected k. Future batches are
    never touched here.
    """
    train = parts["train"]
    validation = parts["validation"]

    rows = []
    for k in k_values:
        model = make_knn(k)
        model.fit(train[feature_cols], train[TARGET])

        for split_name, part in (("train", train), ("validation", validation)):
            predictions = model.predict(part[feature_cols])
            rows.append({
                "k": k,
                "split": split_name,
                "balanced_accuracy": balanced_accuracy_score(
                    part[TARGET], predictions
                ),
                "macro_f1": f1_score(
                    part[TARGET], predictions, average="macro"
                ),
            })

    table = pd.DataFrame(rows)

    # Placeholder selection rule: best validation balanced accuracy,
    # smallest k on ties. Update once the rule is agreed and frozen.
    validation_rows = table[table["split"] == "validation"]
    best = validation_rows.sort_values(
        ["balanced_accuracy", "k"], ascending=[False, True]
    ).iloc[0]

    return table, int(best["k"])


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[1]

    config = yaml.safe_load(
        (project_root / "config.yaml").read_text(encoding="utf-8")
    )
    k_values = config["models"]["knn"]["k_values"]

    df = pd.read_csv(project_root / "data" / "processed" / "ethanol_ethylene.csv")
    parts = make_temporal_splits(df, config)

    table, best_k = select_k(parts, FEATURES, k_values)

    output_dir = project_root / "results" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "knn_validation.csv", index=False)

    print(table.to_string(index=False))
    print(f"\nSelected k: {best_k}")