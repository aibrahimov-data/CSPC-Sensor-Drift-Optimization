"""Overfitting / underfitting report for KNN on the source validation data."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix, recall_score

from src.knn import FEATURES, TARGET, make_knn, select_k
from src.split import make_temporal_splits

GAS_NAMES = {1: "Ethanol", 2: "Ethylene"}


def plot_k_curve(table: pd.DataFrame, selected_k: int, output_path: Path):
    """Balanced accuracy of train and validation for every tested k."""
    curve = table.pivot(index="k", columns="split", values="balanced_accuracy")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(curve.index, curve["train"], marker="o", label="Train (batches 1-3)")
    ax.plot(
        curve.index, curve["validation"], marker="s",
        label="Validation (batches 4-5)",
    )
    ax.axvline(selected_k, color="gray", linestyle="--", label=f"Selected k = {selected_k}")
    ax.set_xlabel("Number of neighbors (k)")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("KNN: training vs validation performance")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def validation_report(parts: dict, k: int):
    """Confusion matrix and per-gas recall on validation for the chosen k."""
    train = parts["train"]
    validation = parts["validation"]

    model = make_knn(k)
    model.fit(train[FEATURES], train[TARGET])
    predictions = model.predict(validation[FEATURES])

    labels = [1, 2]
    matrix = confusion_matrix(validation[TARGET], predictions, labels=labels)
    recalls = recall_score(
        validation[TARGET], predictions, labels=labels, average=None
    )

    names = [GAS_NAMES[label] for label in labels]
    matrix_df = pd.DataFrame(
        matrix,
        index=[f"true_{name}" for name in names],
        columns=[f"pred_{name}" for name in names],
    )
    recall_df = pd.DataFrame({
        "gas": names,
        "n_validation": matrix.sum(axis=1),
        "recall": recalls,
    })
    return matrix_df, recall_df


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[1]

    config = yaml.safe_load(
        (project_root / "config.yaml").read_text(encoding="utf-8")
    )
    k_values = config["models"]["knn"]["k_values"]

    df = pd.read_csv(project_root / "data" / "processed" / "ethanol_ethylene.csv")
    parts = make_temporal_splits(df, config)

    table, best_k = select_k(parts, FEATURES, k_values)

    figures_dir = project_root / "results" / "figures"
    tables_dir = project_root / "results" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    plot_k_curve(table, best_k, figures_dir / "knn_k_curve.png")

    matrix_df, recall_df = validation_report(parts, best_k)
    recall_df.to_csv(tables_dir / "knn_validation_recall.csv", index=False)

    print(f"Selected k: {best_k}\n")
    print("Confusion matrix (validation):")
    print(matrix_df, "\n")
    print("Recall per gas (validation):")
    print(recall_df.to_string(index=False))
