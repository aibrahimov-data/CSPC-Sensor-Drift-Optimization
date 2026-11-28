"""Concentration sensitivity analysis.

Checks whether the classification results depend on differences between the
concentration distributions of ethanol and ethylene.

The final source model (batches 1-5) is evaluated separately on every future
batch (6-10) under three conditions:

* all          - all observations.
* common_range - fitting and test data restricted to 10-300 ppmv.
* matched      - fitting data restricted to 10-300 ppmv; the test data of each
                 batch are subsampled so that ethanol and ethylene have equal
                 counts in every concentration bin.

Two models are compared:

* sensors_only        - the main KNN (selected k, all 16 sensors).
* concentration_only  - a diagnostic Decision Tree that sees concentration
                        only. It is a check for dataset confounding, not a gas
                        sensor model.

Nothing is tuned or selected using these results.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier

from src.drift import in_range
from src.knn import FEATURES, TARGET, make_knn, select_k
from src.split import make_temporal_splits

CONCENTRATION = ["concentration_ppmv"]
BIN_EDGES = [10, 50, 100, 200, 300]
CONDITIONS = ["all", "common_range", "matched"]


def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Add the concentration bin: [10, 50], (50, 100], (100, 200], (200, 300]."""
    binned = df.copy()
    binned["conc_bin"] = pd.cut(
        binned["concentration_ppmv"], bins=BIN_EDGES, include_lowest=True
    )
    return binned


def concentration_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Number of observations per batch, gas and concentration bin."""
    binned = add_bins(df)
    binned["conc_bin"] = binned["conc_bin"].astype(str).replace("nan", "outside 10-300")

    return (
        binned.groupby(["batch_id", TARGET, "conc_bin"])
        .size()
        .rename("n_observations")
        .reset_index()
    )


def match_concentration(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Equal numbers of ethanol and ethylene in every batch and concentration bin."""
    binned = add_bins(df).dropna(subset=["conc_bin"])

    parts = []
    for _, group in binned.groupby(["batch_id", "conc_bin"], observed=True):
        counts = group[TARGET].value_counts()
        if len(counts) < 2:
            continue

        n = counts.min()
        for gas in (1, 2):
            parts.append(group[group[TARGET] == gas].sample(n=n, random_state=seed))

    if not parts:
        return df.iloc[0:0]
    return pd.concat(parts)


def make_concentration_model():
    """Diagnostic model that sees concentration only."""
    return DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=0)


def score(model, test: pd.DataFrame, columns: list):
    """Balanced accuracy and macro F1 (NaN if only one gas is present)."""
    if test[TARGET].nunique() < 2:
        return np.nan, np.nan

    predictions = model.predict(test[columns])
    return (
        balanced_accuracy_score(test[TARGET], predictions),
        f1_score(test[TARGET], predictions, average="macro"),
    )


def sensitivity_table(parts: dict, k: int, concentration_range, seed: int):
    """Results for every condition, model and future batch."""
    low, high = concentration_range

    source = parts["source_train"]
    future = parts["future_test"]
    source_in_range = in_range(source, low, high)
    future_in_range = in_range(future, low, high)

    data = {
        "all": (source, future),
        "common_range": (source_in_range, future_in_range),
        "matched": (source_in_range, match_concentration(future_in_range, seed)),
    }
    models = {
        "sensors_only": (lambda: make_knn(k), FEATURES),
        "concentration_only": (make_concentration_model, CONCENTRATION),
    }

    rows = []
    for condition, (fit_data, test_data) in data.items():
        for model_name, (build, columns) in models.items():
            model = build()
            model.fit(fit_data[columns], fit_data[TARGET])

            for batch, test in test_data.groupby("batch_id"):
                balanced_accuracy, macro_f1 = score(model, test, columns)
                rows.append({
                    "condition": condition,
                    "model": model_name,
                    "batch_id": batch,
                    "n_ethanol": int((test[TARGET] == 1).sum()),
                    "n_ethylene": int((test[TARGET] == 2).sum()),
                    "balanced_accuracy": balanced_accuracy,
                    "macro_f1": macro_f1,
                })

    return pd.DataFrame(rows)


def plot_sensitivity(table: pd.DataFrame, output_path: Path):
    """Balanced accuracy per future batch, one panel per condition."""
    fig, axes = plt.subplots(
        1, 3, figsize=(13, 4), sharey=True, layout="constrained"
    )

    for ax, condition in zip(axes, CONDITIONS):
        part = table[table["condition"] == condition]
        for model_name, group in part.groupby("model"):
            ax.plot(
                group["batch_id"], group["balanced_accuracy"],
                marker="o", label=model_name,
            )
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_title(condition)
        ax.set_xlabel("Batch ID")
        ax.set_xticks([6, 7, 8, 9, 10])
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Balanced accuracy")
    axes[0].legend()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[1]

    config = yaml.safe_load(
        (project_root / "config.yaml").read_text(encoding="utf-8")
    )
    concentration_range = tuple(config["dataset"]["concentration_range"])
    seed = config["project"]["random_seed"]

    df = pd.read_csv(project_root / "data" / "processed" / "ethanol_ethylene.csv")
    parts = make_temporal_splits(df, config)

    # k is selected on the source validation data, as in knn.py.
    _, best_k = select_k(parts, FEATURES, config["models"]["knn"]["k_values"])

    distribution = concentration_distribution(df)
    sensitivity = sensitivity_table(parts, best_k, concentration_range, seed)

    tables_dir = project_root / "results" / "tables"
    figures_dir = project_root / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    distribution.to_csv(tables_dir / "concentration_distribution.csv", index=False)
    sensitivity.to_csv(tables_dir / "concentration_sensitivity.csv", index=False)
    plot_sensitivity(sensitivity, figures_dir / "concentration_sensitivity.png")

    print(f"k used: {best_k}\n")
    print(sensitivity.round(3).to_string(index=False))