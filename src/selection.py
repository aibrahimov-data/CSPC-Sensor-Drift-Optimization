"""Sensor selection strategies.

Every ranking uses source batches 1-5 only (training 1-3, validation 4-5).
Batches 6-10 are never used.

Strategies:

* all        - all 16 sensors (full-array reference, budget 16 only)
* random     - random sensors; one subset per sampling seed
* importance - sensors with the largest permutation importance
* stability  - sensors with the smallest source-period shift (see drift.py)
* hybrid     - mean of the importance rank and the stability rank (50% / 50%)

For every selected sensor all eight of its features are kept.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import brier_score_loss

from src.drift import stability_scores
from src.knn import FEATURES, TARGET, make_knn, select_k
from src.sensors import sensor_features
from src.split import make_temporal_splits

ALL_SENSORS = list(range(1, 17))

CRITERIA = {
    "all": "all 16 sensors (full array)",
    "random": "random sensors, seed = calibration sampling seed",
    "importance": "largest permutation importance (batches 1-3 fit, batches 4-5)",
    "stability": "smallest source-period shift (batches 4-5 vs 1-3)",
    "hybrid": "mean of importance rank and stability rank",
}

RANK_COLUMNS = {
    "importance": "importance_rank",
    "stability": "stability_rank",
    "hybrid": "hybrid_rank",
}


def importance_scores(parts: dict, k: int, n_repeats: int = 20, seed: int = 42):
    """Sensor-group permutation importance on the source validation batches.

    KNN is fitted on batches 1-3. The eight features of one sensor are
    shuffled together across the validation rows, and the increase of the
    Brier score is measured. A larger increase means a more important sensor.
    The Brier score is used because the validation accuracy is already
    saturated, so accuracy alone cannot separate the sensors.
    """
    train = parts["train"]
    validation = parts["validation"]

    model = make_knn(k)
    model.fit(train[FEATURES], train[TARGET])

    positive = list(model.classes_).index(2)
    y_true = validation[TARGET]

    def brier(features):
        probability = model.predict_proba(features)[:, positive]
        return brier_score_loss(y_true, probability, pos_label=2)

    baseline = brier(validation[FEATURES])
    rng = np.random.default_rng(seed)

    rows = []
    for sensor in ALL_SENSORS:
        columns = sensor_features([sensor])
        increases = []

        for _ in range(n_repeats):
            shuffled = validation[FEATURES].copy()
            order = rng.permutation(len(shuffled))
            shuffled[columns] = shuffled[columns].to_numpy()[order]
            increases.append(brier(shuffled) - baseline)

        rows.append({
            "sensor_id": sensor,
            "importance_score": float(np.mean(increases)),
            "importance_std": float(np.std(increases)),
        })

    result = pd.DataFrame(rows)
    result["importance_rank"] = (
        result["importance_score"].rank(ascending=False, method="first").astype(int)
    )
    return result.sort_values("importance_rank").reset_index(drop=True)


def combine_rankings(importance: pd.DataFrame, stability: pd.DataFrame):
    """Merge both rankings and add the hybrid rank (50% / 50% rank sum)."""
    table = importance.merge(stability, on="sensor_id")

    table["hybrid_score"] = 0.5 * table["importance_rank"] + 0.5 * table["stability_rank"]

    # Ties go to the more important sensor, then to the smaller sensor ID.
    order = table.sort_values(["hybrid_score", "importance_rank", "sensor_id"])
    ranks = pd.Series(range(1, len(order) + 1), index=order.index)
    table["hybrid_rank"] = ranks

    return table.sort_values("sensor_id").reset_index(drop=True)


def select_sensors(strategy: str, budget: int, rankings: pd.DataFrame, seed=None):
    """Sorted sensor IDs chosen by one strategy for one sensor budget."""
    if strategy == "all":
        return list(ALL_SENSORS)

    if strategy == "random":
        rng = np.random.default_rng(seed)
        chosen = rng.choice(ALL_SENSORS, size=budget, replace=False)
        return sorted(int(sensor) for sensor in chosen)

    best = rankings.sort_values(RANK_COLUMNS[strategy]).head(budget)
    return sorted(int(sensor) for sensor in best["sensor_id"])


def build_selection_table(rankings: pd.DataFrame, config: dict) -> pd.DataFrame:
    """One row per sensor subset that the experiments will use."""
    budgets = config["sensor_selection"]["budgets"]
    seeds = config["recalibration"]["sampling_seeds"]

    rows = []
    for strategy in config["sensor_selection"]["strategies"]:
        for budget in budgets:
            # Budget 16 is the full array, so only "all" is used there.
            if strategy == "all" and budget != 16:
                continue
            if strategy != "all" and budget == 16:
                continue

            seed_list = seeds if strategy == "random" else [None]

            for seed in seed_list:
                sensors = select_sensors(strategy, budget, rankings, seed)
                rows.append({
                    "strategy": strategy,
                    "budget": budget,
                    "seed": seed,
                    "sensor_ids": ";".join(str(s) for s in sensors),
                    "n_features": 8 * len(sensors),
                    "features": ";".join(sensor_features(sensors)),
                    "criterion": CRITERIA[strategy],
                })

    table = pd.DataFrame(rows)
    table["seed"] = table["seed"].astype("Int64")
    return table


def plot_importance_vs_stability(rankings: pd.DataFrame, output_path: Path):
    """Importance against stability for every sensor (source only)."""
    fig, ax = plt.subplots(figsize=(6.5, 5))

    ax.scatter(rankings["stability_score"], rankings["importance_score"], s=40)
    for _, row in rankings.iterrows():
        ax.annotate(
            str(int(row["sensor_id"])),
            (row["stability_score"], row["importance_score"]),
            textcoords="offset points",
            xytext=(4, 4),
        )

    ax.set_xlabel("Stability score (lower = more stable)")
    ax.set_ylabel("Importance (increase of Brier score when shuffled)")
    ax.set_title("Sensor ranking, source batches only")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[1]

    config = yaml.safe_load(
        (project_root / "config.yaml").read_text(encoding="utf-8")
    )
    low, high = config["dataset"]["concentration_range"]

    df = pd.read_csv(project_root / "data" / "processed" / "ethanol_ethylene.csv")
    parts = make_temporal_splits(df, config)

    # k is selected on the source validation data, as in knn.py.
    _, best_k = select_k(parts, FEATURES, config["models"]["knn"]["k_values"])

    importance = importance_scores(parts, best_k)
    stability = stability_scores(parts, (low, high))
    rankings = combine_rankings(importance, stability)
    selection = build_selection_table(rankings, config)

    tables_dir = project_root / "results" / "tables"
    figures_dir = project_root / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rankings.to_csv(tables_dir / "sensor_rankings.csv", index=False)
    selection.to_csv(tables_dir / "selected_sensors.csv", index=False)
    plot_importance_vs_stability(rankings, figures_dir / "sensor_importance_vs_stability.png")

    print(f"k used for importance: {best_k}\n")
    print("Sensor rankings (source only):")
    print(rankings.sort_values("hybrid_rank").to_string(index=False))
    print("\nSelected sensor subsets:")
    print(selection[["strategy", "budget", "seed", "sensor_ids"]].to_string(index=False))