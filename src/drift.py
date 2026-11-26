"""Temporal drift and sensor stability analysis.

Standardized mean shift of a feature, for one gas and one batch:

    (mean in batch - mean in reference batches) / std in reference batches

The reference is the initial training period (batches 1-3). A sensor-level
value is the mean of the absolute shifts of its eight features.

Two products are created:

* Sensor stability scores (source-only): batches 4-5 compared with batches
  1-3. These are the only values that may be used for sensor selection.
* A descriptive drift table for all batches. The values for batches 6-10 are
  descriptive only and must never be used to select sensors.

Only observations inside the common concentration range are used.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.knn import FEATURES, TARGET
from src.split import make_temporal_splits

GAS_NAMES = {1: "Ethanol", 2: "Ethylene"}


def sensor_of_feature(feature_name: str) -> int:
    """f001-f008 -> sensor 1, f009-f016 -> sensor 2, ..., f128 -> sensor 16."""
    return (int(feature_name[1:]) - 1) // 8 + 1


def in_range(df: pd.DataFrame, low: float, high: float) -> pd.DataFrame:
    """Keep observations inside the common concentration range."""
    return df[df["concentration_ppmv"].between(low, high)]


def compute_shifts(df: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Sensor-level mean absolute standardized shift per batch and gas."""
    grouped = reference.groupby(TARGET)[FEATURES]
    reference_mean = grouped.mean()
    reference_std = grouped.std().replace(0, np.nan)

    rows = []
    for (batch, gas), part in df.groupby(["batch_id", TARGET]):
        shift = (part[FEATURES].mean() - reference_mean.loc[gas]) / reference_std.loc[gas]
        sensor_shift = shift.abs().groupby(sensor_of_feature).mean()

        for sensor, value in sensor_shift.items():
            rows.append({
                "batch_id": batch,
                "gas_id": gas,
                "sensor_id": sensor,
                "mean_abs_shift": value,
                "n_observations": len(part),
            })

    return pd.DataFrame(rows)


def stability_scores(parts: dict, concentration_range) -> pd.DataFrame:
    """Source-only stability: batches 4-5 (pooled) against batches 1-3.

    A lower score means a smaller shift, i.e. a more stable sensor. Rank 1 is
    the most stable sensor. Future batches are never used.
    """
    low, high = concentration_range

    reference = in_range(parts["train"], low, high)
    comparison = in_range(parts["validation"], low, high).assign(batch_id=45)

    shifts = compute_shifts(comparison, reference)

    # Average over the two gases, so each gas has the same weight.
    score = shifts.groupby("sensor_id")["mean_abs_shift"].mean()

    result = score.rename("stability_score").reset_index()
    result["stability_rank"] = result["stability_score"].rank(method="first").astype(int)

    return result.sort_values("stability_rank").reset_index(drop=True)


def plot_drift_heatmap(drift_table: pd.DataFrame, output_path: Path):
    """Sensor x batch heatmap of the mean absolute standardized shift."""
    vmax = np.nanpercentile(drift_table["mean_abs_shift"], 95)

    fig, axes = plt.subplots(
        1, 2, figsize=(11, 5), sharey=True, layout="constrained"
    )

    for ax, gas in zip(axes, (1, 2)):
        grid = (
            drift_table[drift_table["gas_id"] == gas]
            .pivot(index="sensor_id", columns="batch_id", values="mean_abs_shift")
            .reindex(index=range(1, 17), columns=range(1, 11))
        )
        image = ax.imshow(
            grid.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=vmax
        )
        ax.set_xticks(range(10))
        ax.set_xticklabels(range(1, 11))
        ax.set_yticks(range(16))
        ax.set_yticklabels(range(1, 17))
        ax.set_title(GAS_NAMES[gas])
        ax.set_xlabel("Batch ID")

    axes[0].set_ylabel("Sensor ID")
    fig.colorbar(
        image, ax=axes,
        label="Mean |standardized shift| (colour capped at 95th percentile)",
    )
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

    kept = len(in_range(df, low, high))
    print(f"Observations inside {low}-{high} ppmv: {kept} of {len(df)}\n")

    # Descriptive drift table, all batches, reference = batches 1-3.
    drift_table = compute_shifts(
        in_range(df, low, high), in_range(parts["train"], low, high)
    )

    # Source-only stability scores for sensor selection.
    stability = stability_scores(parts, (low, high))

    tables_dir = project_root / "results" / "tables"
    figures_dir = project_root / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    drift_table.to_csv(tables_dir / "sensor_drift_by_batch.csv", index=False)
    stability.to_csv(tables_dir / "sensor_stability.csv", index=False)
    plot_drift_heatmap(drift_table, figures_dir / "sensor_drift_heatmap.png")

    print("Sensor stability (source only, rank 1 = most stable):")
    print(stability.to_string(index=False))