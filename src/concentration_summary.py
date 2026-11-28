"""Descriptive concentration summary per gas and batch.

Creates one table with the number of measurements, ppmv statistics, flags for
suspicious values and the overlap of concentration levels with the reference
batches. It is descriptive only and is not used for tuning or selection.
"""

from pathlib import Path

import pandas as pd
import yaml

from src.knn import TARGET

GAS_NAMES = {1: "Ethanol", 2: "Ethylene"}

# Concentration ranges (ppmv) stated in the team plan for each gas.
STATED_RANGES = {1: (10, 600), 2: (10, 300)}


def concentration_summary(df: pd.DataFrame, common_range, reference_batches):
    """One row per gas and batch.

    Besides counts and ppmv statistics, the table shows how many concentration
    levels of a batch also occur in the reference batches for the same gas.
    This tells whether a temporal comparison at the same gas and the same
    concentration is possible.
    """
    low, high = common_range

    df = df.assign(level=df["concentration_ppmv"].round(1))
    reference = df[df["batch_id"].isin(reference_batches)]

    rows = []
    for (gas, batch), part in df.groupby([TARGET, "batch_id"]):
        concentration = part["concentration_ppmv"]
        level_counts = part["level"].value_counts()

        reference_levels = set(reference.loc[reference[TARGET] == gas, "level"])
        shared = part["level"].isin(reference_levels)
        stated_low, stated_high = STATED_RANGES[gas]

        rows.append({
            "gas_id": gas,
            "gas_name": GAS_NAMES[gas],
            "batch_id": batch,
            "n_measurements": len(part),
            "ppmv_min": concentration.min(),
            "ppmv_median": concentration.median(),
            "ppmv_max": concentration.max(),
            "n_levels": int(level_counts.size),
            "most_common_level_ppmv": level_counts.index[0],
            "most_common_level_share": level_counts.iloc[0] / len(part),
            "n_in_common_range": int(concentration.between(low, high).sum()),
            "n_above_common_range": int((concentration > high).sum()),
            "n_outside_stated_gas_range": int(
                ((concentration < stated_low) | (concentration > stated_high)).sum()
            ),
            "n_missing_or_nonpositive": int(
                (concentration.isna() | (concentration <= 0)).sum()
            ),
            "n_shared_levels_with_reference": len(
                set(part["level"]) & reference_levels
            ),
            "share_obs_at_shared_levels": float(shared.mean()),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[1]

    config = yaml.safe_load(
        (project_root / "config.yaml").read_text(encoding="utf-8")
    )
    common_range = tuple(config["dataset"]["concentration_range"])
    reference_batches = config["splitting"]["train_batches"]

    df = pd.read_csv(project_root / "data" / "processed" / "ethanol_ethylene.csv")

    summary = concentration_summary(df, common_range, reference_batches)

    tables_dir = project_root / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables_dir / "concentration_summary.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print(summary.to_string(index=False))