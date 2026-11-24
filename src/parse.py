from pathlib import Path   # Dataset fayllarini tapmaq ucun
import math  # Yanlis v sonsuz reqemleri yoxlamaq
import pandas as pd #datani cedvele cevirir
import numpy as np

def parse_line(line: str) -> dict:  
    """Convert one UCI measurement into a Python dictionary."""

    parts = line.strip().split()  # metni ayri ayri elementlere cevirir

    if len(parts) != 129:  # Element sayi duz gelmese error versin
        raise ValueError(
            f"Expected 129 fields, received {len(parts)}"
        )

    header = parts[0].split(";")   # Birinci elementler

    if len(header) != 2:
        raise ValueError("Invalid gas/concentration header")

    gas_id = int(header[0])
    concentration = float(header[1])

    if gas_id not in range(1, 7):
        raise ValueError(f"Invalid gas ID: {gas_id}")

    if not math.isfinite(concentration) or concentration <= 0:  # Eger qiymet sonsuzdursa ve ya conc 0dan kicikdirse error versin
        raise ValueError("Invalid concentration")

    features = {}

    for item in parts[1:]:

        index_text, separator, value_text = item.partition(":")

        if separator != ":":  
            raise ValueError(f"Invalid feature: {item}")

        index = int(index_text)
        value = float(value_text)

        if not 1 <= index <= 128:
            raise ValueError(f"Invalid feature index: {index}")

        if not math.isfinite(value):
            raise ValueError(f"Non-finite feature value: {item}")

        feature_name = f"f{index:03d}"

        if feature_name in features:
            raise ValueError(f"Duplicate feature: {feature_name}")

        features[feature_name] = value

    if len(features) != 128:
        raise ValueError("Missing sensor features")

    row = {
        "gas_id": gas_id,
        "concentration_ppmv": concentration,
    }

    for i in range(1, 129):
        feature_name = f"f{i:03d}"
        row[feature_name] = features[feature_name]

    return row

def parse_batch(batch_path: Path, batch_id: int) -> pd.DataFrame:
    """Read all measurements from one batch file."""

    rows = []

    with open(batch_path, "r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            try:
                row = parse_line(line)

            except ValueError as error:
                raise ValueError(
                    f"Error in {batch_path.name}, "
                    f"line {line_number}: {error}"
                ) from error

            row["batch_id"] = batch_id

            row["sample_id"] = (
                f"b{batch_id:02d}_{line_number:05d}"
            )

            rows.append(row)

    if not rows:
        raise ValueError(f"Empty batch file: {batch_path}")

    df = pd.DataFrame(rows)

    feature_columns = [
        f"f{i:03d}" for i in range(1, 129)
    ]

    column_order = [
        "sample_id",
        "batch_id",
        "gas_id",
        "concentration_ppmv",
    ] + feature_columns

    df = df[column_order]

    return df

def load_all_batches(raw_folder: Path) -> pd.DataFrame:
    """Load and combine all 10 UCI batch files."""

    all_batches = []

    for batch_id in range(1, 11):

        filename = f"batch{batch_id}.dat"

        batch_files = list(raw_folder.rglob(filename))

        if len(batch_files) != 1:
            raise FileNotFoundError(
                f"Expected exactly one {filename}, "
                f"found {len(batch_files)}"
            )

        batch_path = batch_files[0]

        print(f"Reading {filename}...")

        batch_df = parse_batch(
            batch_path,
            batch_id
        )

        print(f"  Observations: {len(batch_df)}")

        all_batches.append(batch_df)

    combined_df = pd.concat(
        all_batches,
        ignore_index=True
    )

    if not combined_df["sample_id"].is_unique:
        raise ValueError("Duplicate sample IDs found")

    print("\n===== DATASET SUMMARY =====")

    print("Total observations:", len(combined_df))

    print("Total columns:", len(combined_df.columns))

    print("Total batches:", combined_df["batch_id"].nunique())

    print("\nGas distribution:")
    print(combined_df["gas_id"].value_counts().sort_index())

    return combined_df

def create_binary_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Select ethanol and ethylene measurements."""

    binary_df = df[
        df["gas_id"].isin([1, 2])
    ].copy()

    binary_df = binary_df.reset_index(drop=True)

    expected_counts = {
        1: 2565,  # Ethanol
        2: 2926,  # Ethylene
    }

    actual_counts = binary_df["gas_id"].value_counts().to_dict()

    if actual_counts != expected_counts:
        raise ValueError(
            f"Unexpected gas counts: {actual_counts}"
        )

    if not binary_df["sample_id"].is_unique:
        raise ValueError("Duplicate sample IDs detected")

    print("\n===== BINARY DATASET =====")
    print("Ethanol:", actual_counts[1])
    print("Ethylene:", actual_counts[2])
    print("Total:", len(binary_df))

    return binary_df

def create_data_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Create a data quality summary for each batch."""

    numeric_columns = [
        "concentration_ppmv"
    ] + [f"f{i:03d}" for i in range(1, 129)]

    records = []

    for batch_id, group in df.groupby("batch_id"):

        numeric_data = group[numeric_columns].to_numpy(dtype=float)

        record = {
            "batch_id": batch_id,
            "total_samples": len(group),
            "ethanol_samples": (group["gas_id"] == 1).sum(),
            "ethylene_samples": (group["gas_id"] == 2).sum(),
            "missing_values": group.isna().sum().sum(),
            "invalid_values": (~np.isfinite(numeric_data)).sum(),
            "duplicate_rows": group.drop(
                columns=["sample_id"]
            ).duplicated().sum(),
        }

        records.append(record)

    return pd.DataFrame(records)

if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[1]

    raw_folder = project_root / "data" / "raw" / "uci270"
    output_folder = project_root / "data" / "processed"

    output_folder.mkdir(parents=True, exist_ok=True)

    # 1. Load all six gases
    full_df = load_all_batches(raw_folder)

    # 2. Select ethanol and ethylene
    binary_df = create_binary_dataset(full_df)

    # 3. Check dataset integrity
    assert len(full_df) == 13910
    assert len(binary_df) == 5491
    assert len(binary_df.columns) == 132
    assert binary_df["sample_id"].is_unique
    assert not binary_df.isna().any().any()

    # 4. Create data quality report
    audit_df = create_data_audit(full_df)

    # 5. Save datasets
    full_df.to_csv(
        output_folder / "all_measurements.csv",
        index=False
    )

    binary_df.to_csv(
        output_folder / "ethanol_ethylene.csv",
        index=False
    )

    audit_df.to_csv(
        output_folder / "data_audit.csv",
        index=False
    )

    print("\n===== FILES SAVED =====")

    print("all_measurements.csv")
    print("ethanol_ethylene.csv")
    print("data_audit.csv")

    print("\nParser completed successfully!")