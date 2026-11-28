"""One table with all sensor-level scores, the Figaro model and the subsets.

It only combines the tables created by `src/selection.py`
(`sensor_rankings.csv`, `selected_sensors.csv`) and adds the sensor model, so
all scores have a single source.

Sensor models: the dataset creators give the order of the sensors in the data
(channels CH0-CH15) as TGS2602, TGS2602, TGS2600, TGS2600, TGS2610, TGS2610,
TGS2620, TGS2620, followed by the same eight again (Fonollosa, Rodriguez-Lujan
and Huerta, Data in Brief 3, 2015, 85-89). Channel CH0 is taken as sensor 1.
"""

from pathlib import Path

import pandas as pd

SENSOR_MODELS = {
    1: "TGS2602", 2: "TGS2602", 3: "TGS2600", 4: "TGS2600",
    5: "TGS2610", 6: "TGS2610", 7: "TGS2620", 8: "TGS2620",
    9: "TGS2602", 10: "TGS2602", 11: "TGS2600", 12: "TGS2600",
    13: "TGS2610", 14: "TGS2610", 15: "TGS2620", 16: "TGS2620",
}

SCORE_COLUMNS = [
    "importance_score", "importance_std", "importance_rank",
    "stability_score", "stability_rank",
    "hybrid_score", "hybrid_rank",
]

DATA_DRIVEN = ["importance", "stability", "hybrid"]


def parse_ids(text) -> set:
    return {int(part) for part in str(text).split(";")}


def build_sensor_scores(rankings: pd.DataFrame, selected: pd.DataFrame):
    """One row per sensor with scores, ranks and subset membership."""
    rows = []

    for record in rankings.sort_values("sensor_id").to_dict("records"):
        sensor = int(record["sensor_id"])

        row = {"sensor_id": sensor, "figaro_model": SENSOR_MODELS[sensor]}
        row.update({column: record[column] for column in SCORE_COLUMNS})

        for budget in (4, 8):
            at_budget = selected[selected["budget"] == budget]
            data_driven = at_budget[at_budget["strategy"].isin(DATA_DRIVEN)]
            random_rows = at_budget[at_budget["strategy"] == "random"]

            row[f"selected_in_{budget}"] = ";".join(
                r.strategy
                for r in data_driven.itertuples()
                if sensor in parse_ids(r.sensor_ids)
            )
            row[f"random_seeds_{budget}"] = ";".join(
                str(int(r.seed))
                for r in random_rows.itertuples()
                if sensor in parse_ids(r.sensor_ids)
            )

        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[1]
    tables_dir = project_root / "results" / "tables"

    rankings = pd.read_csv(tables_dir / "sensor_rankings.csv")
    selected = pd.read_csv(tables_dir / "selected_sensors.csv")

    scores = build_sensor_scores(rankings, selected)
    scores.to_csv(tables_dir / "sensor_scores.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print(scores.round(4).to_string(index=False))