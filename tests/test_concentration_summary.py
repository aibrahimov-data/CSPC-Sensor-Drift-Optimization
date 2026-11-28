import pandas as pd

from src.concentration_summary import concentration_summary


def make_df():
    rows = [
        {"batch_id": batch, "gas_id": gas, "concentration_ppmv": float(level)}
        for batch in range(1, 8)
        for gas in (1, 2)
        for level in (10, 50, 100)
    ]
    df = pd.DataFrame(rows)

    # One ethylene value in batch 7 above both the common and stated range.
    mask = (
        (df["batch_id"] == 7)
        & (df["gas_id"] == 2)
        & (df["concentration_ppmv"] == 100.0)
    )
    df.loc[mask, "concentration_ppmv"] = 500.0
    return df


def get_row(summary, gas, batch):
    return summary[(summary["gas_id"] == gas) & (summary["batch_id"] == batch)].iloc[0]


def test_summary_has_one_row_per_gas_and_batch():
    df = make_df()

    summary = concentration_summary(df, (10, 300), [1, 2, 3])

    assert len(summary) == 14
    assert summary["n_measurements"].sum() == len(df)


def test_flags_for_values_outside_the_ranges():
    summary = concentration_summary(make_df(), (10, 300), [1, 2, 3])

    ethylene = get_row(summary, gas=2, batch=7)
    assert ethylene["ppmv_max"] == 500.0
    assert ethylene["n_above_common_range"] == 1
    assert ethylene["n_outside_stated_gas_range"] == 1

    ethanol = get_row(summary, gas=1, batch=7)
    assert ethanol["n_above_common_range"] == 0
    assert ethanol["n_outside_stated_gas_range"] == 0


def test_level_overlap_with_reference_batches():
    summary = concentration_summary(make_df(), (10, 300), [1, 2, 3])

    ethylene = get_row(summary, gas=2, batch=7)
    assert ethylene["n_levels"] == 3
    assert ethylene["n_shared_levels_with_reference"] == 2
    assert abs(ethylene["share_obs_at_shared_levels"] - 2 / 3) < 1e-9

    ethanol = get_row(summary, gas=1, batch=7)
    assert ethanol["n_shared_levels_with_reference"] == 3
    assert ethanol["share_obs_at_shared_levels"] == 1.0