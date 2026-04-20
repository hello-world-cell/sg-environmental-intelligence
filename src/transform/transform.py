"""
Transform module — cleans and reshapes the raw long-format DataFrame
into a wide-format processed dataset ready for quality checks and
recommendation generation.
"""

import pandas as pd
from datetime import datetime

_OUTPUT_PATH = "data/processed/processed_env_data.csv"


def run(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and pivot the raw environmental DataFrame from long to wide format.

    Steps
    -----
    1. Convert timestamp column to datetime.
    2. Remove duplicate rows.
    3. Pivot long -> wide (region x metric), averaging duplicate region/metric pairs.
    4. Reset index so region becomes a regular column.
    5. Add retrieved_at column with the current datetime.
    6. Save to data/processed/processed_env_data.csv.
    7. Return the wide DataFrame.

    Parameters
    ----------
    raw_data : DataFrame with columns [region, timestamp, metric, value]

    Returns
    -------
    Wide DataFrame indexed by region with one column per metric,
    plus a retrieved_at column.
    """
    print("[transform] Processing raw data...")

    if raw_data.empty:
        print("[transform] Warning: received empty DataFrame — skipping.")
        return pd.DataFrame()

    df = raw_data.copy()

    # 1. Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # 2. Remove duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped:
        print(f"[transform] Dropped {dropped} duplicate row(s).")

    # 3. Pivot long -> wide; handle numeric and string metrics separately
    #    because aggfunc="mean" cannot operate on forecast/heat_stress strings.
    df["_num"] = pd.to_numeric(df["value"], errors="coerce")
    numeric_df = df[df["_num"].notna()].copy()
    string_df  = df[df["_num"].isna()].copy()

    parts = []

    if not numeric_df.empty:
        wide_num = numeric_df.pivot_table(
            index="region",
            columns="metric",
            values="_num",
            aggfunc="mean",
        )
        parts.append(wide_num)

    if not string_df.empty:
        wide_str = string_df.pivot_table(
            index="region",
            columns="metric",
            values="value",
            aggfunc=lambda x: x.mode().iloc[0],
        )
        parts.append(wide_str)

    wide = pd.concat(parts, axis=1)

    # 4. Reset index so region becomes a plain column
    wide = wide.reset_index()
    wide.columns.name = None    # remove the "metric" label from the column axis

    # 4b. Fill nulls:
    #     - uv_index: forward-fill then backward-fill (island-wide value shared
    #       across all regions; the single island-wide row seeds the fill)
    #     - all other numeric columns: fill with the column median
    _SKIP_FILL = {"region", "retrieved_at", "forecast_2hr", "heat_stress_level"}

    if "uv_index" in wide.columns:
        n_null = int(wide["uv_index"].isna().sum())
        wide["uv_index"] = wide["uv_index"].ffill().bfill()
        filled = n_null - int(wide["uv_index"].isna().sum())
        if filled:
            print(f"[transform] uv_index          : filled {filled} null(s) via forward/back-fill")

    for col in wide.columns:
        if col in _SKIP_FILL or col == "uv_index":
            continue
        if not pd.api.types.is_numeric_dtype(wide[col]):
            continue
        n_null = int(wide[col].isna().sum())
        if n_null:
            median = wide[col].median()
            wide[col] = wide[col].fillna(median)
            print(f"[transform] {col:<20}: filled {n_null} null(s) with median ({median:.2f})")

    # 5. Add retrieval timestamp
    wide["retrieved_at"] = datetime.now()

    # 6. Print shape and columns
    print(f"[transform] Shape: {wide.shape[0]} rows x {wide.shape[1]} columns")
    print(f"[transform] Columns: {list(wide.columns)}")

    # 7. Save to CSV
    wide.to_csv(_OUTPUT_PATH, index=False)
    print(f"[transform] Saved to {_OUTPUT_PATH}")

    return wide
