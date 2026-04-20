"""
Fetches real-time Wet Bulb Globe Temperature (WBGT) readings from data.gov.sg.

Endpoint: GET /v2/real-time/api/weather?api=wbgt
  - Station metadata (id, name, lat, lon) is embedded inside each reading.
  - Latitude, longitude, wbgt, and heatStress are returned as strings
    and are cast to the appropriate types on parse.

Returns a single combined DataFrame with two metrics per region:
  - wbgt_c            : mean numeric WBGT reading (degrees C)
  - heat_stress_level : plurality heat-stress category (low/moderate/high)
"""

import requests
import pandas as pd

from src.extract.weather_stations import _assign_region

_ENDPOINT = "https://api-open.data.gov.sg/v2/real-time/api/weather"
_PARAMS   = {"api": "wbgt"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mode(series: pd.Series) -> str:
    """Return the most frequent value in a Series (plurality vote)."""
    return series.mode().iloc[0]


def _parse_records(payload: dict) -> tuple[str, list[dict]]:
    """
    Extract (timestamp, readings) from the latest record in the response.

    Response shape:
        data.records[0].datetime          -> timestamp string
        data.records[0].item.readings[]   -> list of station reading objects
    """
    record   = payload["data"]["records"][0]
    timestamp = record["datetime"]
    readings  = record["item"]["readings"]
    return timestamp, readings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_wbgt() -> pd.DataFrame:
    """
    Fetch real-time WBGT station data and return a region-aggregated DataFrame.

    Each station reading embeds its own location and heat stress level.
    Stations are mapped to regions via lat/lon bounding boxes (same rules
    as weather_stations.py). Numeric WBGT is averaged per region;
    heat stress category is chosen by plurality vote per region.

    Aggregation
    -----------
    wbgt          -> mean per region  (metric: "wbgt_c")
    heatStress    -> mode per region  (metric: "heat_stress_level")
                     normalised to lowercase to match engine.py advisory keys

    Returns an empty DataFrame with the standard schema on any error.
    """
    try:
        response = requests.get(_ENDPOINT, params=_PARAMS, timeout=10)
        response.raise_for_status()
        payload = response.json()

        timestamp, readings = _parse_records(payload)

        rows    = []
        skipped = 0

        for reading in readings:
            try:
                lat          = float(reading["location"]["latitude"])
                lon          = float(reading["location"]["longitude"])
                wbgt_value   = float(reading["wbgt"])
                heat_stress  = str(reading["heatStress"]).lower()        # normalise to lowercase
            except (KeyError, ValueError, TypeError):
                skipped += 1
                continue

            region = _assign_region(lat, lon)
            rows.append({
                "region":            region,
                "wbgt_value":        wbgt_value,
                "heat_stress_level": heat_stress,
            })

        print(f"[wbgt] Stations read: {len(rows)} | skipped (incomplete): {skipped}")

        if not rows:
            return pd.DataFrame(columns=["region", "timestamp", "metric", "value"])

        df = pd.DataFrame(rows)

        # --- numeric: mean wbgt per region -----------------------------------
        wbgt_numeric_df = (
            df.groupby("region", as_index=False)
            .agg(value=("wbgt_value", "mean"))
        )
        wbgt_numeric_df["timestamp"] = timestamp
        wbgt_numeric_df["metric"]    = "wbgt_c"

        # --- categorical: mode heat_stress per region ------------------------
        wbgt_level_df = (
            df.groupby("region")["heat_stress_level"]
            .agg(_mode)
            .reset_index()
            .rename(columns={"heat_stress_level": "value"})
        )
        wbgt_level_df["timestamp"] = timestamp
        wbgt_level_df["metric"]    = "heat_stress_level"

        # --- combine into one standardised DataFrame -------------------------
        combined = pd.concat(
            [
                wbgt_numeric_df[["region", "timestamp", "metric", "value"]],
                wbgt_level_df[["region", "timestamp", "metric", "value"]],
            ],
            ignore_index=True,
        )

        return combined

    except requests.RequestException as exc:
        print(f"[wbgt] HTTP error: {exc}")
        return pd.DataFrame(columns=["region", "timestamp", "metric", "value"])

    except (KeyError, IndexError, ValueError) as exc:
        print(f"[wbgt] Failed to parse WBGT response: {exc}")
        return pd.DataFrame(columns=["region", "timestamp", "metric", "value"])
