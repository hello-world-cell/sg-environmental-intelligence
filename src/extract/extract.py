"""
Extract module — orchestrates all data source fetchers and returns
a single combined DataFrame of raw environmental readings.
"""

import time
import pandas as pd

from src.extract.weather_stations import (
    fetch_rainfall,
    fetch_temperature,
    fetch_humidity,
    fetch_wind_speed,
)
from src.extract.forecast import fetch_forecast
from src.extract.wbgt import fetch_wbgt
from src.extract.psi import fetch_psi
from src.extract.uv import fetch_uv


def run() -> pd.DataFrame:
    """
    Call every extractor, concatenate results, and return a single DataFrame.

    Returns
    -------
    DataFrame with columns: region, timestamp, metric, value
        Contains all metrics across all regions from every data source.
        Returns an empty DataFrame (same schema) if all sources fail.
    """
    print("[extract] Fetching environmental data from all sources...")

    fetchers = {
        "rainfall":    fetch_rainfall,
        "temperature": fetch_temperature,
        "humidity":    fetch_humidity,
        "wind_speed":  fetch_wind_speed,
        "forecast":    fetch_forecast,
        "wbgt":        fetch_wbgt,
        "psi":         fetch_psi,
        "uv":          fetch_uv,
    }

    frames = []
    for name, fetcher in fetchers.items():
        try:
            df = fetcher()
            if not df.empty:
                frames.append(df)
                print(f"[extract]   {name:<12} -> {len(df)} rows")
            else:
                print(f"[extract]   {name:<12} -> no data returned")
            time.sleep(1)    # avoid rate-limiting across consecutive API calls
        except Exception as exc:
            print(f"[extract]   {name:<12} -> failed: {exc}")
        print()

    if not frames:
        print("[extract] Warning: all sources failed — returning empty DataFrame.")
        return pd.DataFrame(columns=["region", "timestamp", "metric", "value"])

    combined = pd.concat(frames, ignore_index=True)
    print(f"[extract] Done - {len(combined)} total rows across {combined['metric'].nunique()} metrics.")
    return combined
