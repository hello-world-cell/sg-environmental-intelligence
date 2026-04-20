"""
Fetches the 24-hour PSI (Pollutant Standards Index) readings from data.gov.sg.

The PSI endpoint returns readings pre-aggregated by region — no station-level
parsing or aggregation is needed. Each call yields exactly 5 rows (one per
region: north, south, east, west, central).
"""

import time
import requests
import pandas as pd

_ENDPOINT = "https://api-open.data.gov.sg/v2/real-time/api/psi"


def retry_request(url: str, max_retries: int = 1) -> requests.Response:
    """
    GET url and retry once on 429 (rate limit) after a 5-second wait.
    Raises the underlying HTTP error if the retry also fails.
    """
    response = requests.get(url, timeout=10)
    if response.status_code == 429 and max_retries > 0:
        print(f"[retry] 429 on {url} — waiting 5s before retry...")
        time.sleep(5)
        response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response

_REGIONS = {"north", "south", "east", "west", "central"}


def fetch_psi() -> pd.DataFrame:
    """
    Fetch the latest 24-hour PSI readings and return one row per region.

    Response shape:
        data.items[0].timestamp
        data.items[0].readings.psi_twenty_four_hourly -> {north: int, south: int, ...}

    Returns
    -------
    DataFrame with columns: region, timestamp, metric ('psi_24h'), value
        Returns an empty DataFrame with the standard schema on any error.
    """
    try:
        payload = retry_request(_ENDPOINT).json()

        item      = payload["data"]["items"][0]
        timestamp = item["timestamp"]
        psi_by_region = item["readings"]["psi_twenty_four_hourly"]

        rows = [
            {
                "region":    region,
                "timestamp": timestamp,
                "metric":    "psi_24h",
                "value":     float(value),
            }
            for region, value in psi_by_region.items()
            if region in _REGIONS and value is not None
        ]

        print(f"[psi] Records fetched: {len(rows)}")

        if not rows:
            return pd.DataFrame(columns=["region", "timestamp", "metric", "value"])

        return pd.DataFrame(rows)[["region", "timestamp", "metric", "value"]]

    except requests.RequestException as exc:
        print(f"[psi] HTTP error: {exc}")
        return pd.DataFrame(columns=["region", "timestamp", "metric", "value"])

    except (KeyError, IndexError, TypeError, ValueError) as exc:
        print(f"[psi] Failed to parse PSI response: {exc}")
        return pd.DataFrame(columns=["region", "timestamp", "metric", "value"])
