"""
Fetches the latest UV index reading from data.gov.sg.

UV is measured island-wide (not region-specific), so this returns a single
row with region = "island-wide". The index list is sorted newest-first;
the first entry is always the most recent hourly reading.
"""

import time
import requests
import pandas as pd

_ENDPOINT = "https://api-open.data.gov.sg/v2/real-time/api/uv"


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


def fetch_uv() -> pd.DataFrame:
    """
    Fetch the most recent island-wide UV index reading.

    Response shape:
        data.records[0].index[0].hour   -> timestamp of latest reading
        data.records[0].index[0].value  -> UV index (integer)

    Returns
    -------
    DataFrame with columns: region ('island-wide'), timestamp, metric ('uv_index'), value
        Returns an empty DataFrame with the standard schema on any error.
    """
    try:
        payload = retry_request(_ENDPOINT).json()

        record  = payload["data"]["records"][0]
        latest  = record["index"][0]          # index list is newest-first

        timestamp = latest["hour"]
        uv_value  = float(latest["value"])

        return pd.DataFrame([{
            "region":    "island-wide",
            "timestamp": timestamp,
            "metric":    "uv_index",
            "value":     uv_value,
        }])

    except requests.RequestException as exc:
        print(f"[uv] HTTP error: {exc}")
        return pd.DataFrame(columns=["region", "timestamp", "metric", "value"])

    except (KeyError, IndexError, TypeError, ValueError) as exc:
        print(f"[uv] Failed to parse UV response: {exc}")
        return pd.DataFrame(columns=["region", "timestamp", "metric", "value"])
