"""
Fetches real-time weather station readings from data.gov.sg and returns
region-aggregated DataFrames in a standardised format.
"""

import time
import requests
import pandas as pd

# ---------------------------------------------------------------------------
# Endpoint registry
# ---------------------------------------------------------------------------

_ENDPOINTS = {
    "rainfall":    "https://api-open.data.gov.sg/v2/real-time/api/rainfall",
    "temperature": "https://api-open.data.gov.sg/v2/real-time/api/air-temperature",
    "humidity":    "https://api-open.data.gov.sg/v2/real-time/api/relative-humidity",
    "wind_speed":  "https://api-open.data.gov.sg/v2/real-time/api/wind-speed",
} # can change API here , instead of 4 differnt functions

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assign_region(lat: float, lon: float) -> str:
    """Map a (lat, lon) coordinate to one of five Singapore regions."""
    if lat > 1.38:
        return "north"
    if lat < 1.28:
        return "south"
    if lon > 103.85:          # lat implicitly 1.28–1.38 from above checks
        return "east"
    if lon < 103.80:
        return "west"
    return "central"


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


def _fetch_raw(endpoint: str) -> dict:
    """GET an endpoint and return the parsed JSON, raising on HTTP errors."""
    return retry_request(endpoint).json()


def _parse_stations(payload: dict) -> dict:
    """
    Extract a {stationId: {"lat": ..., "lon": ...}} lookup from the response.

    The API nests station metadata under data -> stations, each with a
    location sub-object containing latitude and longitude.
    """
    stations = {}
    for s in payload["data"]["stations"]:
        stations[s["id"]] = {
            "lat": s["location"]["latitude"],
            "lon": s["location"]["longitude"],
        }
    return stations


def _latest_readings(payload: dict) -> tuple[str, list[dict]]:
    """
    Return (timestamp, readings) from the most recent reading block.

    Readings is a list of {"stationId": ..., "value": ...} dicts.
    """
    readings_list = payload["data"]["readings"]
    latest = readings_list[-1]          # last entry is the most recent
    return latest["timestamp"], latest["data"]


def _build_dataframe(
    payload: dict,
    metric: str,
    agg_func: str,
) -> pd.DataFrame:
    """
    Core builder: parse payload -> assign regions -> aggregate -> standardise.

    Parameters
    ----------
    payload   : raw API JSON dict
    metric    : human-readable metric name (e.g. 'rainfall_mm')
    agg_func  : 'sum' for rainfall, 'mean' for everything else
    """
    station_meta = _parse_stations(payload)
    timestamp, readings = _latest_readings(payload)

    rows = []
    for reading in readings:
        sid = reading["stationId"]
        value = reading["value"]
        meta = station_meta.get(sid)
        if meta is None or value is None:
            continue
        region = _assign_region(meta["lat"], meta["lon"])
        rows.append({"region": region, "value": float(value)})

    if not rows:
        return pd.DataFrame(columns=["region", "timestamp", "metric", "value"])

    df = pd.DataFrame(rows)
    agg = df.groupby("region", as_index=False).agg(value=("value", agg_func))
    agg["timestamp"] = timestamp
    agg["metric"] = metric

    return agg[["region", "timestamp", "metric", "value"]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_rainfall() -> pd.DataFrame:
    """
    Fetch real-time rainfall readings and return region-level sums.

    Rainfall is summed across stations within each region because multiple
    gauges capture the total volumetric contribution across the area.

    Returns
    -------
    DataFrame with columns: region, timestamp, metric ('rainfall_mm'), value
    """
    payload = _fetch_raw(_ENDPOINTS["rainfall"])
    return _build_dataframe(payload, metric="rainfall_mm", agg_func="sum")


def fetch_temperature() -> pd.DataFrame:
    """
    Fetch real-time air-temperature readings and return region-level means.

    Returns
    -------
    DataFrame with columns: region, timestamp, metric ('temperature_c'), value
    """
    payload = _fetch_raw(_ENDPOINTS["temperature"])
    return _build_dataframe(payload, metric="temperature_c", agg_func="mean")


def fetch_humidity() -> pd.DataFrame:
    """
    Fetch real-time relative-humidity readings and return region-level means.

    Returns
    -------
    DataFrame with columns: region, timestamp, metric ('humidity_pct'), value
    """
    payload = _fetch_raw(_ENDPOINTS["humidity"])
    return _build_dataframe(payload, metric="humidity_pct", agg_func="mean")


def fetch_wind_speed() -> pd.DataFrame:
    """
    Fetch real-time wind-speed readings and return region-level means.

    Returns
    -------
    DataFrame with columns: region, timestamp, metric ('wind_speed_kmh'), value
    """
    payload = _fetch_raw(_ENDPOINTS["wind_speed"])
    return _build_dataframe(payload, metric="wind_speed_kmh", agg_func="mean")
