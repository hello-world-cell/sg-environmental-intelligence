"""
Town-level location data for Singapore.

Provides lat/lon coordinates for 25 towns, region groupings, and helpers
to map any town to its nearest weather station and pull its metrics from
the wide processed DataFrame.
"""

import math
import pandas as pd

# ---------------------------------------------------------------------------
# Town coordinates
# ---------------------------------------------------------------------------

TOWNS: dict[str, tuple[float, float]] = {
    "Sengkang":      (1.3868, 103.8914),
    "Punggol":       (1.4043, 103.9022),
    "Yishun":        (1.4304, 103.8354),
    "Woodlands":     (1.4382, 103.7890),
    "Ang Mo Kio":    (1.3691, 103.8454),
    "Bishan":        (1.3526, 103.8352),
    "Toa Payoh":     (1.3343, 103.8470),
    "Novena":        (1.3204, 103.8438),
    "Orchard":       (1.3048, 103.8318),
    "Marina Bay":    (1.2789, 103.8536),
    "Tampines":      (1.3496, 103.9568),
    "Pasir Ris":     (1.3721, 103.9493),
    "Bedok":         (1.3236, 103.9273),
    "Geylang":       (1.3201, 103.8918),
    "Kallang":       (1.3100, 103.8706),
    "Jurong East":   (1.3329, 103.7436),
    "Jurong West":   (1.3404, 103.7090),
    "Bukit Batok":   (1.3590, 103.7637),
    "Choa Chu Kang": (1.3840, 103.7470),
    "Bukit Panjang": (1.3774, 103.7719),
    "Clementi":      (1.3162, 103.7649),
    "Queenstown":    (1.2942, 103.7861),
    "Bukit Merah":   (1.2819, 103.8239),
    "Tanjong Pagar": (1.2762, 103.8458),
    "Buona Vista":   (1.3071, 103.7900),
}

# ---------------------------------------------------------------------------
# Region groupings
# ---------------------------------------------------------------------------

REGION_TOWNS: dict[str, list[str]] = {
    "North": [
        "Woodlands", "Yishun", "Ang Mo Kio",
    ],
    "Northeast": [
        "Sengkang", "Punggol", "Bishan",
    ],
    "East": [
        "Tampines", "Pasir Ris", "Bedok", "Geylang", "Kallang",
    ],
    "West": [
        "Jurong East", "Jurong West", "Bukit Batok",
        "Choa Chu Kang", "Bukit Panjang", "Clementi",
    ],
    "Central": [
        "Orchard", "Novena", "Toa Payoh", "Marina Bay",
        "Queenstown", "Bukit Merah", "Tanjong Pagar", "Buona Vista",
    ],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _euclidean(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Flat-earth Euclidean distance in degrees (sufficient for SG's scale)."""
    return math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)


def get_nearest_station(town_name: str, stations_df: pd.DataFrame) -> str:
    """
    Return the station_id of the weather station closest to a given town.

    Parameters
    ----------
    town_name   : key from TOWNS dict (case-sensitive)
    stations_df : DataFrame with columns [station_id, latitude, longitude]

    Returns
    -------
    station_id string of the nearest station.

    Raises
    ------
    KeyError  if town_name is not in TOWNS.
    ValueError if stations_df is empty.
    """
    if town_name not in TOWNS:
        raise KeyError(f"Unknown town: '{town_name}'. Valid towns: {sorted(TOWNS)}")
    if stations_df.empty:
        raise ValueError("stations_df is empty — no stations to match against.")

    town_lat, town_lon = TOWNS[town_name]

    distances = stations_df.apply(
        lambda row: _euclidean(town_lat, town_lon,
                               float(row["latitude"]), float(row["longitude"])),
        axis=1,
    )

    nearest_idx = distances.idxmin()
    return str(stations_df.loc[nearest_idx, "station_id"])


def get_town_data(
    town_name: str,
    wide_df: pd.DataFrame,
    stations_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return a one-row DataFrame of environmental metrics for a given town.

    Finds the nearest weather station to the town, filters wide_df for
    that station's region row, and appends 'town' and 'nearest_station'
    columns for traceability.

    Parameters
    ----------
    town_name   : key from TOWNS dict
    wide_df     : wide processed DataFrame from transform.run()
                  (expected columns include 'region' plus metric columns)
    stations_df : DataFrame with columns [station_id, latitude, longitude, region]

    Returns
    -------
    Single-row DataFrame with all wide_df columns plus 'town' and
    'nearest_station'. Returns an empty DataFrame if no match is found.
    """
    nearest_id = get_nearest_station(town_name, stations_df)

    # Map station -> region, then pull that region's row from wide_df
    station_row = stations_df[stations_df["station_id"] == nearest_id]
    if station_row.empty:
        return pd.DataFrame()

    region = str(station_row.iloc[0].get("region", ""))
    region_data = wide_df[wide_df["region"] == region]

    if region_data.empty:
        return pd.DataFrame()

    result = region_data.copy()
    result["town"]            = town_name
    result["nearest_station"] = nearest_id

    return result.reset_index(drop=True)
