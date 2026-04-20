"""
Fetches the 2-hour area forecast from data.gov.sg and returns a
region-aggregated DataFrame with qualitative forecast strings.
"""

import requests
import pandas as pd

_ENDPOINT = "https://api-open.data.gov.sg/v2/real-time/api/two-hr-forecast"

# ---------------------------------------------------------------------------
# Area → region lookup
# Each planning area name as returned by the API is mapped to one of the
# five canonical regions. "central" is the catch-all fallback.
# ---------------------------------------------------------------------------

_AREA_REGION: dict[str, str] = {
    # North
    "Lim Chu Kang":              "north",
    "Mandai":                    "north",
    "Seletar":                   "north",
    "Sembawang":                 "north",
    "Sengkang":                  "north",
    "Simpang":                   "north",
    "Sungei Kadut":              "north",
    "Woodlands":                 "north",
    "Yishun":                    "north",
    # South
    "Sentosa":                   "south",
    "Southern Islands":          "south",
    "Bukit Merah":               "south",
    "Telok Blangah":             "south",
    "Harbourfront":              "south",
    # East
    "Bedok":                     "east",
    "Changi":                    "east",
    "Geylang":                   "east",
    "Hougang":                   "east",
    "Marine Parade":             "east",
    "Pasir Ris":                 "east",
    "Paya Lebar":                "east",
    "Pulau Tekong":              "east",
    "Pulau Ubin":                "east",
    "Simei":                     "east",
    "Tampines":                  "east",
    # West
    "Boon Lay":                  "west",
    "Bukit Batok":               "west",
    "Bukit Panjang":             "west",
    "Choa Chu Kang":             "west",
    "Clementi":                  "west",
    "Jalan Bahar":               "west",
    "Jurong East":               "west",
    "Jurong Island":             "west",
    "Jurong West":               "west",
    "Pioneer":                   "west",
    "Tengah":                    "west",
    "Tuas":                      "west",
    "Western Water Catchment":   "west",
    # Central
    "Ang Mo Kio":                "central",
    "Bishan":                    "central",
    "Bukit Timah":               "central",
    "Central Water Catchment":   "central",
    "City":                      "central",
    "Downtown Core":             "central",
    "Kallang":                   "central",
    "Marina Bay":                "central",
    "Marina South":              "central",
    "Museum":                    "central",
    "Newton":                    "central",
    "Novena":                    "central",
    "Orchard":                   "central",
    "Outram":                    "central",
    "Queenstown":                "central",
    "Rochor":                    "central",
    "Serangoon":                 "central",
    "Singapore River":           "central",
    "Tanglin":                   "central",
    "Thomson":                   "central",
    "Toa Payoh":                 "central",
}

# ---------------------------------------------------------------------------
# Keyword fallback for area names not in the explicit lookup
# Checked in order; first match wins; "central" is the final default.
# ---------------------------------------------------------------------------

_KEYWORD_RULES: list[tuple[str, str]] = [
    # north keywords
    ("woodlands",  "north"),
    ("sembawang",  "north"),
    ("yishun",     "north"),
    ("mandai",     "north"),
    ("sengkang",   "north"),
    ("seletar",    "north"),
    # south keywords
    ("sentosa",    "south"),
    ("southern",   "south"),
    ("harbourfront","south"),
    # east keywords
    ("tampines",   "east"),
    ("bedok",      "east"),
    ("changi",     "east"),
    ("pasir ris",  "east"),
    ("hougang",    "east"),
    ("geylang",    "east"),
    # west keywords
    ("jurong",     "west"),
    ("choa chu kang", "west"),
    ("bukit batok","west"),
    ("clementi",   "west"),
    ("bukit panjang","west"),
    ("tengah",     "west"),
    ("tuas",       "west"),
]


def _area_to_region(area: str) -> str:
    """
    Resolve an area name to one of the five Singapore regions.

    Strategy:
    1. Exact match in the lookup table (fastest, most precise).
    2. Case-insensitive keyword scan for names not in the table.
    3. Default to "central".
    """
    region = _AREA_REGION.get(area)
    if region:
        return region

    lower = area.lower()
    for keyword, mapped_region in _KEYWORD_RULES:
        if keyword in lower:
            return mapped_region

    return "central"


def _plurality_forecast(forecasts: list[str]) -> str:
    """Return the most common forecast string in a list (plurality vote)."""
    return pd.Series(forecasts).mode().iloc[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_forecast() -> pd.DataFrame:
    """
    Fetch the current 2-hour area forecast and return one row per region.

    The API delivers one forecast string per planning area. Areas are mapped
    to regions, and the plurality (most frequent) forecast string is chosen
    as the representative value for each region.

    Returns
    -------
    DataFrame with columns: region, timestamp, metric ('forecast_2hr'), value
        value holds a qualitative string e.g. 'Partly Cloudy'.
    """
    response = requests.get(_ENDPOINT, timeout=10)
    response.raise_for_status()
    payload = response.json()

    # The two-hr-forecast v2 endpoint returns items; take the latest.
    items = payload["data"]["items"]
    latest = items[-1]
    timestamp = latest["timestamp"]

    rows = []
    for entry in latest["forecasts"]:
        area = entry["area"]
        forecast_text = entry["forecast"]
        region = _area_to_region(area)
        rows.append({"region": region, "forecast": forecast_text})

    if not rows:
        return pd.DataFrame(columns=["region", "timestamp", "metric", "value"])

    df = pd.DataFrame(rows)
    agg = (
        df.groupby("region")["forecast"]
        .agg(_plurality_forecast)
        .reset_index()
        .rename(columns={"forecast": "value"})
    )
    agg["timestamp"] = timestamp
    agg["metric"] = "forecast_2hr"

    return agg[["region", "timestamp", "metric", "value"]]
