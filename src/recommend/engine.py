"""
Recommendation engine for the Singapore Environmental Intelligence Pipeline.

Generates per-region advisory text from the curated metrics DataFrame.
Heat stress recommendations use the official NEA WBGT heat_stress_level
as the authoritative signal — no manual derivation from raw temperature.
"""

import pandas as pd

# ---------------------------------------------------------------------------
# WBGT heat stress advisories (NEA standard)
# ---------------------------------------------------------------------------

_HEAT_STRESS_ADVISORY: dict[str, str | None] = {
    "low":      None,   # no warning needed
    "moderate": (
        "Moderate heat stress — take breaks every 45 mins "
        "if working outdoors."
    ),
    "high": (
        "High heat stress — NEA advisory in effect. "
        "Minimise strenuous outdoor activity. Stay hydrated."
    ),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lookup(df: pd.DataFrame, region: str, metric: str):
    """
    Return the scalar value for a given region + metric combination,
    or None if the metric is absent for that region.
    """
    mask = (df["region"] == region) & (df["metric"] == metric)
    rows = df.loc[mask, "value"]
    return rows.iloc[0] if not rows.empty else None


def _heat_stress_advisory(level: str | None) -> str | None:
    """
    Map an official WBGT heat_stress_level string to advisory text.

    Returns None for "low" (no action needed) or any unrecognised level.
    Unrecognised levels are treated conservatively as no-warning rather
    than raising, so a new API category never crashes the pipeline.
    """
    if level is None:
        return None
    return _HEAT_STRESS_ADVISORY.get(str(level).lower())


def _rainfall_advisory(rainfall_mm) -> str | None:
    """Return a rainfall advisory if accumulation is significant."""
    if rainfall_mm is None:
        return None
    if float(rainfall_mm) >= 10:
        return f"Heavy rainfall ({rainfall_mm:.1f} mm) — expect flash-flood risk in low-lying areas."
    if float(rainfall_mm) >= 2:
        return f"Light rain ({rainfall_mm:.1f} mm) — carry an umbrella."
    return None


def _forecast_advisory(forecast: str | None) -> str | None:
    """Surface the 2-hour forecast text as an advisory when severe."""
    if forecast is None:
        return None
    lower = forecast.lower()
    if "thunder" in lower:
        return f"2-hr forecast: {forecast}. Avoid open areas and tall structures."
    if "heavy" in lower:
        return f"2-hr forecast: {forecast}. Outdoor activities not recommended."
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(curated_df: pd.DataFrame) -> list[dict]:
    """
    Generate per-region environmental recommendations from curated metrics.

    The heat stress advisory is driven exclusively by the official NEA WBGT
    heat_stress_level — not derived from raw temperature values.

    Parameters
    ----------
    curated_df : DataFrame with columns [region, timestamp, metric, value]
                 as produced by the transform stage.

    Returns
    -------
    List of dicts, one per region, each with keys:
        region, timestamp, advisories (list of advisory strings)
    """
    if curated_df.empty:
        print("[engine] No curated data to generate recommendations from.")
        return []

    regions    = curated_df["region"].unique()
    timestamp  = curated_df["timestamp"].iloc[0]
    results    = []

    for region in sorted(regions):
        advisories = []

        # --- Heat stress: official WBGT level (NEA standard) ----------------
        heat_level = _lookup(curated_df, region, "heat_stress_level")
        advisory   = _heat_stress_advisory(heat_level)
        if advisory:
            advisories.append(advisory)

        # --- Rainfall accumulation ------------------------------------------
        rainfall   = _lookup(curated_df, region, "rainfall_mm")
        advisory   = _rainfall_advisory(rainfall)
        if advisory:
            advisories.append(advisory)

        # --- 2-hour forecast (qualitative) ----------------------------------
        forecast   = _lookup(curated_df, region, "forecast_2hr")
        advisory   = _forecast_advisory(forecast)
        if advisory:
            advisories.append(advisory)

        results.append({
            "region":     region,
            "timestamp":  timestamp,
            "advisories": advisories,
        })

        status = f"{len(advisories)} advisory/advisories" if advisories else "all clear"
        print(f"[engine] {region:<10} heat_stress={heat_level or 'n/a':<10} → {status}")

    return results
