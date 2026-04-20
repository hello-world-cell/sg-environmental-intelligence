"""
Recommend module — generates unified per-region environmental recommendations
across 4 categories plus a status badge.
"""

import pandas as pd

_OUTPUT_PATH = "outputs/recommendations.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(row: pd.Series, col: str):
    """Return row[col] if the column exists and is not NaN, else None."""
    if col not in row.index:
        return None
    val = row[col]
    return None if pd.isna(val) else val


def _heat(row: pd.Series) -> str | None:
    """Return normalised heat_stress_level string or None."""
    h = _get(row, "heat_stress_level")
    return str(h).lower() if h is not None else None


# ---------------------------------------------------------------------------
# Unified recommendation categories
# ---------------------------------------------------------------------------

def _wear_and_bring(row: pd.Series) -> list[str]:
    tips = []
    uv   = _get(row, "uv_index")
    rain = _get(row, "rainfall_mm")
    psi  = _get(row, "psi_24h")
    temp = _get(row, "temperature_c")
    heat = _heat(row)

    if uv is not None:
        if uv >= 8:
            tips.append("Wear a hat and UV-protective clothing; seek shade if UV stays high")
        elif uv >= 6:
            tips.append("Apply SPF 50+ sunscreen and reapply every 2 hours outdoors")
    if rain is not None and rain > 0:
        tips.append("Bring an umbrella - rain is detected; take shelter if caught outside")
    if psi is not None:
        if psi >= 200:
            tips.append("Wear an N95 mask - air quality is unhealthy")
        elif psi >= 100:
            tips.append("Wear a mask outdoors - air quality is moderate")
    if temp is not None and temp >= 32:
        tips.append("Wear light, breathable clothing")
    if heat == "high":
        tips.append("Choose moisture-wicking fabric to manage sweat")
    return tips


def _food_and_drinks(row: pd.Series) -> list[str]:
    tips = []
    temp = _get(row, "temperature_c")
    rain = _get(row, "rainfall_mm")
    heat = _heat(row)

    if (temp is not None and temp >= 33) or heat in ("moderate", "high"):
        tips.append("Drink water regularly - at least 250ml every 30 mins outdoors")
    if heat == "high":
        tips.append("Avoid heavy meals; find a cool shaded spot to eat")
    if rain is not None and rain > 0:
        tips.append("Good day for a warm meal at a covered hawker centre")
    if temp is not None and temp <= 28:
        tips.append("Cooler weather - comfortable for heavier meals")
    return tips


def _sports_and_activities(row: pd.Series) -> list[str]:
    tips = []
    uv   = _get(row, "uv_index")
    rain = _get(row, "rainfall_mm")
    heat = _heat(row)

    if heat == "high":
        tips.append("Avoid strenuous outdoor activity - rest in shade or air-conditioned spaces")
    elif heat == "moderate":
        tips.append("Take a 10-min break every 30 mins of outdoor activity")
    if uv is not None and uv >= 8:
        tips.append("Exercise before 9am or after 6pm; move to shaded areas if already outside")
    if rain is not None and rain > 0:
        tips.append("Outdoor courts may be wet - check conditions or wait for rain to ease")
    if heat == "low" and (uv is None or uv < 6):
        tips.append("Great conditions for outdoor exercise!")
    return tips


def _ideal_for(row: pd.Series) -> list[str]:
    tips = []
    uv   = _get(row, "uv_index")
    rain = _get(row, "rainfall_mm")
    psi  = _get(row, "psi_24h")
    heat = _heat(row)

    if heat == "high" or (psi is not None and psi >= 100):
        tips.append("Indoor activities - malls, museums, or air-conditioned spaces")
    if rain is not None and rain > 0:
        tips.append("Covered hawker centres or indoor cafes")
    if heat == "low" and (rain is None or rain == 0) and (uv is None or uv < 6):
        tips.append("Conditions are good - parks, cycling, or outdoor dining")
    if uv is not None and uv < 3:
        tips.append("Evening walks or outdoor dining")
    return tips


# ---------------------------------------------------------------------------
# Status badge
# ---------------------------------------------------------------------------

def _compute_status(row: pd.Series) -> tuple[str, str]:
    """Return (status_label, color) based on combined environmental score."""
    score = 0
    psi  = _get(row, "psi_24h")
    uv   = _get(row, "uv_index")
    rain = _get(row, "rainfall_mm")
    heat = _heat(row)

    if psi is not None:
        if psi >= 200:   score += 3
        elif psi >= 100: score += 2
        elif psi >= 55:  score += 1

    if uv is not None:
        if uv >= 11:    score += 3
        elif uv >= 8:   score += 2
        elif uv >= 6:   score += 1

    if heat == "high":     score += 3
    elif heat == "moderate": score += 2
    elif heat == "low":    score += 1

    if rain is not None:
        if rain > 5:   score += 2
        elif rain > 0: score += 1

    if score >= 9: return "Avoid",    "red"
    if score >= 6: return "Caution",  "orange"
    if score >= 3: return "Moderate", "amber"
    return "Good", "green"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(processed_data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate unified per-region recommendations across 4 categories.

    Parameters
    ----------
    processed_data : wide DataFrame from transform.run(), one row per region.

    Returns
    -------
    DataFrame with columns:
        region, status, status_color,
        wear_and_bring, food_and_drinks, sports_and_activities, ideal_for
    Each recommendation column holds a list of strings.
    Saved to outputs/recommendations.csv with lists pipe-joined.
    """
    print("[recommend] Generating recommendations...")
    print()

    if processed_data.empty:
        print("[recommend] Warning: received empty DataFrame.")
        return pd.DataFrame(columns=[
            "region", "status", "status_color",
            "wear_and_bring", "food_and_drinks", "sports_and_activities", "ideal_for",
        ])

    rows = []
    for _, row in processed_data.iterrows():
        region = row.get("region", "unknown")

        wear   = _wear_and_bring(row)
        food   = _food_and_drinks(row)
        sports = _sports_and_activities(row)
        ideal  = _ideal_for(row)
        status, color = _compute_status(row)

        n_total = len(wear) + len(food) + len(sports) + len(ideal)
        print(f"[recommend] {region:<12} ({status}) -> {n_total} recommendations")

        rows.append({
            "region":               region,
            "status":               status,
            "status_color":         color,
            "wear_and_bring":       wear,
            "food_and_drinks":      food,
            "sports_and_activities": sports,
            "ideal_for":            ideal,
        })

    results_df = pd.DataFrame(rows)

    csv_df = results_df.copy()
    list_cols = [c for c in csv_df.columns if c not in ("region", "status", "status_color")]
    for col in list_cols:
        csv_df[col] = csv_df[col].apply(lambda lst: " | ".join(lst) if lst else "")

    csv_df.to_csv(_OUTPUT_PATH, index=False)
    print(f"\n[recommend] Saved to {_OUTPUT_PATH}")

    return results_df
