"""
AI summary generation for the Singapore Environmental Intelligence Pipeline.

Uses GPT-4o-mini to produce a short, conversational environmental briefing
based on current real-time conditions.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()

try:
    import streamlit as st
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
except Exception:
    api_key = os.getenv("OPENAI_API_KEY")

_client = OpenAI(api_key=api_key)

_SYSTEM_PROMPT = (
    "You are a friendly Singapore weather assistant giving "
    "practical real-time advice. Be conversational, specific, "
    "and direct. Reference actual metric values in your response. "
    "Keep it to 3-4 sentences."
)

_FALLBACK = "Unable to generate AI summary. See recommendations below."


def generate_ai_summary(metrics_dict: dict) -> str:
    """
    Generate a concise AI weather briefing for the selected town.

    Parameters
    ----------
    metrics_dict : dict of metric values for the town, expected keys:
                   town, temperature_c, humidity_pct, psi_24h, pm25_sub_index,
                   uv_index, rainfall_mm, heat_stress_level, wbgt_c, forecast_2hr

    Returns
    -------
    AI-generated summary string (3-4 sentences), or the fallback string
    if the API call fails.
    """
    def _get(key, fallback="N/A"):
        val = metrics_dict.get(key, fallback)
        return fallback if val is None or str(val).strip() in ("", "nan") else val

    town         = _get("town",              "your area")
    temp         = _get("temperature_c",     "N/A")
    humidity     = _get("humidity_pct",      "N/A")
    psi          = _get("psi_24h",           "N/A")
    pm25         = _get("pm25_sub_index",    "N/A")
    uv           = _get("uv_index",          "N/A")
    rainfall     = _get("rainfall_mm",       "N/A")
    heat_stress  = _get("heat_stress_level", "N/A")
    wbgt         = _get("wbgt_c",            "N/A")
    forecast     = _get("forecast_2hr",      "N/A")

    user_prompt = (
        f"Current conditions in {town}:\n"
        f"- Temperature: {temp}°C, Humidity: {humidity}%\n"
        f"- PSI: {psi}, PM2.5: {pm25}\n"
        f"- UV Index: {uv}/15\n"
        f"- Rainfall: {rainfall}mm\n"
        f"- WBGT Heat Stress: {heat_stress} ({wbgt}°C)\n"
        f"- 2hr Forecast: {forecast}\n\n"
        f"Give a friendly 3-4 sentence summary covering: "
        f"what the weather feels like right now, the biggest "
        f"risks to be aware of, and one practical action to take."
    )

    try:
        response = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=150,
        )
        return response.choices[0].message.content.strip()

    except OpenAIError as exc:
        print(f"[genai] OpenAI API error: {exc}")
        return _FALLBACK

    except Exception as exc:
        print(f"[genai] Unexpected error: {exc}")
        return _FALLBACK
