"""
Singapore Environmental Intelligence — Streamlit Dashboard
Four screens: home, metrics, recommendations, sources
"""

import subprocess
import sys

import pandas as pd
import requests
import streamlit as st

from src.extract.locations import REGION_TOWNS, get_town_data
from src.extract.weather_stations import _assign_region
from src.recommend.genai import generate_ai_summary

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Singapore Environmental Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ─────────────────────────────────────────────────────────────
for _key, _default in [
    ("screen",          "home"),
    ("region",          None),
    ("expanded_region", None),
    ("selected_town",   None),
    ("selected_region", None),
    ("nearest_station", None),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

# ── Data loading ──────────────────────────────────────────────────────────────

_METRICS_COLS = [
    "region", "retrieved_at", "temperature_c", "humidity_pct", "rainfall_mm",
    "wind_speed_kmh", "forecast_2hr", "wbgt_c", "heat_stress_level",
    "psi_24h", "pm25_sub_index", "uv_index",
]

_RECS_COLS = [
    "region", "status", "status_color",
    "wear_and_bring", "food_and_drinks", "sports_and_activities", "ideal_for",
]


@st.cache_data(ttl=300)
def load_metrics() -> pd.DataFrame:
    try:
        return pd.read_csv("data/processed/processed_env_data.csv")
    except FileNotFoundError:
        return pd.DataFrame(columns=_METRICS_COLS)


@st.cache_data(ttl=300)
def load_recommendations() -> pd.DataFrame:
    try:
        df = pd.read_csv("outputs/recommendations.csv")
    except FileNotFoundError:
        return pd.DataFrame(columns=_RECS_COLS)
    for col in ["wear_and_bring", "food_and_drinks", "sports_and_activities", "ideal_for"]:
        df[col] = df[col].apply(_parse_pipe_list)
    return df


@st.cache_data(ttl=3600)
def load_stations() -> pd.DataFrame:
    try:
        r = requests.get(
            "https://api-open.data.gov.sg/v2/real-time/api/rainfall",
            timeout=10,
        )
        r.raise_for_status()
        rows = [
            {
                "station_id": s["id"],
                "latitude":   float(s["location"]["latitude"]),
                "longitude":  float(s["location"]["longitude"]),
                "region":     _assign_region(
                                  float(s["location"]["latitude"]),
                                  float(s["location"]["longitude"]),
                              ),
            }
            for s in r.json()["data"]["stations"]
        ]
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame(columns=["station_id", "latitude", "longitude", "region"])


def _parse_pipe_list(val) -> list[str]:
    if pd.isna(val) or str(val).strip() in ("", "nan"):
        return []
    return [s.strip() for s in str(val).split("|") if s.strip()]


# ── Colour helpers ────────────────────────────────────────────────────────────

_STATUS_COLORS = {
    "Good":     {"text": "#3B6D11", "bg": "#EAF3DE"},
    "Moderate": {"text": "#854F0B", "bg": "#FAEEDA"},
    "Caution":  {"text": "#A32D2D", "bg": "#FCEBEB"},
    "Avoid":    {"text": "#7A1A1A", "bg": "#FCEBEB"},
}

_HEAT_COLORS = {
    "low":      {"text": "#3B6D11", "bg": "#EAF3DE"},
    "moderate": {"text": "#854F0B", "bg": "#FAEEDA"},
    "high":     {"text": "#A32D2D", "bg": "#FCEBEB"},
}

_INFO = {"text": "#185FA5", "bg": "#E6F1FB"}

_STATUS_PRIORITY = {"Avoid": 4, "Caution": 3, "Moderate": 2, "Good": 1}


def _status_badge(status: str, large: bool = False) -> str:
    c = _STATUS_COLORS.get(status, {"text": "#555", "bg": "#eee"})
    size = "1em" if large else "0.82em"
    pad  = "6px 18px" if large else "3px 14px"
    return (
        f'<span style="background:{c["bg"]};color:{c["text"]};padding:{pad};'
        f'border-radius:12px;font-size:{size};font-weight:700;'
        f'border:1px solid {c["text"]}33">{status}</span>'
    )


def _inline_badge(text: str, colors: dict) -> str:
    return (
        f'<span style="background:{colors["bg"]};color:{colors["text"]};'
        f'padding:2px 10px;border-radius:8px;font-size:0.85em;font-weight:600">{text}</span>'
    )


def _psi_badge(psi) -> str:
    v = float(psi)
    if v >= 200: c = _STATUS_COLORS["Avoid"]
    elif v >= 100: c = _STATUS_COLORS["Caution"]
    elif v >= 55:  c = _STATUS_COLORS["Moderate"]
    else:          c = _STATUS_COLORS["Good"]
    return _inline_badge(f"{v:.0f}", c)


def _uv_badge(uv) -> str:
    v = float(uv)
    if v >= 8:  c = _STATUS_COLORS["Caution"]
    elif v >= 6: c = _STATUS_COLORS["Moderate"]
    else:        c = _STATUS_COLORS["Good"]
    return _inline_badge(f"{v:.0f}", c)


def _heat_badge(heat: str) -> str:
    c = _HEAT_COLORS.get(heat.lower(), {"text": "#555", "bg": "#eee"})
    return _inline_badge(heat.title(), c)


def _dot(text: str) -> str:
    lower = text.lower()
    if any(k in lower for k in ("avoid", "stop", "unhealthy", "extreme", "n95")):
        return "🔴"
    if any(k in lower for k in ("high", "caution", "uv", "heat", "mask", "spf", "seek")):
        return "🟠"
    return "🟢"


def _is_missing(val) -> bool:
    if val is None:
        return True
    try:
        return pd.isna(val)
    except (TypeError, ValueError):
        return str(val).strip() in ("", "nan", "None")


def _fmt_val(val, fmt: str, unit: str) -> str:
    if fmt:
        return f"{float(val):{fmt}}{unit}"
    return f"{val}{unit}"


def _island_wide_status(recs_df: pd.DataFrame) -> str:
    if recs_df.empty or "status" not in recs_df.columns:
        return "Good"
    best = max(recs_df["status"].tolist(), key=lambda s: _STATUS_PRIORITY.get(s, 0))
    return best


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🌿 Singapore EI")
    st.markdown("---")
    if st.button("🔄 Refresh Data", use_container_width=True):
        with st.spinner("Running pipeline..."):
            subprocess.run([sys.executable, "main.py"], check=False)
        load_metrics.clear()
        load_recommendations.clear()
        st.rerun()
    st.caption("Cache auto-refreshes every 5 min.")
    st.markdown("---")
    st.markdown("**Navigate**")
    for _label, _screen in [
        ("🏠 Home",            "home"),
        ("📊 Metrics",         "metrics"),
        ("💡 Recommendations", "recommendations"),
        ("🔗 Sources",         "sources"),
    ]:
        if st.button(_label, use_container_width=True, key=f"nav_{_screen}"):
            st.session_state.screen = _screen
            st.rerun()


# ── Screen: home ──────────────────────────────────────────────────────────────

def show_home():
    recs_df = load_recommendations()
    island_status = _island_wide_status(recs_df)

    col_title, col_badge = st.columns([3, 1])
    with col_title:
        st.markdown("# 🌿 Singapore Environmental Intelligence")
        st.markdown("##### Real-time conditions across Singapore")
    with col_badge:
        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='text-align:right'>"
            f"<div style='color:#888;font-size:0.75em;margin-bottom:4px'>Island-wide status</div>"
            f"{_status_badge(island_status, large=True)}"
            f"</div>",
            unsafe_allow_html=True,
        )

    metrics_df = load_metrics()

    if metrics_df.empty:
        st.markdown("---")
        st.info(
            "No data yet — click **Refresh Data** in the sidebar to fetch live conditions."
        )
        return

    try:
        ts = pd.to_datetime(metrics_df["retrieved_at"].iloc[0])
        st.caption(f"Last updated: {ts.strftime('%d %b %Y, %H:%M')}")
    except Exception:
        pass

    st.markdown("---")
    st.markdown("#### Where are you?")
    st.markdown(
        "<div style='color:#555;font-size:0.9em;margin-bottom:12px'>"
        "Select your region, then choose a town.</div>",
        unsafe_allow_html=True,
    )

    stations_df = load_stations()
    region_names = list(REGION_TOWNS.keys())
    region_cols  = st.columns(len(region_names))

    for i, region in enumerate(region_names):
        with region_cols[i]:
            is_open = st.session_state.expanded_region == region
            border  = "#185FA5" if is_open else "#ddd"
            bg      = _INFO["bg"] if is_open else "#f8f9fa"
            label   = f"▼ {region}" if is_open else f"▶ {region}"
            n_towns = len(REGION_TOWNS[region])

            st.markdown(
                f"""
                <div style="border:2px solid {border};background:{bg};border-radius:10px;
                            padding:10px 8px;text-align:center;margin-bottom:4px">
                    <div style="font-weight:700;font-size:1em;color:#1a1a1a">{region}</div>
                    <div style="color:#888;font-size:0.75em">{n_towns} towns</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(label, key=f"region_{region}", use_container_width=True):
                st.session_state.expanded_region = None if is_open else region
                st.rerun()

    expanded = st.session_state.expanded_region
    if expanded and expanded in REGION_TOWNS:
        towns = REGION_TOWNS[expanded]
        st.markdown(
            f"""
            <div style="border:1px solid #aed6f1;background:#eaf4fb;
                        border-radius:10px;padding:14px 18px;margin-top:10px">
                <div style="color:{_INFO['text']};font-weight:600;margin-bottom:10px">
                    {expanded} — select a town
                </div>
            """,
            unsafe_allow_html=True,
        )
        chip_cols = st.columns(min(len(towns), 4))
        for k, town in enumerate(towns):
            with chip_cols[k % len(chip_cols)]:
                if st.button(town, key=f"town_{town}", use_container_width=True):
                    town_row = get_town_data(town, metrics_df, stations_df)
                    if not town_row.empty:
                        r = town_row.iloc[0]
                        st.session_state.selected_town   = town
                        st.session_state.selected_region = expanded
                        st.session_state.nearest_station = r.get("nearest_station", "")
                        st.session_state.region          = r.get("region", "")
                    else:
                        st.session_state.selected_town   = town
                        st.session_state.selected_region = expanded
                        st.session_state.nearest_station = ""
                        st.session_state.region          = ""
                    st.session_state.screen = "metrics"
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔗 View data sources", use_container_width=False):
        st.session_state.screen = "sources"
        st.rerun()


# ── Screen: metrics ───────────────────────────────────────────────────────────

def show_metrics():
    town            = st.session_state.selected_town   or ""
    display_region  = st.session_state.selected_region or ""
    nearest_station = st.session_state.nearest_station or ""
    pipeline_region = st.session_state.region          or "central"

    metrics_df = load_metrics()
    recs_df    = load_recommendations()

    if st.button("← Back to Home"):
        st.session_state.screen = "home"
        st.rerun()

    m_row = metrics_df[metrics_df["region"] == pipeline_region]
    if m_row.empty:
        st.error(f"No data available for {town or pipeline_region}.")
        return
    m = m_row.iloc[0]

    r_row  = recs_df[recs_df["region"] == pipeline_region]
    status = r_row.iloc[0]["status"] if not r_row.empty else "Good"

    st.markdown(
        f"## {town or pipeline_region.title()} &nbsp; {_status_badge(status)}",
        unsafe_allow_html=True,
    )
    if town:
        st.markdown(
            f"**{display_region}** &nbsp;•&nbsp; Nearest station: `{nearest_station}`"
        )

    forecast = m.get("forecast_2hr") or "N/A"
    heat     = str(m.get("heat_stress_level") or "").lower()
    heat_c   = _HEAT_COLORS.get(heat, {"bg": "#eee", "text": "#555"})

    st.markdown(
        f"""
        <div style="background:{heat_c['bg']};border-left:5px solid {heat_c['text']};
                    border-radius:8px;padding:14px 18px;margin:14px 0">
            <b>2-Hour Forecast:</b> {forecast} &nbsp;&nbsp;
            <b>Heat Stress:</b>
            <span style="color:{heat_c['text']};font-weight:700">{heat.title() or 'N/A'}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("#### Current Readings")

    metric_defs = [
        ("Temperature",  "temperature_c",    ".1f", "°C",    None),
        ("Humidity",     "humidity_pct",      ".0f", "%",     None),
        ("Wind Speed",   "wind_speed_kmh",    ".1f", " km/h", None),
        ("PSI (24h)",    "psi_24h",           None,  "",      "psi"),
        ("UV Index",     "uv_index",          None,  "",      "uv"),
        ("Rainfall",     "rainfall_mm",       ".1f", " mm",   None),
        ("WBGT",         "wbgt_c",            ".1f", "°C",    None),
        ("Heat Stress",  "heat_stress_level", None,  "",      "heat"),
        ("PM2.5",        "pm25_sub_index",    ".0f", "",      None),
    ]

    for row_start in range(0, len(metric_defs), 3):
        cols = st.columns(3)
        for j, (label, col, fmt, unit, badge_type) in enumerate(
            metric_defs[row_start : row_start + 3]
        ):
            with cols[j]:
                raw   = m.get(col) if col in m.index else None
                empty = _is_missing(raw)

                if empty:
                    value_html = '<span style="color:#aaa;font-size:1.5em">N/A</span>'
                elif badge_type == "psi":
                    value_html = _psi_badge(raw)
                elif badge_type == "uv":
                    value_html = _uv_badge(raw)
                elif badge_type == "heat":
                    value_html = _heat_badge(str(raw))
                else:
                    value_html = (
                        f'<span style="color:#2c3e50;font-size:1.6em;font-weight:700">'
                        f'{_fmt_val(raw, fmt, unit)}</span>'
                    )

                st.markdown(
                    f"""
                    <div style="border:1px solid #e0e0e0;border-radius:10px;
                                padding:16px;text-align:center;min-height:90px">
                        <div style="color:#888;font-size:0.78em;margin-bottom:8px">{label}</div>
                        <div>{value_html}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    if st.button("Get recommendations →", use_container_width=True, type="primary"):
        st.session_state.screen = "recommendations"
        st.rerun()


# ── Screen: recommendations ───────────────────────────────────────────────────

def show_recommendations():
    region = st.session_state.region or "central"
    town   = st.session_state.selected_town or region.title()

    if st.button("← Back to Metrics"):
        st.session_state.screen = "metrics"
        st.rerun()

    recs_df = load_recommendations()
    r_row   = recs_df[recs_df["region"] == region]
    if r_row.empty:
        st.warning("No recommendations available for this region.")
        return

    r      = r_row.iloc[0]
    status = r.get("status", "Good")

    st.markdown(
        f"## 💡 {town} &nbsp; {_status_badge(status)}",
        unsafe_allow_html=True,
    )

    # ── AI Summary ────────────────────────────────────────────────────────────
    metrics_df   = load_metrics()
    m_row        = metrics_df[metrics_df["region"] == region]
    metrics_dict = m_row.iloc[0].to_dict() if not m_row.empty else {}
    metrics_dict["town"] = town

    with st.spinner("Generating AI summary..."):
        summary = generate_ai_summary(metrics_dict)

    info_c = _INFO
    st.markdown(
        f"""
        <div style="background:{info_c['bg']};border-left:5px solid {info_c['text']};
                    border-radius:8px;padding:16px 20px;margin:14px 0">
            <div style="color:{info_c['text']};font-weight:600;margin-bottom:6px">
                AI Environmental Briefing
            </div>
            <div style="color:#1a1a1a;line-height:1.6">{summary}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Powered by GPT-4o-mini  •  Conditions updated every 5 mins")

    st.markdown("---")

    sections = [
        ("👕 What to wear & bring", "wear_and_bring"),
        ("🍹 Food & drinks",         "food_and_drinks"),
        ("🏃 Sports & activities",   "sports_and_activities"),
        ("✨ Ideal for today",        "ideal_for"),
    ]

    for section_label, col in sections:
        items = r.get(col, [])
        if not isinstance(items, list):
            items = []
        with st.expander(section_label, expanded=True):
            if not items:
                st.caption("No specific recommendations for current conditions.")
            else:
                for item in items:
                    st.markdown(f"{_dot(item)} {item}")


# ── Screen: sources ───────────────────────────────────────────────────────────

_API_SOURCES = [
    {
        "icon": "🌧️",
        "name": "Rainfall",
        "description": "Real-time rainfall readings from island-wide stations.",
        "endpoint": "https://api-open.data.gov.sg/v2/real-time/api/rainfall",
        "metric": "rainfall_mm",
    },
    {
        "icon": "🌡️",
        "name": "Air Temperature",
        "description": "Per-station ambient air temperature readings.",
        "endpoint": "https://api-open.data.gov.sg/v2/real-time/api/air-temperature",
        "metric": "temperature_c",
    },
    {
        "icon": "💧",
        "name": "Relative Humidity",
        "description": "Relative humidity percentage from weather stations.",
        "endpoint": "https://api-open.data.gov.sg/v2/real-time/api/relative-humidity",
        "metric": "humidity_pct",
    },
    {
        "icon": "💨",
        "name": "Wind Speed",
        "description": "10-minute average wind speed from NEA stations.",
        "endpoint": "https://api-open.data.gov.sg/v2/real-time/api/wind-speed",
        "metric": "wind_speed_kmh",
    },
    {
        "icon": "🌤️",
        "name": "2-Hour Forecast",
        "description": "Short-term weather forecasts per planning area.",
        "endpoint": "https://api-open.data.gov.sg/v2/real-time/api/two-hr-forecast",
        "metric": "forecast_2hr",
    },
    {
        "icon": "🔥",
        "name": "WBGT & Heat Stress",
        "description": "Wet Bulb Globe Temperature and heat stress level from NEA.",
        "endpoint": "https://api-open.data.gov.sg/v2/real-time/api/weather?api=wbgt",
        "metric": "wbgt_c / heat_stress_level",
    },
    {
        "icon": "🌫️",
        "name": "PSI (24-Hour)",
        "description": "Pollutant Standards Index broken down by region.",
        "endpoint": "https://api-open.data.gov.sg/v2/real-time/api/psi",
        "metric": "psi_24h / pm25_sub_index",
    },
    {
        "icon": "☀️",
        "name": "UV Index",
        "description": "Island-wide ultraviolet index from NEA.",
        "endpoint": "https://api-open.data.gov.sg/v2/real-time/api/uv",
        "metric": "uv_index",
    },
    {
        "icon": "🧭",
        "name": "Wind Direction",
        "description": "10-minute average wind direction from NEA stations.",
        "endpoint": "https://api-open.data.gov.sg/v2/real-time/api/wind-direction",
        "metric": "wind_direction",
    },
]


def show_sources():
    if st.button("← Back to Home"):
        st.session_state.screen = "home"
        st.rerun()

    st.markdown("# 🔗 Data Sources")
    st.markdown(
        "All environmental data is sourced in real-time from the "
        "[data.gov.sg](https://data.gov.sg) open data platform, provided by the "
        "National Environment Agency (NEA) of Singapore."
    )
    st.markdown("---")

    for i in range(0, len(_API_SOURCES), 3):
        cols = st.columns(3)
        for j, src in enumerate(_API_SOURCES[i : i + 3]):
            with cols[j]:
                st.markdown(
                    f"""
                    <div style="border:1px solid #dce6f0;border-radius:12px;
                                padding:18px;margin-bottom:16px;min-height:180px;
                                background:#fafcff">
                        <div style="font-size:1.8em">{src['icon']}</div>
                        <div style="font-weight:700;font-size:1em;margin:8px 0 4px">
                            {src['name']}
                        </div>
                        <div style="color:#555;font-size:0.84em;margin-bottom:10px;
                                    line-height:1.4">
                            {src['description']}
                        </div>
                        <div style="color:#888;font-size:0.72em;margin-bottom:4px">
                            Metric: <code>{src['metric']}</code>
                        </div>
                        <div style="word-break:break-all;font-size:0.70em;
                                    color:{_INFO['text']};background:{_INFO['bg']};
                                    border-radius:6px;padding:4px 8px;margin-top:6px">
                            {src['endpoint']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    st.markdown(
        "<div style='color:#888;font-size:0.82em;text-align:center'>"
        "Data refreshed every 5 minutes  •  "
        "Source: <a href='https://data.gov.sg' target='_blank'>data.gov.sg</a>  •  "
        "Built with Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Router ────────────────────────────────────────────────────────────────────

_screen = st.session_state.screen

if _screen == "home":
    show_home()
elif _screen == "metrics":
    if st.session_state.selected_town:
        show_metrics()
    else:
        st.session_state.screen = "home"
        st.rerun()
elif _screen == "recommendations":
    if st.session_state.selected_town:
        show_recommendations()
    else:
        st.session_state.screen = "home"
        st.rerun()
elif _screen == "sources":
    show_sources()
else:
    st.session_state.screen = "home"
    st.rerun()
