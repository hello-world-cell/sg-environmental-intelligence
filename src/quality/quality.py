"""
Quality module — validates the wide-format processed DataFrame produced
by transform.py for missing values and out-of-range readings.
"""

import json
import pandas as pd

_REPORT_PATH = "data/processed/quality_report.json"

# Expected valid ranges for each numeric metric (inclusive)
_RANGES: dict[str, tuple[float, float]] = {
    "psi_24h":        (0,   500),
    "uv_index":       (0,    15),
    "rainfall_mm":    (0,   300),
    "temperature_c":  (15,   45),
    "humidity_pct":   (0,   100),
    "wind_speed_kmh": (0,   200),
    "wbgt_c":         (10,   40),
}


def run(processed_data: pd.DataFrame) -> dict:
    """
    Run data quality checks on the wide-format processed DataFrame.

    Checks
    ------
    1. Missing values  — reports null count per column.
    2. Range checks    — flags values outside the expected bounds for each
                         known numeric metric (only if the column exists).

    Parameters
    ----------
    processed_data : wide DataFrame from transform.run()

    Returns
    -------
    quality_report dict with keys:
        total_rows   : int
        null_counts  : {column: null_count}
        range_checks : {metric: {"valid": int, "total": int, "passed": bool}}
    """
    print("[quality] Running data quality checks...")
    print()

    if processed_data.empty:
        print("[quality] Warning: received empty DataFrame — no checks run.")
        return {"total_rows": 0, "null_counts": {}, "range_checks": {}}

    total_rows = len(processed_data)

    # ------------------------------------------------------------------
    # 1. Missing values
    # ------------------------------------------------------------------
    print("[quality] Null check:")
    null_counts = {}
    for col in processed_data.columns:
        n_null = int(processed_data[col].isna().sum())
        null_counts[col] = n_null
        if n_null:
            print(f"[quality]   {col:<20} -> {n_null} null(s) of {total_rows}")

    if not any(null_counts.values()):
        print("[quality]   No nulls found.")
    print()

    # ------------------------------------------------------------------
    # 2. Range checks (only for columns that exist in the DataFrame)
    # ------------------------------------------------------------------
    print("[quality] Range checks:")
    range_checks = {}
    passed = 0
    total_checks = 0

    for metric, (lo, hi) in _RANGES.items():
        if metric not in processed_data.columns:
            continue

        col = processed_data[metric].dropna()
        n_total = len(col)
        n_valid = int(col.between(lo, hi).sum())
        did_pass = n_valid == n_total

        range_checks[metric] = {
            "valid":  n_valid,
            "total":  n_total,
            "passed": did_pass,
        }

        status = "PASS" if did_pass else "FAIL"
        print(f"[quality]   {metric:<20} -> {status} ({n_valid}/{n_total} valid)  [{lo}-{hi}]")

        total_checks += 1
        if did_pass:
            passed += 1

    print()
    print(f"[quality] {passed}/{total_checks} checks passed.")

    # ------------------------------------------------------------------
    # 3. Build and save quality report
    # ------------------------------------------------------------------
    quality_report = {
        "total_rows":   total_rows,
        "null_counts":  null_counts,
        "range_checks": range_checks,
    }

    with open(_REPORT_PATH, "w") as f:
        json.dump(quality_report, f, indent=2, default=str)

    print(f"[quality] Report saved to {_REPORT_PATH}")

    return quality_report
