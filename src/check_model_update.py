"""
Check if a new weather model run is available on Open-Meteo.

Queries the Open-Meteo Metadata API and compares the availability time
against a locally stored state file (.last_model_update).

Returns exit code 0 (= update available) when:
  - A new model run is detected (availability_time changed), OR
  - The last update is still "fresh" (within propagation window of availability_time),
    because Open-Meteo servers need time to fully propagate data.

Exit codes:
    0 = model update available (new or still within propagation window)
    1 = no recent update (data fully propagated, nothing new)
    2 = error (API unreachable, etc.)

Usage:
    python -m src.check_model_update [model_name]

    model_name: Open-Meteo model identifier (default from settings.toml or dwd_icon_d2_15min)
                Examples: dwd_icon_d2_15min, dwd_icon_d2, dwd_icon_eu, dwd_icon
"""

import sys
import os
import json
import requests
from datetime import datetime, timezone, timedelta

DEFAULT_MODEL = "dwd_icon_d2_15min"
DEFAULT_PROPAGATION_MINUTES = 15
STATE_FILE = ".last_model_update"
META_URL_TEMPLATE = "https://api.open-meteo.com/data/{model}/static/meta.json"


def load_last_state(state_path):
    """Load the last known availability time from the state file."""
    if not os.path.exists(state_path):
        return None
    try:
        with open(state_path, "r") as f:
            data = json.load(f)
            return data.get("last_run_availability_time")
    except Exception:
        return None


def save_state(state_path, availability_time, model):
    """Save the current availability time to the state file."""
    with open(state_path, "w") as f:
        json.dump({
            "model": model,
            "last_run_availability_time": availability_time,
            "checked_at": datetime.now(timezone.utc).isoformat()
        }, f, indent=2)


def ts_to_str(unix_ts):
    """Convert a Unix timestamp to a human-readable UTC string."""
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def check_update(model=None, propagation_minutes=None):
    """
    Check if a weather model update is available on Open-Meteo.

    Args:
        model: Open-Meteo model identifier (e.g. 'dwd_icon_d2_15min').
               Defaults to DEFAULT_MODEL if None.
        propagation_minutes: Minutes to keep signalling "update available"
               after availability_time. Defaults to DEFAULT_PROPAGATION_MINUTES.

    Returns:
        dict with keys:
            is_update (bool): True if an update is available or still fresh
            exit_code (int): 0 = update, 1 = no update, 2 = error
            message (str): Human-readable status message
            meta (dict|None): Raw metadata from API (if successful)
    """
    model = model or DEFAULT_MODEL
    propagation_minutes = propagation_minutes if propagation_minutes is not None else DEFAULT_PROPAGATION_MINUTES
    url = META_URL_TEMPLATE.format(model=model)

    lines = []
    lines.append("================================================================")
    lines.append(f"  Open-Meteo Model Update Check  —  {model}")
    lines.append("================================================================")
    lines.append(f"  API: {url}")
    lines.append("")

    # --- Fetch metadata ---
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        meta = resp.json()
    except requests.exceptions.HTTPError as e:
        msg = f"ERROR: HTTP {resp.status_code} — {e}"
        if resp.status_code == 404:
            msg += f"\n  -> Model '{model}' not found. Check the model name."
        lines.append(msg)
        return {"is_update": False, "exit_code": 2, "message": "\n".join(lines), "meta": None}
    except Exception as e:
        lines.append(f"ERROR: Could not reach Open-Meteo API — {e}")
        return {"is_update": False, "exit_code": 2, "message": "\n".join(lines), "meta": None}

    # --- Extract fields ---
    init_time = meta.get("last_run_initialisation_time")
    mod_time = meta.get("last_run_modification_time")
    avail_time = meta.get("last_run_availability_time")
    update_interval = meta.get("update_interval_seconds")
    temporal_res = meta.get("temporal_resolution_seconds")

    if avail_time is None:
        lines.append("ERROR: 'last_run_availability_time' not found in API response.")
        return {"is_update": False, "exit_code": 2, "message": "\n".join(lines), "meta": meta}

    # --- Display current status ---
    now_utc = datetime.now(timezone.utc)
    avail_dt = datetime.fromtimestamp(avail_time, tz=timezone.utc)
    age = now_utc - avail_dt
    age_minutes = age.total_seconds() / 60

    lines.append(f"  Model Initialisation : {ts_to_str(init_time)}")
    lines.append(f"  Data Modified        : {ts_to_str(mod_time)}")
    lines.append(f"  Data Available (API) : {ts_to_str(avail_time)}")
    lines.append(f"  Data Age             : {age_minutes:.1f} min")
    if update_interval:
        lines.append(f"  Update Interval      : {update_interval // 3600}h ({update_interval}s)")
    if temporal_res:
        lines.append(f"  Temporal Resolution  : {temporal_res}s")
    lines.append("")

    # --- Compare with stored state ---
    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    state_path = os.path.join(project_root, STATE_FILE)

    last_avail = load_last_state(state_path)
    is_new_run = (last_avail is None) or (avail_time != last_avail)
    is_fresh = age_minutes <= propagation_minutes

    # Always update state file when a new run is detected
    if is_new_run:
        save_state(state_path, avail_time, model)

    # --- Decision ---
    if last_avail is None:
        lines.append("  First run — no previous state found.")
        lines.append("  => NEW UPDATE (initialising state file)")
        return {"is_update": True, "exit_code": 0, "message": "\n".join(lines), "meta": meta}

    if is_new_run:
        lines.append(f"  Previous availability : {ts_to_str(last_avail)}")
        lines.append(f"  Current  availability : {ts_to_str(avail_time)}")
        lines.append(f"  => NEW UPDATE DETECTED ({age_minutes:.1f} min ago)")
        return {"is_update": True, "exit_code": 0, "message": "\n".join(lines), "meta": meta}

    if is_fresh:
        remaining = propagation_minutes - age_minutes
        lines.append(f"  Last update           : {ts_to_str(avail_time)} ({age_minutes:.1f} min ago)")
        lines.append(f"  => UPDATE STILL FRESH  (propagation window: {remaining:.1f} min remaining)")
        return {"is_update": True, "exit_code": 0, "message": "\n".join(lines), "meta": meta}

    lines.append(f"  Last update           : {ts_to_str(avail_time)} ({age_minutes:.1f} min ago)")
    lines.append(f"  => No recent update. Data fully propagated.")
    return {"is_update": False, "exit_code": 1, "message": "\n".join(lines), "meta": meta}


def main():
    """CLI entry point."""
    # Try to read model/propagation from settings.toml, fall back to CLI args / defaults
    model = None
    propagation_minutes = None

    try:
        from src.config import settings
        model_update_cfg = settings.get('weather', {}).get('open_meteo', {}).get('model_update', {})
        model = model_update_cfg.get('model') or None
        propagation_minutes = model_update_cfg.get('propagation_minutes')
    except Exception:
        pass

    # CLI argument overrides settings.toml
    if len(sys.argv) > 1:
        model = sys.argv[1]

    result = check_update(model=model, propagation_minutes=propagation_minutes)
    print(result["message"])
    sys.exit(result["exit_code"])


if __name__ == "__main__":
    main()

