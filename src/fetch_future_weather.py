"""
Script to fetch forecast weather data from Open-Meteo and store it in InfluxDB.
Optionally checks for a model update before fetching.

Usage:
    python -m src.fetch_future_weather [--force]
"""

import os
import sys
import argparse
import pandas as pd
from src.config import settings
from src.weather_fetcher import WeatherFetcher
from src.check_model_update import STATE_FILE

def main():
    parser = argparse.ArgumentParser(description="Fetch forecast weather data from Open-Meteo.")
    parser.add_argument("--force", action="store_true",
                        help="Force fetch: delete state file and skip model update check.")
    args = parser.parse_args()

    print("Starting Future Weather (Forecast) data fetch...")
    print(f"Local Time: {pd.Timestamp.now()}")

    # Configuration Check
    if 'weather' not in settings or 'open_meteo' not in settings['weather']:
        print("Error: '[weather.open_meteo]' section missing in settings.toml")
        return

    if 'influxdb' not in settings or 'buckets' not in settings['influxdb'] or 'regressor_future' not in settings['influxdb']['buckets']:
         print("Error: Missing 'regressor_future' in settings.")
         return

    # --- Model Update Check ---
    if args.force:
        # Delete state file and skip check
        project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        state_path = os.path.join(project_root, STATE_FILE)
        if os.path.exists(state_path):
            os.remove(state_path)
            print(f"--force: Deleted state file '{STATE_FILE}'.")
        print("--force: Skipping model update check. Forcing fetch...")
    else:
        model_update_cfg = settings['weather']['open_meteo'].get('model_update', {})
        model_name = model_update_cfg.get('model', '')
        propagation_minutes = model_update_cfg.get('propagation_minutes')

        if model_name:
            from src.check_model_update import check_update
            result = check_update(model=model_name, propagation_minutes=propagation_minutes)
            print(result["message"])

            if result["exit_code"] == 1:
                print("No new model data available. Skipping fetch.")
                return
            elif result["exit_code"] == 2:
                print("Warning: Model update check failed. Proceeding with fetch anyway.")
            else:
                print("New model data available. Proceeding with fetch...")
        else:
            print("Model update check disabled (no model configured).")

    # --- Fetch Weather Data ---
    station_settings = settings['station']
    om_forecast = settings['weather']['open_meteo']['forecast']
    
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": station_settings['latitude'],
        "longitude": station_settings['longitude'],
        "minutely_15": om_forecast.get('minutely_15', []),
        "hourly": om_forecast.get('hourly', []),
        "models": om_forecast.get('models', 'best_match'),
        "tilt": station_settings['tilt'],
        "azimuth": station_settings['azimuth'],
        "forecast_days": om_forecast.get('forecast_days', 3)
    }

    fetcher = WeatherFetcher()
    
    try:
        df = fetcher.fetch_data(url, params, station_settings)
        if not df.empty:
            print("Preview of data to be written:")
            print(df.head())
            
            bucket = settings['influxdb']['buckets']['regressor_future']
            measurement = settings['influxdb']['measurements']['regressor_future']
            
            fetcher.write_to_influx(df, bucket, measurement)
        else:
            print("No data fetched.")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

