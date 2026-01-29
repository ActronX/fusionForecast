"""
Script to fetch forecast weather data from Open-Meteo and store it in InfluxDB.
Refactored to use shared WeatherFetcher.
"""

import pandas as pd
from src.config import settings
from src.weather_fetcher import WeatherFetcher

def main():
    print("Starting Future Weather (Forecast) data fetch...")
    print(f"Local Time: {pd.Timestamp.now()}")

    # Configuration Check
    if 'weather' not in settings or 'open_meteo' not in settings['weather']:
        print("Error: '[weather.open_meteo]' section missing in settings.toml")
        return

    if 'influxdb' not in settings or 'buckets' not in settings['influxdb'] or 'regressor_future' not in settings['influxdb']['buckets']:
         print("Error: Missing 'regressor_future' in settings.")
         return

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
