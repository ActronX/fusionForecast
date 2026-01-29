"""
Script to fetch historic weather data from Open-Meteo and store it in InfluxDB.
Refactored to use shared WeatherFetcher.
"""

import pandas as pd
from datetime import date
from src.config import settings
from src.weather_fetcher import WeatherFetcher

def main():
    print("Starting historic weather data fetch...")
    print(f"Local Time: {pd.Timestamp.now()}")

    # Configuration Check
    if 'weather' not in settings or 'open_meteo' not in settings['weather']:
        print("Error: '[weather.open_meteo]' section missing in settings.toml")
        return

    if 'influxdb' not in settings or 'buckets' not in settings['influxdb'] or 'regressor_history' not in settings['influxdb']['buckets']:
         print("Error: Missing required InfluxDB settings in settings.toml.")
         return

    station_settings = settings['station']
    om_historic = settings['weather']['open_meteo']['historic']
    
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    
    end_date = om_historic['end_date']
    if end_date.lower() == "today":
        end_date = date.today().strftime("%Y-%m-%d")

    params = {
        "latitude": station_settings['latitude'],
        "longitude": station_settings['longitude'],
        "start_date": om_historic['start_date'],
        "end_date": end_date,
        "minutely_15": om_historic.get('minutely_15', []),
        "hourly": om_historic.get('hourly', []),
        "models": om_historic.get('models', 'icon_d2'),
        "tilt": station_settings['tilt'], 
        "azimuth": station_settings['azimuth']
    }

    fetcher = WeatherFetcher()
    
    try:
        df = fetcher.fetch_data(url, params, station_settings)
        if not df.empty:
            print("Preview of data to be written:")
            print(df.head())
            
            bucket = settings['influxdb']['buckets']['regressor_history']
            measurement = settings['influxdb']['measurements']['regressor_history']
            
            fetcher.write_to_influx(df, bucket, measurement)
        else:
            print("No data fetched.")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
