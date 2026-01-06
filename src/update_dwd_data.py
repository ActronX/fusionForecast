"""
Script to fetch forecast DWD weather data from Open-Meteo and store it in InfluxDB.
"""

import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from src.config import settings
from src.db import InfluxDBWrapper
from influxdb_client import Point
from influxdb_client.client.write_api import SYNCHRONOUS

def fetch_forecast_data():
    """Fetches forecast data from Open-Meteo API based on settings."""
    
    # Setup the Open-Meteo API client
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Get Open-Meteo Forecast parameters from settings
    # We expect a [open_meteo] section with a [open_meteo.forecast] subtable
    if 'open_meteo' not in settings:
        print("Error: '[open_meteo]' section missing in settings.toml")
        return pd.DataFrame()

    om_settings = settings['open_meteo']
    om_forecast = om_settings.get('forecast', {})
    
    url = om_forecast.get('url', "https://api.open-meteo.com/v1/forecast")
    params = {
        "latitude": om_settings['latitude'],
        "longitude": om_settings['longitude'],
        "minutely_15": om_forecast.get('minutely_15', "global_tilted_irradiance_instant"),
        "models": om_forecast.get('models', 'best_match'),
        "tilt": om_settings['tilt'],
        "azimuth": om_settings['azimuth'],
        "forecast_days": om_forecast.get('forecast_days', 3)
    }

    print(f"Requesting forecast data from {url} with params: {params}")
    responses = openmeteo.weather_api(url, params=params)
    
    # Process first location (only one requested)
    response = responses[0]
    print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
    
    # Process minutely_15 data
    minutely_15 = response.Minutely15()
    
    # Verify variable exists
    # Assuming single variable requested for minutely_15
    minutely_15_values = minutely_15.Variables(0).ValuesAsNumpy()

    minutely_15_data = {
        "date": pd.date_range(
            start=pd.to_datetime(minutely_15.Time(), unit="s", utc=True),
            end=pd.to_datetime(minutely_15.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=minutely_15.Interval()),
            inclusive="left"
        ),
        "global_tilted_irradiance": minutely_15_values 
        # Note: We map minutely_15 values to 'global_tilted_irradiance' 
        # to match the field name expected by the rest of the pipeline.
    }

    df = pd.DataFrame(data=minutely_15_data)
    
    # Clean data (usually forecasts shouldn't have NaNs but good practice)
    original_len = len(df)
    df.dropna(inplace=True)
    dropped_count = original_len - len(df)
    if dropped_count > 0:
        print(f"Dropped {dropped_count} rows with NaNs.")
        
    print(f"Fetched {len(df)} valid forecast points.")
    return df

def write_to_influx(df):
    """Writes the dataframe to InfluxDB using same keys as historic fetcher."""
    

    bucket = settings['buckets']['b_dwd_historic']
    measurement = settings['measurements']['m_dwd_historic']
    field = settings['fields']['f_dwd_historic']
    
    db_wrapper = InfluxDBWrapper()
    write_api = db_wrapper.client.write_api(write_options=SYNCHRONOUS)
    
    points = []
    print(f"Preparing points for InfluxDB bucket '{bucket}'...")
    
    for _, row in df.iterrows():
        point = Point(measurement)\
            .field(field, float(row["global_tilted_irradiance"]))\
            .time(row["date"])
        points.append(point)

    print(f"Writing {len(points)} points to InfluxDB...")
    write_api.write(bucket=bucket, org=settings['influxdb']['org'], record=points)
    print("Successfully wrote data to InfluxDB.")

def main():
    print("Starting DWD forecast data fetch...")

    # Configuration Check
    required_keys = ['b_dwd_historic', 'm_dwd_historic', 'f_dwd_historic']
    if any(k not in settings.get('buckets', {}) for k in ['b_dwd_historic']):
         print("Error: Missing 'b_dwd_historic' in settings.")
         return

    try:
        df = fetch_forecast_data()
        if not df.empty:
            write_to_influx(df)
        else:
            print("No data fetched.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
