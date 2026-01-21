"""
Script to fetch historic weather data from Open-Meteo and store it in InfluxDB.
"""

import openmeteo_requests
import requests_cache
import pandas as pd
from datetime import date
from retry_requests import retry
from src.config import settings
from src.db import InfluxDBWrapper
from influxdb_client import Point
from influxdb_client.client.write_api import SYNCHRONOUS
from src.calc_effective_irradiance import calculate_effective_irradiance

def fetch_weather_data():
    """Fetches weather data from Open-Meteo API based on settings."""
    
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Get Open-Meteo parameters from settings
    if 'open_meteo' not in settings:
        print("Error: '[open_meteo]' section missing in settings.toml")
        return pd.DataFrame()

    om_settings = settings['open_meteo']
    om_historic = om_settings.get('historic', {})
    
    url = om_historic.get('url', "https://historical-forecast-api.open-meteo.com/v1/forecast")
    
    end_date = om_historic['end_date']
    if end_date.lower() == "today":
        end_date = date.today().strftime("%Y-%m-%d")

    params = {
        "latitude": om_settings['latitude'],
        "longitude": om_settings['longitude'],
        "start_date": om_historic['start_date'],
        "end_date": end_date,
        "minutely_15": [
            om_historic.get('minutely_15', "global_tilted_irradiance_instant"), 
            "diffuse_radiation", 
            "direct_normal_irradiance",
            "temperature_2m",
            "wind_speed_10m"
        ],
        "models": om_historic.get('models', 'icon_d2'),
        "tilt": om_settings['tilt'], 
        "azimuth": om_settings['azimuth']
    }

    print(f"Requesting data from {url} with params: {params}")
    responses = openmeteo.weather_api(url, params=params)
    
    # Process first location (only one requested)
    response = responses[0]
    print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
    
    # Process minutely_15 data
    minutely_15 = response.Minutely15()
    
    # Verify variable exists
    # Assuming single variable 'global_tilted_irradiance_instant' mapped to 'global_tilted_irradiance'
    minutely_15_values = minutely_15.Variables(0).ValuesAsNumpy()
    diffuse_radiation_values = minutely_15.Variables(1).ValuesAsNumpy()
    direct_normal_irradiance_values = minutely_15.Variables(2).ValuesAsNumpy()
    temperature_2m_values = minutely_15.Variables(3).ValuesAsNumpy()
    wind_speed_10m_values = minutely_15.Variables(4).ValuesAsNumpy()

    minutely_15_data = {
        "date": pd.date_range(
            start=pd.to_datetime(minutely_15.Time(), unit="s", utc=True),
            end=pd.to_datetime(minutely_15.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=minutely_15.Interval()),
            inclusive="left"
        ),
        "global_tilted_irradiance": minutely_15_values,
        "diffuse_radiation": diffuse_radiation_values,
        "direct_normal_irradiance": direct_normal_irradiance_values,
        "temperature_2m": temperature_2m_values,
        "wind_speed_10m": wind_speed_10m_values
        # Note: We map minutely_15 values to 'global_tilted_irradiance' 
        # to match the field name expected by the rest of the pipeline.
    }

    df = pd.DataFrame(data=minutely_15_data)
    
    # Clean data
    original_len = len(df)
    df.dropna(inplace=True)
    dropped_count = original_len - len(df)
    if dropped_count > 0:
        print(f"Dropped {dropped_count} rows with NaNs.")
        
    print(f"Fetched {len(df)} valid data points.")
    return df

def write_to_influx(df):
    """Writes the dataframe to InfluxDB."""
    

    bucket = settings['buckets']['b_regressor_history']
    measurement = settings['measurements']['m_regressor_history']
    field_irradiance = settings['fields']['f_regressor_history']
    field_diffuse = settings['fields'].get('f_diffuse', 'diffuse_radiation')
    field_direct = settings['fields'].get('f_direct', 'direct_normal_irradiance')
    field_temp_amb = settings['fields'].get('f_temp_amb', 'temperature_2m')
    field_wind_speed = settings['fields'].get('f_wind_speed', 'wind_speed_10m')
    
    db_wrapper = InfluxDBWrapper()
    write_api = db_wrapper.client.write_api(write_options=SYNCHRONOUS)
    
    points = []
    print(f"Target Measurement: '{measurement}'")
    print(f"Target Fields: '{field_irradiance}', '{field_diffuse}', '{field_direct}', '{field_temp_amb}', '{field_wind_speed}'")
    print(f"Preparing points for InfluxDB bucket '{bucket}'...")
    
    for _, row in df.iterrows():
        point = Point(measurement)\
            .field(field_irradiance, float(row["global_tilted_irradiance"]))\
            .field(field_diffuse, float(row["diffuse_radiation"]))\
            .field(field_direct, float(row["direct_normal_irradiance"]))\
            .field(field_temp_amb, float(row["temperature_2m"]))\
            .field(field_wind_speed, float(row["wind_speed_10m"]))\
            .time(row["date"])
        points.append(point)

    print(f"Writing {len(points)} points to InfluxDB...")
    write_api.write(bucket=bucket, org=settings['influxdb']['org'], record=points)
    print("Successfully wrote data to InfluxDB.")

def main():
    print("Starting historic weather data fetch...")

    # Configuration Check
    required_keys = ['b_regressor_history', 'm_regressor_history', 'f_regressor_history']
    if any(k not in settings.get('buckets', {}) for k in ['b_regressor_history']) or \
       any(k not in settings.get('measurements', {}) for k in ['m_regressor_history']) or \
       any(k not in settings.get('fields', {}) for k in ['f_regressor_history']):
         print("Error: Missing required InfluxDB settings in settings.toml.")
         return

    try:
        df = fetch_weather_data()
        if not df.empty:
            write_to_influx(df)
            
            # Check if we should calculate Perez POA / Effective Irradiance
            if settings.get('prophet', {}).get('use_pvlib', False):
                print("Perez/Effective GHI calculation is enabled. Running calculation...")
                calculate_effective_irradiance(is_future=False)
        else:
            print("No data fetched.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
