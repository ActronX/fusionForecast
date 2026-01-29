"""
Shared utilities for weather data fetching and processing.
Contains logic for internal physics calculations (pvlib) and Open-Meteo response handling.
"""

import pandas as pd
import pvlib
import requests_cache
import openmeteo_requests
from retry_requests import retry

def setup_openmeteo_client(cache_path='.cache', expire_after=3600, retries=5, backoff_factor=0.2):
    """
    Configures and returns the Open-Meteo API client with caching and retries.
    
    Args:
        cache_path (str): Path to the cache file.
        expire_after (int): Cache expiration in seconds.
        retries (int): Number of retries for failed requests.
        backoff_factor (float): Backoff factor for retries.
        
    Returns:
        openmeteo_requests.Client: The configured client.
    """
    cache_session = requests_cache.CachedSession(cache_path, expire_after=expire_after)
    retry_session = retry(cache_session, retries=retries, backoff_factor=backoff_factor)
    return openmeteo_requests.Client(session=retry_session)

def calculate_gti_and_clearsky(df, station_settings):
    """
    Calculates Global Tilted Irradiance (GTI) and Clear Sky GHI for the given data.
    
    Modifies the DataFrame in-place to add/overwrite 'global_tilted_irradiance' and add 'clearsky_ghi'.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'diffuse_radiation' and 'direct_normal_irradiance'.
                           Index must be a DatetimeIndex in UTC.
        station_settings (dict): Dictionary containing 'latitude', 'longitude', 'tilt', and 'azimuth'.
        
    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    print("Calculating Solar Position and GTI with pvlib...")
    
    # Azimuth conversion: 
    # Settings/Open-Meteo: 0=South, -90=East, 90=West, 180=North
    # pvlib: 0=North, 90=East, 180=South, 270=West
    # Conversion: pvlib_azimuth = settings_azimuth + 180
    surface_azimuth = (station_settings['azimuth'] + 180) % 360
    surface_tilt = station_settings['tilt']
    
    location = pvlib.location.Location(
        station_settings['latitude'],
        station_settings['longitude'],
        tz='UTC', 
        altitude=station_settings.get('altitude', 0)
    )
    
    solpos = location.get_solarposition(df.index)
    
    # Estimate GHI
    # GHI = DNI * cos(zenith) + DHI
    dni = df['direct_normal_irradiance']
    dhi = df['diffuse_radiation']
    
    # Reconstruct GHI for safety
    ghi = (dni * pvlib.tools.cosd(solpos['zenith']) + dhi).fillna(0)
    ghi[ghi < 0] = 0
    
    # Calculate Total Irradiance (GTI) using 'isotropic' model
    irradiance = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=solpos['zenith'],
        solar_azimuth=solpos['azimuth'],
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        model='isotropic'
    )
    
    # Overwrite/Add GTI
    df['global_tilted_irradiance'] = irradiance['poa_global'].fillna(0)
    
    # Calculate and add Clear Sky GHI
    clearsky = location.get_clearsky(df.index)
    df['clearsky_ghi'] = clearsky['ghi']
    
    return df

def process_hourly_data(response, hourly_params_list):
    """
    Extracts hourly data from Open-Meteo response, resamples to 15min, and returns a DataFrame.
    
    Args:
        response: The Open-Meteo response object (single location).
        hourly_params_list (list): List of variable names requested in 'hourly'.
        
    Returns:
        pd.DataFrame: Hourly data resampled to 15min (interpolated), indexed by date.
                      Returns None or empty DataFrame if no hourly data requested/available.
    """
    if not hourly_params_list:
        return pd.DataFrame()
        
    hourly = response.Hourly()
    
    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )
    }
    
    for i, var_name in enumerate(hourly_params_list):
         hourly_data[var_name] = hourly.Variables(i).ValuesAsNumpy()
         
    df_hourly = pd.DataFrame(data=hourly_data)
    df_hourly.set_index("date", inplace=True)
    
    # Resample hourly to 15min (interpolated)
    df_hourly_resampled = df_hourly.resample('15min').interpolate(method='linear')
    
    return df_hourly_resampled
