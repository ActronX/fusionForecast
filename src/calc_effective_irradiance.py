"""
Unified script to calculate Effective Plane of Array (POA) irradiance and Cell Temperature.
It applies the Perez model for orientation, the ASHRAE IAM model for reflection losses,
and the SAPM cell temperature model.
"""

import pandas as pd
import pvlib
import numpy as np
import datetime
from src.config import settings
from src.db import InfluxDBWrapper
from influxdb_client import Point
from influxdb_client.client.write_api import SYNCHRONOUS

def calculate_effective_irradiance(is_future=False):
    """
    Fetches DNI/DHI/Temp/Wind, calculates Perez POA, applies IAM, 
    calculates Cell Temperature, and writes results to InfluxDB.
    """
    mode_text = "Future" if is_future else "Historic"
    print(f"Starting Unified {mode_text} Calculation (IAM + SAPM)...")

    # 1. Connect to InfluxDB
    db = InfluxDBWrapper()
    query_api = db.client.query_api()
    write_api = db.client.write_api(write_options=SYNCHRONOUS)

    if is_future:
        bucket = settings['buckets']['b_regressor_future']
        measurement = settings['measurements']['m_regressor_future']
        # Determine Time Range for Future
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        start_time = now_utc - datetime.timedelta(hours=1)
        end_time = now_utc + datetime.timedelta(days=settings['forecast_parameters']['forecast_days'] + 1)
        range_str = f"start: {start_time.isoformat()}, stop: {end_time.isoformat()}"
    else:
        bucket = settings['buckets']['b_regressor_history']
        measurement = settings['measurements']['m_regressor_history']
        range_str = f"start: -{settings['forecast_parameters']['training_days']}d"

    field_diffuse = settings['fields'].get('f_diffuse', 'diffuse_radiation')
    field_direct = settings['fields'].get('f_direct', 'direct_normal_irradiance')
    field_temp_amb = settings['fields'].get('f_temp_amb', 'temperature_2m')
    field_wind_speed = settings['fields'].get('f_wind_speed', 'wind_speed_10m')
    field_poa = settings['fields'].get('f_poa_perez', 'global_tilted_perez')
    field_effective = settings['fields'].get('f_effective_irradiance', 'effective_irradiance')
    field_temp_cell = settings['fields'].get('f_temp_cell', 'temperature_cell')

    # 2. Fetch Data
    print(f"Fetching data from '{bucket}'...")
    query = f"""
    from(bucket: "{bucket}")
      |> range({range_str})
      |> filter(fn: (r) => r["_measurement"] == "{measurement}")
      |> filter(fn: (r) => r["_field"] == "{field_diffuse}" or r["_field"] == "{field_direct}" or r["_field"] == "{field_temp_amb}" or r["_field"] == "{field_wind_speed}")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    """
    
    df = query_api.query_data_frame(query)

    if df.empty:
        print(f"No data found in InfluxDB bucket '{bucket}'.")
        return

    # Cleanup DataFrame
    df['_time'] = pd.to_datetime(df['_time'])
    df.set_index('_time', inplace=True)
    df.sort_index(inplace=True)

    # 3. Configuration
    lat = settings['open_meteo']['latitude']
    lon = settings['open_meteo']['longitude']
    tilt = settings['open_meteo']['tilt']
    
    # Azimuth conversion (Open-Meteo South=0 to pvlib North=0)
    om_azimuth = settings['open_meteo']['azimuth']
    pvlib_azimuth = (om_azimuth + 180) % 360
    
    print(f"Location: {lat}, {lon}. Tilt: {tilt}. Azimuth (pvlib): {pvlib_azimuth}")

    # 4. Solar Position
    print("Calculating solar position...")
    solpos = pvlib.solarposition.get_solarposition(df.index, lat, lon)

    # 5. Perez POA Components
    print("Calculating Perez POA components...")
    dni = df[field_direct]
    dhi = df[field_diffuse]
    dni_extra = pvlib.irradiance.get_extra_radiation(df.index)
    
    zenith_rad = np.radians(solpos['zenith'])
    ghi = dni * np.cos(zenith_rad) + dhi
    
    # get_total_irradiance returns global, direct, sky_diffuse, ground_diffuse
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=pvlib_azimuth,
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        dni_extra=dni_extra,
        solar_zenith=solpos['zenith'],
        solar_azimuth=solpos['azimuth'],
        model='perez'
    )

    # 6. AOI (Angle of Incidence)
    print("Calculating Angle of Incidence (AOI)...")
    aoi = pvlib.irradiance.aoi(
        surface_tilt=tilt,
        surface_azimuth=pvlib_azimuth,
        solar_zenith=solpos['zenith'],
        solar_azimuth=solpos['azimuth']
    )

    # 7. Apply IAM (Incidence Angle Modifier) using ASHRAE model
    print("Applying ASHRAE IAM model...")
    # Get IAM parameter from settings, default to 0.05
    iam_b = settings.get('pvlib', {}).get('parameters', {}).get('iam_b', 0.05)
    iam_direct = pvlib.iam.ashrae(aoi, b=iam_b)
    iam_diffuse = pvlib.iam.marion_diffuse('ashrae', surface_tilt=tilt)
    sky_iam = iam_diffuse['sky']
    ground_iam = iam_diffuse['ground']

    # 8. Effective Irradiance
    poa_direct_eff = poa['poa_direct'] * iam_direct
    poa_sky_eff = poa['poa_sky_diffuse'] * sky_iam
    poa_ground_eff = poa['poa_ground_diffuse'] * ground_iam
    
    poa_global_eff = poa_direct_eff + poa_sky_eff + poa_ground_eff

    # 9. Cell Temperature (SAPM)
    print("Calculating cell temperature (SAPM: close_mount_glass_glass)...")
    temp_amb = df[field_temp_amb]
    wind_speed = df[field_wind_speed]
    
    #TEMPERATURE_MODEL_PARAMETERS = {
    #    'sapm': {
    #        'open_rack_glass_glass': {'a': -3.47, 'b': -.0594, 'deltaT': 3},
    #        'close_mount_glass_glass': {'a': -2.98, 'b': -.0471, 'deltaT': 1},
    #        'open_rack_glass_polymer': {'a': -3.56, 'b': -.0750, 'deltaT': 3},
    #        'insulated_back_glass_polymer': {'a': -2.81, 'b': -.0455, 'deltaT': 0},
    #    },
    #    'pvsyst': {'freestanding': {'u_c': 29.0, 'u_v': 0},
    #               'insulated': {'u_c': 15.0, 'u_v': 0},
    #               'semi_integrated': {'u_c': 20.0, 'u_v': 0}}
    #}


    # Get SAPM parameters from settings, default to close_mount_glass_glass
    sapm_params = settings.get('pvlib', {}).get('parameters', {})
    s_a = sapm_params.get('sapm_a', -2.98)
    s_b = sapm_params.get('sapm_b', -0.0471)
    s_dT = sapm_params.get('sapm_deltaT', 1)

    temp_cell = pvlib.temperature.sapm_cell(
        poa_global_eff,
        temp_amb,
        wind_speed,
        a=s_a,
        b=s_b,
        deltaT=s_dT
    )

    # 10. Write to InfluxDB
    print(f"Writing results to InfluxDB bucket '{bucket}'...")
    points = []
    for time in poa_global_eff.index:
        val_poa = poa['poa_global'].loc[time]
        val_eff = poa_global_eff.loc[time]
        val_cell = temp_cell.loc[time]
        
        if pd.isna(val_poa) and pd.isna(val_eff) and pd.isna(val_cell):
            continue
            
        point = Point(measurement)\
            .time(time)
        
        if not pd.isna(val_poa):
            point.field(field_poa, float(val_poa))

        if not pd.isna(val_eff):
            point.field(field_effective, float(val_eff))
        
        if not pd.isna(val_cell):
            point.field(field_temp_cell, float(val_cell))
            
        points.append(point)

    if points:
        write_api.write(bucket=bucket, org=settings['influxdb']['org'], record=points)
        print(f"Successfully wrote {len(points)} points.")
    else:
        print("No valid points to write.")

if __name__ == "__main__":
    # If run directly without arguments, assume historic
    import sys
    is_future_flag = "--future" in sys.argv
    calculate_effective_irradiance(is_future=is_future_flag)
