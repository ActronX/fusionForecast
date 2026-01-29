"""
Unified Weather Fetcher Class
Encapsulates logic for fetching data from Open-Meteo and writing to InfluxDB.
"""

import pandas as pd
from influxdb_client import Point
from influxdb_client.client.write_api import SYNCHRONOUS
from src.config import settings
from src.db import InfluxDBWrapper
from src.weather_utils import setup_openmeteo_client, calculate_gti_and_clearsky, process_hourly_data


class WeatherFetcher:
    def __init__(self):
        """Initialize the WeatherFetcher with API clients."""
        self.openmeteo = setup_openmeteo_client()
        self.db_wrapper = InfluxDBWrapper()

    def fetch_data(self, url, params, station_settings):
        """
        Fetches and processes weather data from Open-Meteo.
        
        Args:
            url (str): The Open-Meteo API endpoint URL.
            params (dict): Dictionary of parameters for the API call.
            station_settings (dict): Station settings for GTI calculation (lat, lon, tilt, azimuth).
            
        Returns:
            pd.DataFrame: Processed weather data with calculated fields, or empty DataFrame on failure.
        """
        print(f"Requesting data from {url} with params: {params}")
        try:
            responses = self.openmeteo.weather_api(url, params=params)
        except Exception as e:
            print(f"Error calling Open-Meteo API: {e}")
            return pd.DataFrame()

        if not responses:
            print("No response received from Open-Meteo.")
            return pd.DataFrame()

        # Process first location (only one requested)
        response = responses[0]
        print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")

        # Process minutely_15 data
        minutely_15 = response.Minutely15()
        if not minutely_15:
            print("No minutely_15 data in response.")
            return pd.DataFrame()

        # Assuming variables order matches request: [diffuse, direct]
        # We need to be careful if the order changes, but for now we rely on the specific call structure.
        # Ideally we'd map variables by name if the library supports it easily, 
        # but the generated client uses index access.
        minutely_15_diffuse = minutely_15.Variables(0).ValuesAsNumpy()
        minutely_15_direct = minutely_15.Variables(1).ValuesAsNumpy()

        minutely_15_data = {
            "date": pd.date_range(
                start=pd.to_datetime(minutely_15.Time(), unit="s", utc=True),
                end=pd.to_datetime(minutely_15.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=minutely_15.Interval()),
                inclusive="left"
            ),
            "diffuse_radiation": minutely_15_diffuse,
            "direct_normal_irradiance": minutely_15_direct
        }

        df = pd.DataFrame(data=minutely_15_data)
        df.set_index("date", inplace=True)

        # Calculate Solar Position, GTI, and Clear Sky
        df = calculate_gti_and_clearsky(df, station_settings)

        # Process hourly data if requested
        if params.get("hourly"):
            df_hourly_resampled = process_hourly_data(response, params["hourly"])
            if not df_hourly_resampled.empty:
                # Merge with minutely data
                df = df.join(df_hourly_resampled, how='left')

        # Reset index to make 'date' a column again
        df.reset_index(inplace=True)

        # Clean data
        original_len = len(df)
        df.dropna(inplace=True)
        dropped_count = original_len - len(df)
        if dropped_count > 0:
            print(f"Dropped {dropped_count} rows with NaNs.")

        print(f"Fetched {len(df)} valid data points.")
        return df

    def write_to_influx(self, df, bucket, measurement):
        """
        Writes the dataframe to InfluxDB.
        
        Args:
            df (pd.DataFrame): Data to write.
            bucket (str): Target InfluxDB bucket.
            measurement (str): Target InfluxDB measurement.
        """
        if df.empty:
            print("No data to write.")
            return

        write_api = self.db_wrapper.client.write_api(write_options=SYNCHRONOUS)

        points = []
        print(f"Target Measurement: '{measurement}'")
        print(f"Preparing points for InfluxDB bucket '{bucket}'...")

        for _, row in df.iterrows():
            point = Point(measurement).time(row["date"])

            # Add all columns as fields, excluding 'date'
            for col in df.columns:
                if col != "date":
                    point.field(col, float(row[col]))

            points.append(point)

        print(f"Writing {len(points)} points to InfluxDB...")
        try:
            write_api.write(bucket=bucket, org=settings['influxdb']['org'], record=points)
            print("Successfully wrote data to InfluxDB.")
        except Exception as e:
            print(f"Failed to write to InfluxDB: {e}")
