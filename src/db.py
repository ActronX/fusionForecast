
import pandas as pd
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS
from src.config import settings

class InfluxDBWrapper:
    def __init__(self):
        self.url = settings['influxdb']['url']
        self.token = settings['influxdb']['token']
        self.org = settings['influxdb']['org']
        
        try:
            self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
            # Check connection status
            if not self.client.ping():
                raise ConnectionError(f"Could not connect to InfluxDB at {self.url}. Please check URL and network.")
            
            # Optional: Check if org exists or accessible (requires high privs usually, skipping)
            
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            self.query_api = self.client.query_api()
            print(f"Successfully connected to InfluxDB at {self.url}")
            
        except Exception as e:
            print(f"CRITICAL: Failed to initialize InfluxDB connection: {e}")
            raise

    def query_dataframe(self, query):
        """
        Executes a Flux query and returns the result as a pandas DataFrame.
        Handles errors gracefully.
        """
        try:
            df = self.query_api.query_data_frame(query=query)
            if isinstance(df, list):
                # If multiple tables are returned, concatenate them
                df = pd.concat(df, ignore_index=True)
            return df
        except Exception as e:
            error_msg = str(e)
            if "could not find bucket" in error_msg.lower():
                print(f"ERROR: Bucket missing or inaccessible. Details: {e}")
            elif "unauthorized" in error_msg.lower():
                print(f"ERROR: Authorization failed. Check your token. Details: {e}")
            elif "failed to connect" in error_msg.lower():
                 print(f"ERROR: Connection failed during query. Details: {e}")
            else:
                print(f"Query failed with unexpected error: {e}")
            # Re-raise to ensure caller knows it failed
            raise

    def write_dataframe(self, df, bucket, measurement, field_columns=None, tag_columns=None):
        """
        Writes a pandas DataFrame to InfluxDB.
        """
        try:
            # Prepare DataFrame for InfluxDB
            # Ensure index is datetime and named 'time' if possible, or handled by the client
            
            # The client handles DataFrame writing well if we set data_frame_measurement_name etc.
            # But let's be explicit with write_points if needed, or use the write_api.write helper
            
            self.write_api.write(
                bucket=bucket,
                org=self.org,
                record=df,
                data_frame_measurement_name=measurement,
                data_frame_tag_columns=tag_columns
            )
            print(f"Successfully wrote {len(df)} records to {bucket}.")
        except Exception as e:
            if "not found" in str(e).lower() and "bucket" in str(e).lower():
                 print(f"ERROR: Target bucket '{bucket}' does not exist.")
            else:
                 print(f"Write failed: {e}")
            raise

    def close(self):
        self.client.close()
