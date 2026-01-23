
import pandas as pd
import sys
import os
from src.config import settings
from src.db import InfluxDBWrapper

def import_csv(file_path):
    """
    Imports historical PV data from a CSV file.
    Expected CSV format: timestamp, value
    Example:
    2024-01-01 12:00:00, 1500.5
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    print(f"Reading data from {file_path}...")
    try:
        # Try to read CSV
        # We assume first column is time, second is value
        df = pd.read_csv(file_path, header=None, names=['time', 'value'])
        
        # Convert time
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        # Scale if necessary
        produced_scale = settings['model']['preprocessing'].get('produced_scale', 1.0)
        if produced_scale != 1.0:
            df['value'] = df['value'] * produced_scale
            
        # Prepare for InfluxDB (Rename 'value' to the field name in settings)
        field_name = settings['influxdb']['fields']['produced']
        df.rename(columns={'value': field_name}, inplace=True)
        
        # Initialize DB
        db = InfluxDBWrapper()
        
        bucket = settings['influxdb']['buckets']['history_produced']
        measurement = settings['influxdb']['measurements']['produced']
        
        print(f"Writing {len(df)} points to {bucket} (Measurement: {measurement})...")
        db.write_dataframe(df, bucket, measurement)
        print("Import complete!")
        
    except Exception as e:
        print(f"Import failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 -m src.import_pv_history <path_to_csv>")
    else:
        import_csv(sys.argv[1])
