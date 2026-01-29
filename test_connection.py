
import sys
from src.db import InfluxDBWrapper
from src.config import settings

def test_connection():
    print(f"Testing connection to InfluxDB at {settings['influxdb']['url']}...")
    try:
        db = InfluxDBWrapper()
        # Simple query to check connectivity (e.g. list buckets, but we can't easily list buckets via flux without privileges sometimes)
        # Instead, try a simple range query that returns empty but verifies syntax and auth
        # Or checking the health is better but client doesn't expose it easily in wrapper vs wrapper.client.health()
        
        health = db.client.health()
        if health.status == "pass":
            print("Connection successful! Health check passed.")
        else:
            print(f"Connection failed via health check. Status: {health.status}")
            print(f"Message: {health.message}")
            sys.exit(1)
            
        
        
        # Check if buckets exist
        print("\nChecking Configuration & Buckets...")
        
        
        # Define the mapping of logical names to their sub-keys in [influxdb.buckets], etc.
        config_map = [
            ("Production History", "history_produced", "produced", "produced", "Training: PV yield (Input)"),
            ("Regressor History", "regressor_history", "regressor_history", "regressor_history", "Training: Weather (Input/Aligned)"),
            ("Regressor Future", "regressor_future", "regressor_future", "regressor_future", "Forecast: Weather (Input)"),
            ("Target Forecast (PV)", "target_forecast", "forecast", "forecast", "Output: PV Forecast (Result)"),
        ]

        expected_buckets = set()
        print(f"\nConfiguration from 'settings.toml':")
        print(f"{'Data Type':<25} | {'Purpose':<32} | {'Bucket (Settings Key)':<35} | {'Measurement':<20} | {'Field':<20}")
        print("-" * 140)
        
        for name, b_key, m_key, f_key, purpose in config_map:
            bucket = settings['influxdb']['buckets'].get(b_key, "N/A")
            meas = settings['influxdb']['measurements'].get(m_key, "N/A")
            field = settings['influxdb']['fields'].get(f_key, "N/A")
            
            p_bucket = f"{bucket} ({b_key})" if bucket else "MISSING"
            if bucket: expected_buckets.add(bucket)
            
            # Convert field to string if it's a list (e.g. for regressors)
            field_str = str(field)
            print(f"{name:<25} | {purpose:<32} | {p_bucket:<35} | {meas:<20} | {field_str:<20}")

        print("-" * 140)

        
        try:
            buckets_api = db.client.buckets_api()
            # find_buckets() might fail if the token doesn't have read permissions for buckets
            existing_buckets_obj = buckets_api.find_buckets().buckets
            existing_bucket_names = {b.name for b in existing_buckets_obj}
            
            missing_buckets = expected_buckets - existing_bucket_names
            
            if missing_buckets:
                print(f"\nERROR: The following buckets are MISSING in InfluxDB:")
                for b in missing_buckets:
                    print(f"  - {b}")
                print("Please create these buckets manually in InfluxDB.")
            else:
                print("\nSuccess: All configured buckets exist.")
                
        except Exception as e:
            print(f"\nWARNING: Could not query InfluxDB for existing buckets (likely insufficient permissions).")
            print(f"Error details: {e}")
            print("Please manually verify that the buckets listed above exist in your InfluxDB.")

        
    except Exception as e:
        print(f"Connection failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_connection()
