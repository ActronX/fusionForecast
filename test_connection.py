
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
            if bucket: 
                if isinstance(bucket, list):
                     for b in bucket: expected_buckets.add(b)
                else:
                     expected_buckets.add(bucket)
            
            # Helper to safely stringify lists for display
            s_bucket = str(p_bucket)
            s_meas = str(meas)
            s_field = str(field)

            
            print(f"{name:<25} | {purpose:<32} | {s_bucket:<35} | {s_meas:<20} | {s_field:<20}")

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

    # -------------------------------------------------------------------------
    # Training History Check (Merged from check_training_history.py)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 30)
    print("Training History Check")
    print("=" * 30)
    
    try:
        from src.history_validation import get_max_available_training_days
        
        current_setting = settings['model']['training_days']
        print(f"Current Configuration: {current_setting} days")
        
        # Check distinct maximum (e.g. 10 years) to find absolute limit
        # Suppress verbose output from data_loader to keep test_connection output clean(er)
        # or keep it verbose if user wants details. Let's keep it verbose=True as requested.
        print("-" * 30)
        days, density, actual, expected, start_date = get_max_available_training_days(max_days=3650, verbose=True)
        
        print("-" * 30)
        print("RESULT:")
        if start_date:
            print(f"  Common Start Date:   {start_date}")
            print(f"  Max Valid History:   {days} days")
            print(f"  Data Coverage:       {density:.2%} (All fields checked)")
            
            print("-" * 30)
            print("SUGGESTION:")
            if days > current_setting:
                print(f"  \033[92m[UPGRADE]\033[0m You have {days} days of valid data available.")
                print(f"  Suggestion: Update settings.toml 'training_days' from {current_setting} to {days}.")
            elif days < current_setting:
                print(f"  \033[93m[DOWNGRADE]\033[0m Your config ({current_setting}) exceeds valid history ({days}).")
                print(f"  Suggestion: Update settings.toml 'training_days' from {current_setting} to {days}.")
            else:
                print(f"  \033[94m[OK]\033[0m Configuration match. You are using the maximum available data.")
        else:
            print("  Status:              \033[91mNO DATA FOUND\033[0m")
            print("  (This is expected if the database is empty)")

    except ImportError:
        print("WARNING: Could not import src.data_loader. Skipping history check.")
    except Exception as e:
        print(f"WARNING: History check failed: {e}")

if __name__ == "__main__":
    test_connection()
