
import sys
import os

# Ensure we can import from src
# Ensure we can import from src and root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.train import train_model
from src.forecast import run_forecast
import src.fetch_historic_weather
import src.update_future_weather
import test_connection

def main():
    print("=== Starting FusionForecast Pipeline ===")
    
    print("\n--- Step 1: Connection Test ---")
    try:
        test_connection.test_connection()
    except SystemExit as e:
        if e.code != 0:
            print(f"Pipeline stopped due to connection failure.")
            sys.exit(e.code)
    except Exception as e:
        print(f"Error during connection test: {e}")
        sys.exit(1)

    print("\n--- Step 2: Fetch Historic Weather ---")
    try:
        src.fetch_historic_weather.main()
    except Exception as e:
        print(f"Error fetching historic weather: {e}")
        # We continue, assuming it might not be critical or let user see error

    print("\n--- Step 3: Update Future Weather ---")
    try:
        src.update_future_weather.main()
    except Exception as e:
        print(f"Error updating future weather: {e}")

    print("\n--- Step 4: Training ---")
    try:
        train_model()
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)
        
    print("\n--- Step 5: Forecasting ---")
    try:
        run_forecast()
    except Exception as e:
        print(f"Error during forecasting: {e}")
        sys.exit(1)
        
    print("\n=== Pipeline Complete ===")

if __name__ == "__main__":
    main()
