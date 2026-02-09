from src.data_loader import fetch_training_data
import pandas as pd
from src.weather_utils import get_solar_position
from src.config import settings

def main():
    try:
        print("Testing fetch_training_data with nighttime zeroing...")
        # Fetch data (verbose=True to see the message)
        data = fetch_training_data(verbose=False)
        
        if data:
            df, regressors = data
            print(f"Successfully fetched {len(df)} rows.")
            
            # Verify using pvlib via wrapper
            
            # Ensure ds is datetime
            df['ds'] = pd.to_datetime(df['ds'])
            
            # Calculate solar position
            print("Calculating solar position for verification...")
            solpos = get_solar_position(df['ds'])
            df['elevation'] = solpos['elevation'].values
            
            # Check if any data with y > 0 has elevation <= -1 (matching your current setting)
            # Note: You set the threshold to -1 in preprocess.py
            violations = df[(df['y'] > 0) & (df['elevation'] <= -1)]
            
            if not violations.empty:
                print(f"WARNING: Found {len(violations)} non-zero values when elevation <= -1!")
                print(violations[['ds', 'y', 'elevation']].head())
            else:
                print("Verification PASSED: No non-zero values found when sun is below -1 degree elevation.")

        else:
            print("No data returned.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
