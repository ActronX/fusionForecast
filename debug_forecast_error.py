
import pandas as pd
import numpy as np
import neuralprophet
import torch
from src.config import settings

def debug_step_96():
    print("--- Debugging Step 96 Error ---")
    
    # 1. Load Model
    model_path = settings['model']['path']
    print(f"Loading model from {model_path}...")
    try:
        model = neuralprophet.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Check regressors
    if hasattr(model, 'future_regressors'):
        print(f"Model Regressors: {list(model.future_regressors.keys())}")
    
    # 2. Construct Mock Dataframe for Step 96
    # 96 steps of history (n_lags=192 actually, wait. The loop uses tail(192))
    # Step 0 produced 96 predictions. Step 96 takes those 96 + previous 96 history.
    # Total input to predict (for n_lags=192) has length 192 (history) + 96 (future) = 288 rows.
    
    n_lags = 192
    n_forecasts = 96
    total_rows = n_lags + n_forecasts
    
    # Create timestamps
    start_date = pd.Timestamp("2026-02-09 00:00:00")
    dates = pd.date_range(start=start_date, periods=total_rows, freq="15min")
    
    # Create Dataframe
    df = pd.DataFrame({'ds': dates})
    
    # Add Y values
    # First n_lags are history (some real numbers)
    # Last n_forecasts are NaN (future)
    y_history = np.random.rand(n_lags) * 1000
    y_future = [np.nan] * n_forecasts
    df['y'] = np.concatenate([y_history, y_future])
    
    # Add Regressors
    # "global_tilted_irradiance", "temperature_2m", "snow_depth", "wind_speed_10m"
    # Fill with dummy data
    for reg in ["global_tilted_irradiance", "temperature_2m", "snow_depth", "wind_speed_10m"]:
        df[reg] = np.random.rand(total_rows) * 10
        
    print(f"Created dummy dataframe: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # 3. Predict
    print("\nAttempting prediction with complete dataframe...")
    try:
        forecast = model.predict(df)
        print("Success! Prediction worked.")
    except Exception as e:
        print(f"\nFAILURE: {e}")
        # import traceback
        # traceback.print_exc()
        
    # 4. Debug: Try 3 regressors?
    # What if the model really expects 3?
    # But how? The error says 'tensor b (4)'. b is usually weights. a is input.
    # So model expects 4. But input has 3?
    # How can input have 3 if we gave it 4 columns?
    
    # Maybe one regressor name is mismatched?
    # Let's check exact names in model
    reg_names = list(model.future_regressors.keys())
    print(f"\nExact Model Regressors: {reg_names}")
    
    # Verify columns match
    missing = [r for r in reg_names if r not in df.columns]
    if missing:
        print(f"MISSING COLUMNS IN DF: {missing}")
    else:
        print("All model regressors present in DF.")

    # 5. Inspect Model Internals (Sparsity)
    if hasattr(model, 'regressor_params'):
        print("\nRegressor Params:")
        print(model.regressor_params)

if __name__ == "__main__":
    debug_step_96()
