"""
Forecasting Script for FusionForecast

This script generates multi-step forecasts for PV production using a trained NeuralProphet model.
It leverages Autoregression (AR) to incorporate recent intraday performance into the forecast.

Key Features:
- Fetches future weather regressors (forecasts) from InfluxDB.
- Fetches recent historical data (intraday context) for AR initialization.
- Performs recursive chunked forecasting to generate a continuous prediction.
- Writes results back to InfluxDB.
"""

import os
import sys
import logging
import warnings
import pandas as pd
import numpy as np
import torch
import neuralprophet

# Add project root to path
sys.path.append(os.getcwd())

# Internal modules
from src.config import settings
from src.db import InfluxDBWrapper
from src.preprocess import postprocess_forecast
from src.data_loader import fetch_intraday_data, fetch_future_regressors

# Suppress common noise warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", message=".*DataFrameGroupBy.apply operated on the grouping columns.*")
warnings.filterwarnings("ignore", message=".*DataFrame is highly fragmented.*")
warnings.filterwarnings("ignore", message=".*Trying to infer the `batch_size`.*")

# Suppress NeuralProphet INFO logs
logging.getLogger("NP.df_utils").setLevel(logging.ERROR)
logging.getLogger("NP.data.processing").setLevel(logging.ERROR)
logging.getLogger("NP").setLevel(logging.ERROR)


def run_forecast():
    """
    Main execution pipeline for forecasting.
    
    Steps:
    1. Load the trained NeuralProphet model.
    2. Fetch future regressors (weather forecast).
    3. Fetch historical context (recent production & weather) for AR.
    4. Generate predictions using a recursive loop.
    5. Post-process and save to InfluxDB.
    """
    print("----------------------------------------------------------------")
    print("Starting Forecast Pipeline (NeuralProphet)")
    print("----------------------------------------------------------------")

    # 1. Load Model
    # ----------------------------------------------------------------
    model_path = settings['model']['path']
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run training first.")
        return
        
    print(f"Loading model from: {model_path}")
    # weights_only=False required for full model unpickling (PyTorch > 2.6 safety check)
    model = torch.load(model_path, weights_only=False)
    
    # Restore trainer state if applicable (needed for some NP versions)
    if hasattr(model, 'restore_trainer'):
        model.restore_trainer()

    # Extract model configuration
    n_lags = getattr(model, 'n_lags', 0)
    n_forecasts = getattr(model, 'n_forecasts', 1)
    print(f"Model Configuration: n_lags={n_lags}, n_forecasts={n_forecasts}")


    # 2. Database & Future Regressors
    # ----------------------------------------------------------------
    db = InfluxDBWrapper()
    forecast_days = settings['model']['forecast_days']
    
    # Fetch future weather data (regressors) from InfluxDB
    # This returns a DataFrame with 'ds' and regressor columns (e.g., irradiance, temp)
    df_future_final = fetch_future_regressors(db, forecast_days)
    
    if df_future_final.empty:
        print("Error: Could not fetch future regressors. Aborting.")
        return

    # Identify regressor columns (all columns except time 'ds')
    regressor_names = [col for col in df_future_final.columns if col != 'ds']
    
    # Check if we have enough future data to cover the requested horizon
    duration = df_future_final['ds'].max() - df_future_final['ds'].min()
    available_days = duration.total_seconds() / (24 * 3600)
    if available_days < (forecast_days * 0.5):
        print(f"Error: Insufficient future regressor data ({available_days:.2f}/{forecast_days} days).")
        return


    # 3. Initial Historical Context (for Autoregression)
    # ----------------------------------------------------------------
    # If the model uses Autoregression (n_lags > 0), it needs the most recent 
    # historical values of 'y' (production) to predict the first step.
    
    df_history_final = pd.DataFrame() # Default empty
    
    if n_lags > 0:
        print(f"Autoregression enabled (n_lags={n_lags}). Fetching intraday history...")
        
    if n_lags > 0:
        print(f"Autoregression enabled (n_lags={n_lags}). Fetching intraday history...")
        
        # Calculate how much history we need to fill the lag window
        # We fetch 2x the minimum requirement to be safe against gaps
        hours_needed = (n_lags * 15) / 60
        fetch_hours = max(24, hours_needed * 2)
        
        print(f"Fetching last {int(fetch_hours)} hours of context data...")
        df_history_final = fetch_intraday_data(db, fetch_hours, regressor_names)

        if df_history_final.empty:
            print("Warning: Could not fetch intraday history. AR context will be zero-filled.")
        else:
            last_row = df_history_final.iloc[-1]
            print(f"  > Latest context data: {last_row['ds']} -> {last_row['y']:.2f} W")


    # 4. Recursive Forecasting Loop
    # ----------------------------------------------------------------
    # NeuralProphet (with n_forecasts < horizon) predicts in chunks.
    # For multi-step AR, we must feed the PREDICTIONS of step t as the HISTORY for step t+1.
    
    print(f"Starting recursive forecasting for {len(df_future_final)} periods...")
    
    # Initialize history with real data
    current_history = df_history_final.tail(n_lags).copy()
    future_points = df_future_final.copy()
    predictions = []
    
    # Filter future to start after history ends (avoid overlap)
    if not current_history.empty:
        last_history_ds = current_history['ds'].max()
        original_len = len(future_points)
        future_points = future_points[future_points['ds'] > last_history_ds].copy()
        filtered_count = original_len - len(future_points)
        if filtered_count > 0:
            print(f"  > Filtered {filtered_count} overlapping timestamps from future data")
    
    # Iterate through the future dataframe in chunks of size 'n_forecasts'
    for i in range(0, len(future_points), n_forecasts):
        
        # Take the next chunk of future regressors
        chunk = future_points.iloc[i : i + n_forecasts].copy()
        
        # Fill any missing values in regressors to avoid model crash
        for r in regressor_names:
            if chunk[r].isna().any():
                chunk[r] = chunk[r].interpolate(method='linear', limit_direction='both').fillna(0)

        actual_len = len(chunk)
        chunk['y'] = np.nan # Placeholder for target
        
        # Padding: NeuralProphet expects input length to match n_forecasts exactly
        # If the last chunk is smaller, we pad it with dummy data
        if actual_len < n_forecasts:
            pad_len = n_forecasts - actual_len
            last_ds = chunk['ds'].max()
            padding = pd.DataFrame({
                'ds': pd.date_range(start=last_ds + pd.Timedelta('15min'), periods=pad_len, freq='15min'),
                'y': np.nan
            })
            # Copy last known regressor values to padding
            for col in regressor_names:
                padding[col] = chunk[col].iloc[-1]
                
            chunk = pd.concat([chunk, padding], ignore_index=True)

        # Prepare Input: Concatenate (Last known History) + (Future Chunk)
        step_input = pd.concat([current_history.tail(n_lags), chunk], ignore_index=True)
        step_input = step_input.drop_duplicates(subset='ds', keep='last')  # Remove duplicate timestamps
        step_input['y'] = pd.to_numeric(step_input['y'], errors='coerce')

        try:
            # Predict
            step_forecast = model.predict(step_input)
            
            # Diagonal Retrieval:
            # When n_forecasts > 1, the model outputs multiple columns (yhat1, yhat2, ...).
            # for the k-th step into the future, we want yhat{k} from the row corresponding to that step.
            chunk_preds = []
            for k in range(1, actual_len + 1):
                col_name = f'yhat{k}'
                # The prediction for step k is aligned at row (n_lags + k - 1)
                row_idx = n_lags + k - 1
                
                if col_name in step_forecast.columns and row_idx < len(step_forecast):
                    val = step_forecast.iloc[row_idx][col_name]
                    chunk_preds.append(val)
                else:
                    chunk_preds.append(0.0)
            
            # Save results for this chunk
            res_chunk = future_points.iloc[i : i + actual_len].copy()
            res_chunk['yhat'] = chunk_preds
            predictions.append(res_chunk)
            
            # Update History for Next Iteration:
            # The predictions we just made ('yhat') become the 'y' (history) for the next step.
            new_history = res_chunk.copy()
            new_history['y'] = res_chunk['yhat']
            
            # Ensure we only keep necessary columns for history
            keep_cols = ['ds', 'y'] + [col for col in regressor_names if col in new_history.columns]
            new_history = new_history[keep_cols]
            
            # Append to history buffer
            current_history = pd.concat([current_history, new_history], ignore_index=True)
            
            # Progress Logging
            if (i + actual_len) % n_forecasts == 0 or (i + actual_len) == len(future_points):
                print(f"  > Progress: {min(i + actual_len, len(future_points))}/{len(future_points)} steps generated")
                
        except Exception as e:
            print(f"Error during prediction at step {i}: {e}")
            break
            
    if not predictions:
        print("Error: No predictions generated.")
        return
        
    # Combine all chunks
    forecast = pd.concat(predictions, ignore_index=True)
    print(f"Generated total {len(forecast)} forecast points.")
    
    # Standardize output column
    if 'yhat1' in forecast.columns and 'yhat' not in forecast.columns:
        forecast.rename(columns={'yhat1': 'yhat'}, inplace=True)
    
    
    # 5. Post-Processing & Output
    # ----------------------------------------------------------------
    # Filter to future only
    min_future_ds = df_future_final['ds'].min()
    forecast = forecast[forecast['ds'] >= min_future_ds].copy()
    
    # Apply standard post-processing (clipping negatives, etc.)
    forecast = postprocess_forecast(forecast)
    
    # Apply shifts if configured (e.g., if forecasts are physically offset)
    forecast_offset = settings['model']['preprocessing'].get('forecast_offset', '0m')
    if forecast_offset != "0m":
        try:
            forecast['ds'] = forecast['ds'] + pd.Timedelta(forecast_offset)
        except Exception as e:
            print(f"Warning: Could not apply offset '{forecast_offset}': {e}")

    # Prepare for Database Write
    forecast_to_write = forecast[['ds', 'yhat']].copy()
    
    # Apply simple night filter (force 0 if below threshold)
    threshold = settings['model'].get('preprocessing', {}).get('night_threshold', 50)
    forecast_to_write.loc[forecast_to_write['yhat'] <= threshold, 'yhat'] = 0
    
    # InfluxDB Formatting
    target_bucket = settings['influxdb']['buckets']['target_forecast']
    target_meas = settings['influxdb']['measurements']['forecast']
    target_field = settings['influxdb']['fields']['forecast']
    
    print(f"Writing {len(forecast_to_write)} records to InfluxDB ({target_bucket})...")
    forecast_to_write.rename(columns={'ds': 'time', 'yhat': target_field}, inplace=True)
    forecast_to_write.set_index('time', inplace=True)
    
    db.write_dataframe(
        df=forecast_to_write,
        bucket=target_bucket,
        measurement=target_meas
    )
    
    print("Forecast pipeline completed successfully.")


if __name__ == "__main__":
    run_forecast()
