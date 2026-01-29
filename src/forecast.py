import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings

sys.path.append(os.getcwd())

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", message=".*DataFrameGroupBy.apply operated on the grouping columns.*")
warnings.filterwarnings("ignore", message=".*DataFrame is highly fragmented.*")
warnings.filterwarnings("ignore", message=".*Trying to infer the `batch_size`.*")

import neuralprophet
from neuralprophet import NeuralProphet
from src.config import settings
from src.db import InfluxDBWrapper
from src.preprocess import postprocess_forecast, prepare_prophet_dataframe 

def run_forecast():
    print("Starting forecast pipeline... (NeuralProphet)")
    
    # Load trained model
    model_path = settings['model']['path']
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run training first.")
        return
        
    print(f"Loading model from {model_path}...")
    # Use torch.load with weights_only=False for PyTorch 2.6+ compatibility
    model = torch.load(model_path, weights_only=False)
    
    # Restore trainer if needed for inference
    if hasattr(model, 'restore_trainer'):
        model.restore_trainer()
        
    # Check AutoRegressive configuration
    n_lags = getattr(model, 'n_lags', 0)
    n_forecasts = getattr(model, 'n_forecasts', 1)
    print(f"Model configuration: n_lags={n_lags}, n_forecasts={n_forecasts}")

    # Initialize DB
    db = InfluxDBWrapper()
    
    # Forecast horizon configuration
    forecast_days = settings['model']['forecast_days']
    
    # Prepare future regressors (weather data)
    print(f"Fetching regressor data for next {forecast_days} days...")
    regressor_offset = settings['model']['preprocessing'].get('regressor_offset', '0m')
    regressor_scale = settings['model']['preprocessing'].get('regressor_scale', 1.0)
    
    # Handle list or string
    reg_config = settings['influxdb']['fields']['regressor_future']
    if isinstance(reg_config, list):
        regressor_fields = reg_config
    else:
        regressor_fields = [reg_config]

    regressor_filter = " or ".join([f'r["_field"] == "{f}"' for f in regressor_fields])
    
    query_regressor = f'''
    import "date"
    from(bucket: "{settings['influxdb']['buckets']['regressor_future']}")
      |> range(start: now(), stop: date.add(d: {forecast_days}d, to: now()))
      |> filter(fn: (r) => r["_measurement"] == "{settings['influxdb']['measurements']['regressor_future']}")
      |> filter(fn: (r) => {regressor_filter})
      |> map(fn: (r) => ({{ r with _value: r._value * {regressor_scale} }}))
      |> timeShift(duration: {regressor_offset})
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    df_future = db.query_dataframe(query_regressor)
    
    if df_future.empty:
        print("Error: No future regressor data found.")
        return

    df_future = prepare_prophet_dataframe(df_future, freq='15min')
    
    # Interpolate and prepare regressor columns
    regressor_names = []
    for field in regressor_fields:
        if field not in df_future.columns:
             print(f"Warning: Future regressor field '{field}' missing. Filling with 0.")
             df_future[field] = 0.0
        df_future[field] = df_future[field].interpolate(method='linear', limit_direction='both')
        regressor_names.append(field)
    
    df_future_final = df_future[['ds'] + regressor_names].copy()
    
    # Validate data sufficiency
    duration = df_future_final['ds'].max() - df_future_final['ds'].min()
    available_days = duration.total_seconds() / (24 * 3600)
    if available_days < (forecast_days * 0.5):
        print(f"Error: Insufficient future regressor data ({available_days:.2f}/{forecast_days} days).")
        return

    # Fetch historical context if AutoRegressive mode is enabled
    if n_lags > 0:
        print(f"AR mode enabled (n_lags={n_lags}). Fetching historical context...")
        
        # Calculate required history (fetch 2x for safety, minimum 24 hours)
        hours_needed = (n_lags * 15) / 60
        fetch_hours = max(24, hours_needed * 2)
        print(f"Fetching last {int(fetch_hours)} hours of history...")
        
        # Fetch historical target data from live bucket
        live_bucket = settings['influxdb']['buckets']['live']
        live_meas = settings['influxdb']['measurements']['live']
        live_field = settings['influxdb']['fields']['live']
        
        # Scaling for live data might differ, check if it's already in Watts
        # Based on check_db_history.py, live data mean is ~80-3000, looks like Watts.
        
        query_history_y = f'''
        import "date"
        from(bucket: "{live_bucket}")
          |> range(start: date.sub(d: {int(fetch_hours)}h, from: now()), stop: now())
          |> filter(fn: (r) => r["_measurement"] == "{live_meas}")
          |> filter(fn: (r) => r["_field"] == "{live_field}")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        df_hist_y = db.query_dataframe(query_history_y)
        
        # Fallback to stats bucket if live data is unavailable
        if df_hist_y.empty:
            print(f"Warning: Live data unavailable. Using stats bucket.")
            target_field = settings['influxdb']['fields']['produced']
            target_meas = settings['influxdb']['measurements']['produced']
            target_bucket = settings['influxdb']['buckets']['history_produced']
            prod_scale = settings['model']['preprocessing'].get('produced_scale', 1.0)
            
            query_history_y = f'''
            import "date"
            from(bucket: "{target_bucket}")
              |> range(start: date.sub(d: {int(fetch_hours)}h, from: now()), stop: now())
              |> filter(fn: (r) => r["_measurement"] == "{target_meas}")
              |> filter(fn: (r) => r["_field"] == "{target_field}")
              |> map(fn: (r) => ({{ r with _value: r._value * {prod_scale} }}))
              |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''
            df_hist_y = db.query_dataframe(query_history_y)
            y_col = target_field
        else:
            y_col = live_field

        # Fetch historical regressors
        reg_hist_bucket = settings['influxdb']['buckets']['regressor_history']
        reg_hist_meas = settings['influxdb']['measurements']['regressor_history']
        regressor_scale = settings['model']['preprocessing'].get('regressor_scale', 1.0)
        regressor_offset = settings['model']['preprocessing'].get('regressor_offset', '0m')
        
        # Re-use regressor_filter since fields should match
        
        query_history_reg = f'''
        import "date"
        from(bucket: "{reg_hist_bucket}")
          |> range(start: date.sub(d: {int(fetch_hours)}h, from: now()), stop: now())
          |> filter(fn: (r) => r["_measurement"] == "{reg_hist_meas}")
          |> filter(fn: (r) => {regressor_filter})
          |> map(fn: (r) => ({{ r with _value: r._value * {regressor_scale} }}))
          |> timeShift(duration: {regressor_offset})
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        df_hist_reg = db.query_dataframe(query_history_reg)

        
        if not df_hist_y.empty and not df_hist_reg.empty:
            # Process and merge historical data
            df_hist_y = prepare_prophet_dataframe(df_hist_y, freq='15min')
            df_hist_reg = prepare_prophet_dataframe(df_hist_reg, freq='15min')
            df_history = pd.merge(df_hist_y, df_hist_reg, on='ds', how='inner')
            
            # Rename target column
            if y_col in df_history.columns:
                df_history.rename(columns={y_col: 'y'}, inplace=True)
            
            # Ensure all regressors are present and interpolated
            for r in regressor_names:
                if r not in df_history.columns:
                    print(f"Warning: Missing regressor {r}. Filling with 0.")
                    df_history[r] = 0
                else:
                    df_history[r] = df_history[r].interpolate(method='linear', limit_direction='both')
            
            # Resample to ensure continuous 15-minute intervals and fill gaps
            df_history_final = df_history.copy()
            df_history_final['ds'] = pd.to_datetime(df_history_final['ds'])
            df_history_final = df_history_final.set_index('ds').resample('15min').mean()
            df_history_final = df_history_final.fillna(0)
            df_history_final = df_history_final.reset_index()
            df_history_final = df_history_final[['ds', 'y'] + regressor_names].copy()
            
            print(f"Loaded {len(df_history_final)} historical data points")
            
    # Generate predictions using recursive chunked forecasting
    print(f"Starting chunked forecasting for {len(df_future_final)} periods (step size={n_forecasts})...")
    
    current_history = df_history_final.tail(n_lags).copy()
    future_points = df_future_final.copy()
    predictions = []
    
    # Process future data in chunks
    for i in range(0, len(future_points), n_forecasts):
        chunk = future_points.iloc[i : i + n_forecasts].copy()
        
        # Ensure regressors have no missing values
        for r in regressor_names:
            if chunk[r].isna().any():
                chunk[r] = chunk[r].interpolate(method='linear', limit_direction='both').fillna(0)

        actual_len = len(chunk)
        chunk['y'] = np.nan
        
        # Pad chunk to exactly n_forecasts if needed (required for multi-step prediction)
        if actual_len < n_forecasts:
            pad_len = n_forecasts - actual_len
            last_ds = chunk['ds'].max()
            padding = pd.DataFrame({
                'ds': pd.date_range(start=last_ds + pd.Timedelta('15min'), periods=pad_len, freq='15min'),
                'y': np.nan
            })
            for col in regressor_names:
                padding[col] = chunk[col].iloc[-1]
            chunk = pd.concat([chunk, padding], ignore_index=True)

        # Combine history with future chunk for prediction
        step_input = pd.concat([current_history.tail(n_lags), chunk], ignore_index=True)
        step_input['y'] = pd.to_numeric(step_input['y'], errors='coerce')

        try:
            # Generate multi-step predictions
            step_forecast = model.predict(step_input)
            
            # Extract predictions using diagonal retrieval
            # For multi-step forecasting, the k-th step prediction is in column 'yhat{k}' at row 'n_lags + k - 1'
            chunk_preds = []
            for k in range(1, actual_len + 1):
                col_name = f'yhat{k}'
                row_idx = n_lags + k - 1
                if col_name in step_forecast.columns and row_idx < len(step_forecast):
                    chunk_preds.append(step_forecast.iloc[row_idx][col_name])
                else:
                    chunk_preds.append(0.0)
            
            # Store predictions for this chunk
            res_chunk = future_points.iloc[i : i + actual_len].copy()
            res_chunk['yhat'] = chunk_preds
            predictions.append(res_chunk)
            
            # Update history with predictions for next recursive iteration
            new_history = res_chunk.copy()
            new_history['y'] = res_chunk['yhat']
            keep_cols = ['ds', 'y'] + [col for col in regressor_names if col in new_history.columns]
            new_history = new_history[keep_cols]
            current_history = pd.concat([current_history, new_history], ignore_index=True)
            
            # Progress update
            if (i + actual_len) % n_forecasts == 0 or (i + actual_len) == len(future_points):
                print(f"Progress: {min(i + actual_len, len(future_points))}/{len(future_points)} steps")
                
        except Exception as e:
            print(f"Error during prediction at step {i}: {e}")
            break
            
    if not predictions:
        print("Error: No predictions generated.")
        return
        
    forecast = pd.concat(predictions, ignore_index=True)
    print(f"Generated {len(forecast)} forecast points.")
    
    # Ensure yhat column exists
    if 'yhat1' in forecast.columns and 'yhat' not in forecast.columns:
        forecast.rename(columns={'yhat1': 'yhat'}, inplace=True)
    
    # Filter to include only future periods
    min_future_ds = df_future_final['ds'].min()
    forecast = forecast[forecast['ds'] >= min_future_ds].copy()
    
    # Apply postprocessing
    forecast = postprocess_forecast(forecast)
    
    # Apply forecast offset if configured
    forecast_offset = settings['model']['preprocessing'].get('forecast_offset', '0m')
    if forecast_offset != "0m":
        try:
            forecast['ds'] = forecast['ds'] + pd.Timedelta(forecast_offset)
        except Exception as e:
            print(f"Warning: Could not apply offset '{forecast_offset}': {e}")

    # Prepare data for writing to InfluxDB
    forecast_to_write = forecast[['ds', 'yhat']].copy()
    
    # Apply night threshold filter
    threshold = settings['model'].get('preprocessing', {}).get('night_threshold', 50)
    forecast_to_write.loc[forecast_to_write['yhat'] <= threshold, 'yhat'] = 0
    
    # Write forecast to InfluxDB
    print(f"Writing forecast to {settings['influxdb']['buckets']['target_forecast']}...")
    forecast_to_write.rename(columns={'ds': 'time', 'yhat': settings['influxdb']['fields']['forecast']}, inplace=True)
    forecast_to_write.set_index('time', inplace=True)
    
    db.write_dataframe(
        df=forecast_to_write,
        bucket=settings['influxdb']['buckets']['target_forecast'],
        measurement=settings['influxdb']['measurements']['forecast']
    )
    
    print("Forecast completed successfully.")

if __name__ == "__main__":
    run_forecast()
