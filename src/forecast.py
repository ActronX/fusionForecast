
import os
import pickle
import pandas as pd
import numpy as np
import sys
sys.path.append(os.getcwd())
from src.config import settings
from src.db import InfluxDBWrapper
from src.preprocess import truncate_time_column, postprocess_forecast, prepare_prophet_dataframe
# NeuralProphet import not strictly needed for loading pickle if class is available, but good practice
import neuralprophet
from neuralprophet import NeuralProphet 

def run_forecast():
    print("Starting forecast pipeline... (NeuralProphet)")
    
    # 1. Load Model
    model_path = settings['model']['path']
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run training first.")
        return
        
    print(f"Loading model from {model_path}...")
    # with open(model_path, "rb") as f:
    #     model = pickle.load(f)
    
    import torch
    # Fix for PyTorch 2.6+ weights_only=True default
    # neuralprophet.load doesn't expose weights_only, so we use torch.load directly
    print("Loading with torch.load(weights_only=False)...")
    model = torch.load(model_path, weights_only=False)
    # model = neuralprophet.load(model_path)
    
    # Needs to restore trainer for inference if it was stripped or not saved fully
    if hasattr(model, 'restore_trainer'):
        print("Restoring trainer...")
        model.restore_trainer()
        
    # Check if model has n_lags enabled (AR model)
    n_lags = getattr(model, 'n_lags', 0)
    n_forecasts = getattr(model, 'n_forecasts', 1)
    print(f"Model n_lags: {n_lags}, n_forecasts: {n_forecasts}")

    # Initialize DB
    db = InfluxDBWrapper()
    
    # Forecast Horizon
    forecast_days = settings['model']['forecast_days']
    
    # --- Prepare Future Regressors ---
    print(f"Fetching regressor data for next {forecast_days} days...")
    regressor_offset = settings['model']['preprocessing'].get('regressor_offset', '0m')
    regressor_scale = settings['model']['preprocessing'].get('regressor_scale', 1.0)
    
    regressor_fields = [settings['influxdb']['fields']['regressor_future']]
    
    # Future Regressors Query
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
    
    # Interpolate future regressors
    regressor_names = []
    for field in regressor_fields:
        reg_name = field
        df_future[reg_name] = df_future[reg_name].interpolate(method='linear', limit_direction='both')
        regressor_names.append(reg_name)
    
    # Filter columns
    df_future_final = df_future[['ds'] + regressor_names].copy()
    
    # --- Data Sufficiency Check ---
    duration = df_future_final['ds'].max() - df_future_final['ds'].min()
    available_days = duration.total_seconds() / (24 * 3600)
    if available_days < (forecast_days * 0.5):
         print(f"Error: Insufficient future regressor data ({available_days:.2f}/{forecast_days} days).")
         return

    # --- Fetch Historical Context (if AR enabled) ---
    df_input = df_future_final
    
    if n_lags > 0:
        print(f"AR enabled (n_lags={n_lags}). Fetching historical context...")
        # We need at least n_lags of history. Let's fetch 2x to be safe.
        # Assuming 15min freq.
        # n_lags * 15min. 
        # Example: 24 lags * 15min = 6 hours.
        hours_needed = (n_lags * 15) / 60
        fetch_hours = max(24, hours_needed * 2) # Fetch at least 24h context
        
        print(f"Fetching last {fetch_hours} hours of history...")
        
        # 1. Fetch Historical Target (y)
        # Try live bucket first for dense AR context
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
        
        # If live is empty, fallback to stats (less likely to be helpful for AR tail)
        if df_hist_y.empty:
            print(f"Warning: {live_bucket} empty. Falling back to stats bucket.")
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

        # 2. Fetch Historical Regressors
        reg_hist_bucket = settings['influxdb']['buckets']['regressor_history']
        reg_hist_meas = settings['influxdb']['measurements']['regressor_history']
        regressor_scale = settings['model']['preprocessing'].get('regressor_scale', 1.0)
        regressor_offset = settings['model']['preprocessing'].get('regressor_offset', '0m')
        
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
            # Process History
            df_hist_y = prepare_prophet_dataframe(df_hist_y, freq='15min')
            df_hist_reg = prepare_prophet_dataframe(df_hist_reg, freq='15min')
            
            # Merge
            df_history = pd.merge(df_hist_y, df_hist_reg, on='ds', how='inner')
            
            # Rename y col
            if y_col in df_history.columns:
                df_history.rename(columns={y_col: 'y'}, inplace=True)
            
            # Ensure regressors are present
            for r in regressor_names:
                if r not in df_history.columns:
                     print(f"Warning: Historical regressor {r} missing. Filling with 0.")
                     df_history[r] = 0
                else:
                     df_history[r] = df_history[r].interpolate(method='linear', limit_direction='both')
            
            # Select columns
            df_history_final = df_history.copy()
            df_history_final['ds'] = pd.to_datetime(df_history_final['ds'])
            df_history_final = df_history_final.set_index('ds').resample('15min').mean()
            df_history_final = df_history_final.fillna(0) # IMPORTANT: Fill gaps with 0 for PV stability
            df_history_final = df_history_final.reset_index()
            
            # Select required columns
            df_history_final = df_history_final[['ds', 'y'] + regressor_names].copy()
            
            print(f"Found {len(df_history_final)} historical points after resampling/zero-filling.")
            print(f"DEBUG: History range: {df_history_final['ds'].min()} to {df_history_final['ds'].max()}")
            print(f"DEBUG: Future range: {df_future_final['ds'].min()} to {df_future_final['ds'].max()}")
            
    # Predict Loop (Recursive Chunks)
    print(f"Starting chunked forecasting for {len(df_future_final)} periods (step={n_forecasts})...")
    
    current_history = df_history_final.tail(n_lags).copy()
    future_points = df_future_final.copy()
    predictions = []
    
    # Process in chunks of n_forecasts
    for i in range(0, len(future_points), n_forecasts):
        # 1. Prepare Chunk
        chunk = future_points.iloc[i : i + n_forecasts].copy()
        
        # Ensure regressors are not NaN
        for r in regressor_names:
            if chunk[r].isna().any():
                chunk[r] = chunk[r].interpolate(method='linear', limit_direction='both').fillna(0)

        actual_len = len(chunk)
        chunk['y'] = np.nan
        
        # 2. Pad chunk to exactly n_forecasts if needed (NeuralProphet requirement for multi-step)
        if actual_len < n_forecasts:
            last_ds = chunk['ds'].max()
            freq = '15min'
            pad_len = n_forecasts - actual_len
            padding = pd.DataFrame({
                'ds': pd.date_range(start=last_ds + pd.Timedelta(freq), periods=pad_len, freq=freq),
                'y': np.nan
            })
            for col in regressor_names:
                padding[col] = chunk[col].iloc[-1]
            chunk = pd.concat([chunk, padding], ignore_index=True)

        # 3. Concatenate history + chunk
        step_input = pd.concat([current_history.tail(n_lags), chunk], ignore_index=True)
        step_input['y'] = pd.to_numeric(step_input['y'], errors='coerce')
        
        if i == 0:
             print(f"DEBUG: step_input tail:\n{step_input.tail(5)}")
             print(f"DEBUG: step_input head of chunk:\n{step_input.iloc[n_lags:n_lags+5]}")

        try:
            # 4. Predict multi-step
            step_forecast = model.predict(step_input)
            
            # 5. Extract results using "diagonal" retrieval
            # In NeuralProphet multi-step mode (n_forecasts > 1), without intermediate 'y',
            # the k-th step prediction for row 'n_lags + k - 1' is found in column 'yhat{k}'.
            
            chunk_preds = []
            for k in range(1, actual_len + 1):
                col_name = f'yhat{k}'
                row_idx = n_lags + k - 1
                if col_name in step_forecast.columns and row_idx < len(step_forecast):
                    val = step_forecast.iloc[row_idx][col_name]
                    chunk_preds.append(val)
                else:
                    # Fallback to yhat1 if available (should be captured by the loop logic though)
                    chunk_preds.append(0.0)
            
            # 6. Save results
            res_chunk = future_points.iloc[i : i + actual_len].copy()
            res_chunk['yhat'] = chunk_preds
            predictions.append(res_chunk)
            
            # 7. Update history for NEXT chunk (Recursive transition)
            new_history = res_chunk.copy()
            new_history['y'] = res_chunk['yhat']
            # Select only columns needed for the model (ds, y, and regressors)
            keep_cols = ['ds', 'y'] + [col for col in regressor_names if col in new_history.columns]
            new_history = new_history[keep_cols]
            current_history = pd.concat([current_history, new_history], ignore_index=True)
            
            if (i + actual_len) % n_forecasts == 0 or (i + actual_len) == len(future_points):
                print(f"Predicted {min(i + actual_len, len(future_points))}/{len(future_points)} steps...")
                
        except Exception as e:
            print(f"Error at step {i}: {e}")
            import traceback
            traceback.print_exc()
            break
            
    if not predictions:
        print("Error: No predictions generated in loop.")
        return
        
    forecast = pd.concat(predictions, ignore_index=True)
    print(f"Generated {len(forecast)} forecast points.")
    
    # Debug: check columns
    # print(f"DEBUG: Forecast columns: {forecast.columns.tolist()}")

    # Rename yhat1 to yhat
    if 'yhat1' in forecast.columns and 'yhat' not in forecast.columns:
        forecast.rename(columns={'yhat1': 'yhat'}, inplace=True)
    
    # Rename yhat1 to yhat
    if 'yhat1' in forecast.columns and 'yhat' not in forecast.columns:
        forecast.rename(columns={'yhat1': 'yhat'}, inplace=True)
    
    # Extract only the future part
    # We want rows where ds > last_history_date OR simply rows that were in df_future_final
    # Simpler: Filter by ds > now()
    # Or strict: Only keep points >= min(df_future_final['ds'])
    
    # Note: postprocess_forecast might expect specific columns
    # We slice forecast to match the requested future range
    min_future_ds = df_future_final['ds'].min()
    forecast = forecast[forecast['ds'] >= min_future_ds].copy()
    
    # Postprocess
    forecast = postprocess_forecast(forecast)
    
    # Prepare to write back to InfluxDB
    forecast_offset = settings['model']['preprocessing'].get('forecast_offset', '0m')
    if forecast_offset != "0m":
        try:
            print(f"Applying forecast offset: {forecast_offset}")
            forecast['ds'] = forecast['ds'] + pd.Timedelta(forecast_offset)
        except Exception as e:
            print(f"Warning: Could not apply forecast offset '{forecast_offset}': {e}")

    print(f"Target Measurement: '{settings['influxdb']['measurements']['forecast']}'")
    print(f"Writing forecast to {settings['influxdb']['buckets']['target_forecast']}...")
    
    forecast_to_write = forecast[['ds', 'yhat']].copy()
    
    # Filter: Set values <= night_threshold to 0
    threshold = settings['model'].get('tuning', {}).get('night_threshold', 50)
    print(f"Applying output filter: values <= {threshold} set to 0")
    forecast_to_write.loc[forecast_to_write['yhat'] <= threshold, 'yhat'] = 0
    
    forecast_to_write.rename(columns={'ds': 'time', 'yhat': settings['influxdb']['fields']['forecast']}, inplace=True)
    forecast_to_write.set_index('time', inplace=True)
    
    # Write
    db.write_dataframe(
        df=forecast_to_write,
        bucket=settings['influxdb']['buckets']['target_forecast'],
        measurement=settings['influxdb']['measurements']['forecast']
    )
    
    print("Forecast complete.")

if __name__ == "__main__":
    run_forecast()
