
import os
import pickle
import pandas as pd
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
    print(f"Model n_lags: {n_lags}")

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
        target_field = settings['influxdb']['fields']['produced']
        target_meas = settings['influxdb']['measurements']['produced']
        target_bucket = settings['influxdb']['buckets']['history_produced']
        prod_scale = settings['model']['preprocessing'].get('produced_scale', 1.0)
        prod_offset = settings['model']['preprocessing'].get('produced_offset', '0m')

        query_history_y = f'''
        import "date"
        from(bucket: "{target_bucket}")
          |> range(start: date.sub(d: {int(fetch_hours)}h, from: now()), stop: now())
          |> filter(fn: (r) => r["_measurement"] == "{target_meas}")
          |> filter(fn: (r) => r["_field"] == "{target_field}")
          |> map(fn: (r) => ({{ r with _value: r._value * {prod_scale} }}))
          |> timeShift(duration: {prod_offset})
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        df_hist_y = db.query_dataframe(query_history_y)
        
        # 2. Fetch Historical Regressors
        # Using regressor_history bucket
        reg_hist_bucket = settings['influxdb']['buckets']['regressor_history']
        reg_hist_meas = settings['influxdb']['measurements']['regressor_history']
        # Same fields as future
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
            if target_field in df_history.columns:
                df_history.rename(columns={target_field: 'y'}, inplace=True)
            
            # Ensure regressors are present
            for r in regressor_names:
                if r not in df_history.columns:
                     print(f"Warning: Historical regressor {r} missing. Filling with 0.")
                     df_history[r] = 0
                else:
                     df_history[r] = df_history[r].interpolate(method='linear', limit_direction='both')
            
            # Select columns
            df_history_final = df_history[['ds', 'y'] + regressor_names].copy()
            
            print(f"Found {len(df_history_final)} historical points.")
            
            # Combine: History + Future
            # Future has no 'y', so we leave it NaN (NP handles this for prediction)
            df_future_final['y'] = pd.NA
            
            df_input = pd.concat([df_history_final, df_future_final], ignore_index=True)
            df_input = df_input.sort_values('ds').reset_index(drop=True)
            
            # Fill missing regressors if any (due to join gaps)
            for r in regressor_names:
                df_input[r] = df_input[r].interpolate(method='linear', limit_direction='both')
                
        else:
            print("Warning: Could not fetch sufficient history for AR. Forecasting might be degraded.")

    print(f"Forecasting for {len(df_input)} points (History + Future)...")
    
    # Predict
    forecast = model.predict(df_input)
    
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
