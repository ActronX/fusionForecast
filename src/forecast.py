
import os
import pickle
import pandas as pd
from src.config import settings
from src.db import InfluxDBWrapper
from src.preprocess import truncate_time_column, postprocess_forecast, prepare_prophet_dataframe

def run_forecast():
    print("Starting forecast pipeline...")
    
    # 1. Load Model
    model_path = settings['model']['model_path']
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run training first.")
        return
        
    print(f"Loading model from {model_path}...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    # Initialize DB
    db = InfluxDBWrapper()
    
    # Forecast Horizon
    forecast_days = settings['forecast_parameters']['forecast_days']
    # Start looking for regressor data from now (or end of history)
    # We just need future regressor data. 
    # Usually we query from now until now + forecast_days
    
    # Flux query for future regressor
    # Using date.add to define relative future time
    
    print(f"Fetching regressor data for next {forecast_days} days...")
    regressor_offset = settings['preprocessing'].get('regressor_offset', '0m')
    regressor_scale = settings['preprocessing'].get('regressor_scale', 1.0)
    query_regressor = f'''
    import "date"
    from(bucket: "{settings['buckets']['b_regressor_future']}")
      |> range(start: now(), stop: date.add(d: {forecast_days}d, to: now()))
      |> filter(fn: (r) => r["_measurement"] == "{settings['measurements']['m_regressor_future']}")
      |> filter(fn: (r) => r["_field"] == "{settings['fields']['f_regressor_future']}")
      |> map(fn: (r) => ({{ r with _value: r._value * {regressor_scale} }}))
      |> timeShift(duration: {regressor_offset})
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    df_regressor = db.query_dataframe(query_regressor)
    
    if df_regressor.empty:
        print("Error: No future regressor data found. Cannot forecast without 'solarcast'.")
        print(f"  > Bucket: {settings['buckets']['b_regressor_future']}")
        print(f"  > Measurement: {settings['measurements']['m_regressor_future']}")
        print(f"  > Field: {settings['fields']['f_regressor_future']}")
        print("  > Possible causes: incorrect names locally or missing future data in DB.")
        return

    # Preprocess Regressor
    # Rename value column to match regressor name used in training
    regressor_name = settings['measurements']['m_regressor_history'] 
    df_regressor.rename(columns={settings['fields']['f_regressor_future']: regressor_name}, inplace=True)
    
    # Standardize regressor using helper (consistent 15min freq)
    df_regressor = prepare_prophet_dataframe(df_regressor, freq='15min')
    
    # Interpolate missing values (e.g. if upsampling from 1h to 30min)
    df_regressor[regressor_name] = df_regressor[regressor_name].interpolate(method='linear', limit_direction='both')
    
    # Create Future DataFrame
    # Ideally we use the regressor's index as the future dataframe index
    # Prophet's make_future_dataframe is good but we already have the timestamps from the regressor
    future = df_regressor[['ds', regressor_name]].copy()
    
    # Check if we have enough data
    if future.empty:
        print("Error: Future dataframe is empty.")
        return

    print(f"Forecasting for {len(future)} points...")
    
    # Predict
    forecast = model.predict(future)
    
    # Postprocess
    forecast = postprocess_forecast(forecast)
    
    # Prepare to write back to InfluxDB
    # We need 'ds' (time) and 'yhat' (value)
    # Write to b_target_forecast, m_forecast, f_forecast
    
    # Apply Forecast Offset
    forecast_offset = settings['preprocessing'].get('forecast_offset', '0m')
    if forecast_offset != "0m":
        try:
            print(f"Applying forecast offset: {forecast_offset}")
            forecast['ds'] = forecast['ds'] + pd.Timedelta(forecast_offset)
        except Exception as e:
            print(f"Warning: Could not apply forecast offset '{forecast_offset}': {e}")

    print(f"Target Measurement: '{settings['measurements']['m_forecast']}'")
    print(f"Target Field: '{settings['fields']['f_forecast']}'")
    print(f"Writing forecast to {settings['buckets']['b_target_forecast']}...")
    
    # Convert to list of Points or write DF directly
    # To write DF, we need to set the index to time
    forecast_to_write = forecast[['ds', 'yhat']].copy()
    
    # Filter: Set values <= mape_threshold to 0
    # This prevents writing small noise values to DB
    threshold = settings.get('prophet', {}).get('tuning', {}).get('night_threshold', 50)
    print(f"Applying output filter: values <= {threshold} set to 0")
    forecast_to_write.loc[forecast_to_write['yhat'] <= threshold, 'yhat'] = 0
    
    forecast_to_write.rename(columns={'ds': 'time', 'yhat': settings['fields']['f_forecast']}, inplace=True)
    forecast_to_write.set_index('time', inplace=True)
    
    # Write
    db.write_dataframe(
        df=forecast_to_write,
        bucket=settings['buckets']['b_target_forecast'],
        measurement=settings['measurements']['m_forecast']
    )
    
    print("Forecast complete.")

if __name__ == "__main__":
    run_forecast()
