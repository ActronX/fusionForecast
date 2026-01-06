import os
import pickle
import pandas as pd
from src.config import settings
from src.db import InfluxDBWrapper
from src.preprocess import postprocess_forecast

def run_consumption_forecast():
    print("Starting consumption forecast pipeline...")
    
    # 1. Load Model
    model_path = settings['model']['model_path_consumption']
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run consumption training first.")
        return
        
    print(f"Loading model from {model_path}...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    # Initialize DB
    db = InfluxDBWrapper()
    
    # Forecast Horizon
    forecast_days = settings['forecast_parameters']['forecast_days']
    
    print(f"Forecasting for next {forecast_days} days...")
    
    # Create Future DataFrame
    # Since we have no regressors, we can just use make_future_dataframe
    future = model.make_future_dataframe(periods=forecast_days * 48, freq='30min') # 48 * 30min = 24h
    
    # Filter for future only (optional, but clean)
    # make_future_dataframe includes history by default
    last_history_date = model.history['ds'].max()
    future = future[future['ds'] > last_history_date]
    
    if future.empty:
        print("Error: Future dataframe is empty.")
        return

    print(f"Forecasting for {len(future)} points...")
    
    # Predict
    forecast = model.predict(future)
    
    # Postprocess
    forecast = postprocess_forecast(forecast)
    
    # Prepare to write back to InfluxDB
    # Apply Forecast Offset
    # Reusing forecast_offset or check for specific consumption one
    forecast_offset = settings['preprocessing'].get('consumption_forecast_offset', settings['preprocessing'].get('forecast_offset', '0m'))
    
    if forecast_offset != "0m":
        try:
            print(f"Applying forecast offset: {forecast_offset}")
            forecast['ds'] = forecast['ds'] + pd.Timedelta(forecast_offset)
        except Exception as e:
            print(f"Warning: Could not apply forecast offset '{forecast_offset}': {e}")

    target_bucket = settings['buckets']['b_target_forecast']
    target_measurement = settings['measurements']['m_forecast']
    target_field = settings['fields']['f_consumption_forecast']

    print(f"Writing forecast to:")
    print(f"  > Bucket:      {target_bucket}")
    print(f"  > Measurement: {target_measurement}")
    print(f"  > Field:       {target_field}")

    # Convert to list of Points or write DF directly
    forecast_to_write = forecast[['ds', 'yhat']].copy()
    forecast_to_write.rename(columns={'ds': 'time', 'yhat': target_field}, inplace=True)
    forecast_to_write.set_index('time', inplace=True)
    
    db.write_dataframe(
        df=forecast_to_write,
        bucket=target_bucket,
        measurement=target_measurement
    )
    
    print("Consumption forecast complete.")

if __name__ == "__main__":
    run_consumption_forecast()
