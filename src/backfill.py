import os
import pickle
import pandas as pd
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import settings
from src.db import InfluxDBWrapper
from src.preprocess import postprocess_forecast, prepare_prophet_dataframe

def backfill_forecast():
    print("Starting forecast backfill...")
    
    # 1. Load Model
    model_path = settings['model']['model_path']
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Run training first.")
        return
        
    print(f"Loading model from {model_path}...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    db = InfluxDBWrapper()
    
    # 2. Fetch Historical Regressor
    days = 14 # Backfill last 14 days
    range_start = f"-{days}d"
    
    print(f"Fetching historical regressor data for last {days} days...")
    regressor_scale = settings['preprocessing'].get('regressor_scale', 1.0)
    regressor_offset = settings['preprocessing'].get('regressor_offset', '0m')
    
    query = f'''
    from(bucket: "{settings['buckets']['b_regressor_history']}")
      |> range(start: {range_start})
      |> filter(fn: (r) => r["_measurement"] == "{settings['measurements']['m_regressor_history']}")
      |> filter(fn: (r) => r["_field"] == "{settings['fields']['f_regressor_history']}")
      |> map(fn: (r) => ({{ r with _value: r._value * {regressor_scale} }}))
      |> timeShift(duration: {regressor_offset})
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    df = db.query_dataframe(query)
    
    if df.empty:
        print("No historical regressor data found.")
        return

    # Preprocess
    regressor_name = settings['measurements']['m_regressor_history']
    df.rename(columns={settings['fields']['f_regressor_history']: regressor_name}, inplace=True)
    df = prepare_prophet_dataframe(df, freq='30min')
    df[regressor_name] = df[regressor_name].interpolate(method='linear', limit_direction='both')
    
    # Create Prediction DataFrame (ds + regressor)
    future = df[['ds', regressor_name]].copy()
    
    print(f"Generating backfill for {len(future)} points...")
    forecast = model.predict(future)
    forecast = postprocess_forecast(forecast)
    
    # Apply Offset
    forecast_offset = settings['preprocessing'].get('forecast_offset', '0m')
    if forecast_offset != "0m":
         forecast['ds'] = forecast['ds'] + pd.Timedelta(forecast_offset)

    # Write to DB
    print(f"Writing backfill to {settings['buckets']['b_target_forecast']}...")
    to_write = forecast[['ds', 'yhat']].copy()
    to_write.rename(columns={'ds': 'time', 'yhat': settings['fields']['f_forecast']}, inplace=True)
    to_write.set_index('time', inplace=True)
    
    db.write_dataframe(
        df=to_write,
        bucket=settings['buckets']['b_target_forecast'],
        measurement=settings['measurements']['m_forecast']
    )
    print("Backfill complete.")

if __name__ == "__main__":
    backfill_forecast()
