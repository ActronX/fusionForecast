
import os
import pickle
import pandas as pd
from src.config import settings
from src.db import InfluxDBWrapper
from src.preprocess import truncate_time_column, postprocess_forecast, prepare_prophet_dataframe

def run_forecast():
    print("Starting forecast pipeline...")
    
    # 1. Load Model
    model_path = settings['model']['path']
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run training first.")
        return
        
    print(f"Loading model from {model_path}...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    # Initialize DB
    db = InfluxDBWrapper()
    
    # Forecast Horizon
    forecast_days = settings['model']['forecast_days']
    # Start looking for regressor data from now (or end of history)
    # We just need future regressor data. 
    # Usually we query from now until now + forecast_days
    
    # Flux query for future regressor
    # Using date.add to define relative future time
    
    print(f"Fetching regressor data for next {forecast_days} days...")
    regressor_offset = settings['model']['preprocessing'].get('regressor_offset', '0m')
    regressor_scale = settings['model']['preprocessing'].get('regressor_scale', 1.0)
    
    # Check if we should use Perez POA / Effective
    use_pvlib = settings['model'].get('prophet', {}).get('use_pvlib', False)
    
    if use_pvlib:
        regressor_fields = [
            settings['influxdb']['fields'].get('effective_irradiance', 'effective_irradiance'),
            settings['influxdb']['fields'].get('temp_cell', 'temperature_cell')
        ]
    else:
        regressor_fields = [settings['influxdb']['fields']['regressor_future']]
    
    print(f"Using regressor fields: {regressor_fields}")
    
    # Filter for all requested regressor fields
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
    df_regressor = db.query_dataframe(query_regressor)
    
    if df_regressor.empty:
        print("Error: No future regressor data found. Cannot forecast without 'solarcast'.")
        print(f"  > Bucket: {settings['influxdb']['buckets']['regressor_future']}")
        print(f"  > Measurement: {settings['influxdb']['measurements']['regressor_future']}")
        print(f"  > Fields: {regressor_fields}")
        print("  > Possible causes: incorrect names locally or missing future data in DB.")
        return

    # Preprocess Regressor
    # Standardize regressor using helper (consistent 15min freq)
    df_regressor = prepare_prophet_dataframe(df_regressor, freq='15min')
    
    # Interpolate and rename each regressor field
    regressor_names = []
    for field in regressor_fields:
        reg_name = field
        df_regressor[reg_name] = df_regressor[reg_name].interpolate(method='linear', limit_direction='both')
        regressor_names.append(reg_name)
    
    # Check data sufficiency (User requested 50% threshold)
    duration = df_regressor['ds'].max() - df_regressor['ds'].min()
    available_days = duration.total_seconds() / (24 * 3600)
    
    if available_days < (forecast_days * 0.5):
         print(f"Error: Insufficient future regressor data.")
         print(f"  > Requested Horizon: {forecast_days} days")
         print(f"  > Available: {available_days:.2f} days")
         print("  > Threshold: 50% of requested horizon required.")
         print("  > Hint: Please run 'fetch_future_weather' to fetch more forecast data.")
         return

    # Create Future DataFrame
    future = df_regressor[['ds'] + regressor_names].copy()
    
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
    forecast_offset = settings['model']['preprocessing'].get('forecast_offset', '0m')
    if forecast_offset != "0m":
        try:
            print(f"Applying forecast offset: {forecast_offset}")
            forecast['ds'] = forecast['ds'] + pd.Timedelta(forecast_offset)
        except Exception as e:
            print(f"Warning: Could not apply forecast offset '{forecast_offset}': {e}")

    print(f"Target Measurement: '{settings['influxdb']['measurements']['forecast']}'")
    print(f"Target Field: '{settings['influxdb']['fields']['forecast']}'")
    print(f"Writing forecast to {settings['influxdb']['buckets']['target_forecast']}...")
    
    # Convert to list of Points or write DF directly
    # To write DF, we need to set the index to time
    forecast_to_write = forecast[['ds', 'yhat']].copy()
    
    # Filter: Set values <= night_threshold to 0
    # This prevents writing small noise values to DB
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
