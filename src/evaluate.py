import pandas as pd
import numpy as np
from math import sqrt
import sys
from datetime import datetime, timedelta

# Ensure src can be imported
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.db import InfluxDBWrapper
from src.config import settings

def calculate_metrics():
    print("----------------------------------------------------------------")
    print("               FusionForecast Evaluation Metrics                ")
    print("----------------------------------------------------------------")

    # 1. Initialize DB
    print("Connecting to InfluxDB...")
    try:
        db = InfluxDBWrapper()
    except Exception as e:
        print(f"Failed to connect to DB: {e}")
        return

    # 2. Define Time Range (default: last 7 days to now)
    # We want to evaluate the quality of forecasts made for the past.
    eval_days = 90
    range_start = f"-{eval_days}d"
    range_stop = "now()" # Data up to now

    print(f"Evaluation Period: Last {eval_days} days")

    # 3. Fetch Actual Data (Produced)
    print(f"Fetching actual data from bucket '{settings['buckets']['b_history_produced']}'...")
    produced_scale = settings['preprocessing'].get('produced_scale', 1.0)
    
    # Using aggregateWindow to ensure alignment/cleanliness if needed, but let's stick to raw or simple mean
    # matching the train.py logic roughly but for evaluation we want to compare what happened vs what was predicted.
    # Assuming forecasts are 30min or 1h? train.py uses 30min.
    
    query_actual = f'''
    from(bucket: "{settings['buckets']['b_history_produced']}")
      |> range(start: {range_start}, stop: {range_stop})
      |> filter(fn: (r) => r["_measurement"] == "{settings['measurements']['m_produced']}")
      |> filter(fn: (r) => r["_field"] == "{settings['fields']['f_produced']}")
      |> map(fn: (r) => ({{ r with _value: r._value * {produced_scale} }}))
      |> aggregateWindow(every: 30m, fn: mean, createEmpty: false)
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    df_actual = db.query_dataframe(query_actual)

    if df_actual is None or df_actual.empty:
        print(" [!] No actual data found in the specified range.")
        return

    # Rename column for merge
    actual_col = settings['fields']['f_produced']
    df_actual = df_actual[['_time', actual_col]].rename(columns={'_time': 'ds', actual_col: 'y_actual'})
    df_actual['ds'] = pd.to_datetime(df_actual['ds']).dt.tz_convert(None) # Remove timezone for simple merge if needed, or keep? 
    # Usually pandas merge works best if both are tz-naive or both tz-aware. Let's make them tz-naive (UTC or local) to be safe or rely on pandas.
    # Influx returns UTC. Let's keep it as is first, if merge fails, we adjust.
    # Actually, to be safe, let's normalize to UTC naive
    df_actual['ds'] = pd.to_datetime(df_actual['ds']).dt.tz_localize(None)

    # 4. Fetch Forecast Data
    print(f"Fetching forecast data from bucket '{settings['buckets']['b_target_forecast']}'...")
    
    query_forecast = f'''
    from(bucket: "{settings['buckets']['b_target_forecast']}")
      |> range(start: {range_start}, stop: {range_stop})
      |> filter(fn: (r) => r["_measurement"] == "{settings['measurements']['m_forecast']}")
      |> filter(fn: (r) => r["_field"] == "{settings['fields']['f_forecast']}")
      |> aggregateWindow(every: 30m, fn: mean, createEmpty: false)
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    df_forecast = db.query_dataframe(query_forecast)

    if df_forecast is None or df_forecast.empty:
        print(" [!] No forecast data found in the specified range. Cannot evaluate.")
        return

    forecast_col = settings['fields']['f_forecast']
    df_forecast = df_forecast[['_time', forecast_col]].rename(columns={'_time': 'ds', forecast_col: 'y_forecast'})
    df_forecast['ds'] = pd.to_datetime(df_forecast['ds']).dt.tz_localize(None)

    # 5. Merge Data
    print("Aligning data...")
    df_merged = pd.merge(df_actual, df_forecast, on='ds', how='inner')
    
    if df_merged.empty:
        print(" [!] No overlap found between actuals and forecasts.")
        print(f"     Actuals range: {df_actual['ds'].min()} to {df_actual['ds'].max()}")
        print(f"     Forecast range: {df_forecast['ds'].min()} to {df_forecast['ds'].max()}")
        return

    # 6. Calculate Metrics
    y_true = df_merged['y_actual']
    y_pred = df_merged['y_forecast']
    
    # Filter by threshold for all metrics if desired
    threshold = settings.get('prophet', {}).get('tuning', {}).get('night_threshold', 50)
    valid_mask = y_true > threshold

    if valid_mask.sum() > 0:
        # MAE
        mae = np.mean(np.abs(y_true[valid_mask] - y_pred[valid_mask]))
        # RMSE
        rmse = sqrt(np.mean((y_true[valid_mask] - y_pred[valid_mask]) ** 2))
    else:
        mae = np.nan
        rmse = np.nan
    
    # MAPE (Handle division by zero and low production)
    # Filter out low values in y_true to avoid noise (threshold configurable)
    threshold = settings.get('prophet', {}).get('tuning', {}).get('night_threshold', 50)
    valid_mask = y_true > threshold
    if valid_mask.sum() > 0:
        mape = np.mean(np.abs((y_true[valid_mask] - y_pred[valid_mask]) / y_true[valid_mask])) * 100
        accuracy = 100 - mape
    else:
        mape = np.nan
        accuracy = np.nan

    # WMAPE (Weighted MAPE) - Better for solar/wind with zeros
    # Sum(Abs(Error)) / Sum(Actual)
    if np.sum(y_true) != 0:
        wmape = (np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)) * 100
        wmape_accuracy = 100 - wmape
    else:
        wmape = np.nan
        wmape_accuracy = np.nan

    # Coverage
    # Expected points: (Max Time - Min Time) / Frequency.
    # Or simply: how many points did we actually find in this window?
    # Let's define Coverage as: Points Found / Theoretical Points in Range of (Min(ds), Max(ds))
    if len(df_merged) > 1:
        time_span = df_merged['ds'].max() - df_merged['ds'].min()
        total_seconds = time_span.total_seconds()
        # Assuming 30min intervals
        expected_points = (total_seconds / 1800) + 1
        coverage = (len(df_merged) / expected_points) * 100
    else:
        coverage = 100.0 if len(df_merged) == 1 else 0.0

    # 7. Print Results
    print("\n----------------------------------------------------------------")
    print("RESULTS")
    print("----------------------------------------------------------------")
    print(f"{'Metric':<20} | {'Value':<15}")
    print(f"{'-'*20}-|-{'-'*15}")
    print(f"{'Data Points':<20} | {len(df_merged):<15}")
    print(f"{'Coverage (est)':<20} | {coverage:.2f}%")
    print(f"{'MAE':<20} | {mae:.4f}")
    print(f"{'RMSE':<20} | {rmse:.4f}")
    
    if not np.isnan(mape):
        print(f"{'MAPE':<20} | {mape:.2f}%")
        print(f"{'Accuracy (1-MAPE)':<20} | {accuracy:.2f}%")
    else:
        print(f"{'MAPE':<20} | N/A (Zeros in actuals)")
        print(f"{'Accuracy':<20} | N/A")

    if not np.isnan(wmape):
        print(f"{'WMAPE':<20} | {wmape:.2f}%")
        print(f"{'Accuracy (1-WMAPE)':<20} | {wmape_accuracy:.2f}%")
    else:
        print(f"{'WMAPE':<20} | N/A (Sum actuals is 0)")

    print("\n----------------------------------------------------------------")
    # Interpretation hint
    if not np.isnan(accuracy):
        if accuracy > 80:
             print(">> Excellent forecast quality.")
        elif accuracy > 60:
             print(">> Good forecast quality.")
        else:
             print(">> Forecast quality needs improvement.")
    print("----------------------------------------------------------------")

if __name__ == "__main__":
    calculate_metrics()
