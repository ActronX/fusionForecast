import pandas as pd
import numpy as np
import sys
import os

# Ensure src can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.db import InfluxDBWrapper
from src.config import settings
from src.preprocess import preprocess_data, prepare_prophet_dataframe

def diagnose_data():
    print("----------------------------------------------------------------")
    print("               FusionForecast Data Diagnosis                    ")
    print("----------------------------------------------------------------")

    db = InfluxDBWrapper()
    
    # 1. Fetch Sample Data (Last 14 days)
    # We want to see the raw relationship between the Regressor (Solarcast) and Produced (Actual)
    days = 14
    range_start = f"-{days}d"
    
    print(f"Fetching last {days} days of data for analysis...")

    # Fetch Produced
    scale_produced = settings['preprocessing']['produced_scale']
    query_produced = f'''
    from(bucket: "{settings['buckets']['b_history_produced']}")
      |> range(start: {range_start})
      |> filter(fn: (r) => r["_measurement"] == "{settings['measurements']['m_produced']}")
      |> filter(fn: (r) => r["_field"] == "{settings['fields']['f_produced']}")
      |> map(fn: (r) => ({{ r with _value: r._value * {scale_produced} }}))
      |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    df_produced = db.query_dataframe(query_produced)
    
    # Fetch Regressor
    scale_regressor = settings['preprocessing']['regressor_scale']
    query_regressor = f'''
    from(bucket: "{settings['buckets']['b_regressor_history']}")
      |> range(start: {range_start})
      |> filter(fn: (r) => r["_measurement"] == "{settings['measurements']['m_regressor_history']}")
      |> filter(fn: (r) => r["_field"] == "{settings['fields']['f_regressor_history']}")
      |> map(fn: (r) => ({{ r with _value: r._value * {scale_regressor} }}))
      |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    df_regressor = db.query_dataframe(query_regressor)

    if df_produced.empty or df_regressor.empty:
        print("Error: Could not fetch data for diagnosis.")
        return

    # Prepare for merge
    prod_col = settings['fields']['f_produced']
    reg_col = settings['fields']['f_regressor_history']
    
    df_produced = df_produced[['_time', prod_col]].rename(columns={'_time': 'ds', prod_col: 'actual'})
    df_regressor = df_regressor[['_time', reg_col]].rename(columns={'_time': 'ds', reg_col: 'forecast'})
    
    # Timezone agnostic
    df_produced['ds'] = pd.to_datetime(df_produced['ds']).dt.tz_localize(None)
    df_regressor['ds'] = pd.to_datetime(df_regressor['ds']).dt.tz_localize(None)
    
    # Merge
    df = pd.merge(df_produced, df_regressor, on='ds', how='inner')
    
    if df.empty:
        print("Error: No overlap between Actual and Regressor data.")
        return

    # 2. Analyze Scale
    print("\n[Scale Analysis]")
    mean_actual = df['actual'].mean()
    mean_forecast = df['forecast'].mean()
    max_actual = df['actual'].max()
    max_forecast = df['forecast'].max()
    
    print(f"Actual   | Mean: {mean_actual:10.2f} | Max: {max_actual:10.2f}")
    print(f"Forecast | Mean: {mean_forecast:10.2f} | Max: {max_forecast:10.2f}")
    
    ratio = mean_actual / mean_forecast if mean_forecast != 0 else 0
    print(f"Ratio (Actual/Forecast): {ratio:.4f}")
    
    if ratio > 100 or ratio < 0.01:
        print(">> CRITICAL: Massive scale difference detected! Check 'scale' settings (kW vs W).")
    elif ratio > 2 or ratio < 0.5:
        print(">> WARNING: Significant scale difference. Forecast might be under/over estimating.")

    # 3. Analyze Time Shift
    print("\n[Time Shift Analysis]")
    # Find cross correlation
    # Resample to ensure regular interval for shift calculation
    df_idx = df.set_index('ds').resample('1h').mean().fillna(0)
    
    shifts = range(-5, 6) # check -5 to +5 hours
    best_corr = -1
    best_shift = 0
    
    print(f"{'Shift (Hours)':<15} | {'Correlation':<15}")
    print("-" * 35)
    
    for shift in shifts:
        shifted_forecast = df_idx['forecast'].shift(shift)
        corr = df_idx['actual'].corr(shifted_forecast)
        print(f"{shift:<15} | {corr:.4f}")
        
        if corr > best_corr:
            best_corr = corr
            best_shift = shift
            
    print("-" * 35)
    print(f"Best Correlation: {best_corr:.4f} at shift {best_shift} hours.")
    
    if abs(best_shift) > 0:
        print(f">> WARNING: Optimal alignment requires shifting forecast by {best_shift} hours.")
        print(f"   If shift is POSITIVE (+X), Forecast is EARLY (needs to be delayed).")
        print(f"   If shift is NEGATIVE (-X), Forecast is LATE (needs to be advanced).")
    else:
        print(">> Time alignment looks good.")

    print("----------------------------------------------------------------")

if __name__ == "__main__":
    diagnose_data()
