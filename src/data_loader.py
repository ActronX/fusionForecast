"""
Shared data loading functionality for training and tuning.
Eliminates code duplication between train.py and tune.py.
"""

import pandas as pd
import sys
from src.db import InfluxDBWrapper
from src.config import settings
from src.preprocess import preprocess_data, prepare_prophet_dataframe


def fetch_training_data(verbose: bool = True):
    """
    Fetches and prepares training data from InfluxDB.
    
    This function is used by both train.py and tune.py to ensure
    consistent data loading and preprocessing.
    
    Args:
        verbose: If True, print progress messages.
    
    Returns:
        tuple: (df_prophet, regressor_names) where df_prophet is the merged
               DataFrame ready for Prophet, and regressor_names is a list
               of regressor column names.
        None: If data fetching fails.
    """
    if verbose:
        print("Fetching training data...")
    
    db = InfluxDBWrapper()
    training_days = settings['model']['training_days']
    range_start = f"-{training_days}d"

    # 1. Fetch Produced Data (Target 'y')
    if verbose:
        print(f"Fetching produced data from {settings['influxdb']['buckets']['history_produced']}...")
    
    produced_scale = settings['model']['preprocessing'].get('produced_scale', 1.0)
    produced_offset = settings['model']['preprocessing'].get('produced_offset', '0m')
    
    query_produced = f'''
    from(bucket: "{settings['influxdb']['buckets']['history_produced']}")
      |> range(start: {range_start})
      |> filter(fn: (r) => r["_measurement"] == "{settings['influxdb']['measurements']['produced']}")
      |> filter(fn: (r) => r["_field"] == "{settings['influxdb']['fields']['produced']}")
      |> map(fn: (r) => ({{ r with _value: r._value * {produced_scale} }}))
      |> timeShift(duration: {produced_offset})
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    df_produced = db.query_dataframe(query_produced)
    
    if df_produced.empty:
        print(f"Error: No produced data found for training.")
        print(f"  > Bucket: {settings['influxdb']['buckets']['history_produced']}")
        print(f"  > Measurement: {settings['influxdb']['measurements']['produced']}")
        print(f"  > Field: {settings['influxdb']['fields']['produced']}")
        print("  > Possible causes: Data missing in time range, or incorrect measurement/field names.")
        return None

    # Preprocess Produced
    df_prophet = preprocess_data(df_produced, value_column=settings['influxdb']['fields']['produced'], is_prophet_input=True)
    df_prophet = prepare_prophet_dataframe(df_prophet, freq='15min')

    # 2. Fetch Regressor Data (History)
    if verbose:
        print(f"Fetching regressor data from {settings['influxdb']['buckets']['regressor_history']}...")
    
    regressor_offset = settings['model']['preprocessing'].get('regressor_offset', '0m')
    regressor_scale = settings['model']['preprocessing'].get('regressor_scale', 1.0)
    
    # Handle list or string for compatibility
    reg_config = settings['influxdb']['fields']['regressor_history']
    if isinstance(reg_config, list):
        regressor_fields = reg_config
    else:
        regressor_fields = [reg_config]
    
    if verbose:
        print(f"Using regressor fields: {regressor_fields}")
    
    # Filter for all requested regressor fields
    regressor_filter = " or ".join([f'r["_field"] == "{f}"' for f in regressor_fields])
    
    query_regressor = f'''
    from(bucket: "{settings['influxdb']['buckets']['regressor_history']}")
      |> range(start: {range_start})
      |> filter(fn: (r) => r["_measurement"] == "{settings['influxdb']['measurements']['regressor_history']}")
      |> filter(fn: (r) => {regressor_filter})
      |> map(fn: (r) => ({{ r with _value: r._value * {regressor_scale} }}))
      |> timeShift(duration: {regressor_offset})
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    df_regressor = db.query_dataframe(query_regressor)
    
    if df_regressor.empty:
        print("Error: No regressor data found.")
        print(f"  > Bucket: {settings['influxdb']['buckets']['regressor_history']}")
        print(f"  > Measurement: {settings['influxdb']['measurements']['regressor_history']}")
        print(f"  > Fields: {regressor_fields}")
        return None

    # Preprocess Regressor
    df_regressor = prepare_prophet_dataframe(df_regressor, freq='15min')
    
    # Interpolate regressor fields
    regressor_names = []
    for field in regressor_fields:
        if field not in df_regressor.columns:
            print(f"Warning: Regressor field '{field}' missing in fetched data. Filling with 0.")
            df_regressor[field] = 0.0
            
        df_regressor[field] = df_regressor[field].interpolate(method='linear', limit_direction='both')
        regressor_names.append(field)

    # 3. Merge on 'ds'
    df_prophet = pd.merge(df_prophet, df_regressor[['ds'] + regressor_names], on='ds', how='inner')
    df_prophet.dropna(inplace=True)
    
    if verbose:
        print(f"Training data shape after merge: {df_prophet.shape}")
    
    if df_prophet.empty:
        print("Error: Training data empty after merging regressor. Check time alignment.")
        return None
    
    # 4. Fetch lagged regressor data (e.g., Production_W for real-time correction)
    lagged_reg_config = settings['model']['neuralprophet'].get('lagged_regressors', {})
    if lagged_reg_config:
        if verbose:
            print(f"Fetching lagged regressor data from {settings['influxdb']['buckets']['live']}...")
        
        # Fetch Production_W from live bucket
        if 'Production_W' in lagged_reg_config:
            query_production = f'''
            from(bucket: "{settings['influxdb']['buckets']['live']}")
              |> range(start: {range_start})
              |> filter(fn: (r) => r["_measurement"] == "{settings['influxdb']['measurements']['live']}")
              |> filter(fn: (r) => r["_field"] == "{settings['influxdb']['fields']['live']}")
              |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''
            df_production = db.query_dataframe(query_production)
            
            if not df_production.empty:
                df_production = prepare_prophet_dataframe(df_production, freq='15min')
                df_production.rename(columns={settings['influxdb']['fields']['live']: 'Production_W'}, inplace=True)
                
                # Merge with main dataframe
                df_prophet = pd.merge(df_prophet, df_production[['ds', 'Production_W']], on='ds', how='left')
                df_prophet['Production_W'] = df_prophet['Production_W'].fillna(0)  # Fill missing with 0
                
                if verbose:
                    print(f"Added Production_W column for lagged regressor (shape: {df_prophet.shape})")
            else:
                print("Warning: No Production_W data found. Lagged regressor will be skipped during training.")
                df_prophet['Production_W'] = 0.0  # Add column with zeros as fallback

    return df_prophet, regressor_names



def validate_data_sufficiency(df: pd.DataFrame, required_days: int, tolerance: float = 0.9) -> bool:
    """
    Validates that the DataFrame contains sufficient historical data.
    
    Args:
        df: DataFrame with 'ds' column.
        required_days: Number of days required.
        tolerance: Minimum fraction of required days (default 0.9 = 90%).
    
    Returns:
        bool: True if sufficient data, False otherwise.
    """
    data_duration_days = (df['ds'].max() - df['ds'].min()).days
    
    if data_duration_days < (required_days * tolerance):
        print(f"Error: Insufficient historical data.")
        print(f"  > Requested: {required_days} days")
        print(f"  > Available: {data_duration_days} days")
        print("  > Please ensure buckets contain enough history or reduce 'training_days' in settings.toml")
        return False
    
    return True
