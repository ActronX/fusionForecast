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


def fetch_intraday_data(db: InfluxDBWrapper, fetch_hours: float, regressor_fields: list) -> pd.DataFrame:
    """
    Fetches intraday historical data (target + regressors) for autoregression context.
    
    Args:
        db: InfluxDBWrapper instance
        fetch_hours: Number of hours of history to fetch
        regressor_fields: List of regressor field names to include
        
    Returns:
        pd.DataFrame: Prepared historical dataframe (resampled to 15min, gap-filled)
    """
    # Fetch historical target data from live bucket
    live_bucket = settings['influxdb']['buckets']['live']
    live_meas = settings['influxdb']['measurements']['live']
    live_field = settings['influxdb']['fields']['live']
    
    query_history_y = f'''
    import "date"
    from(bucket: "{live_bucket}")
      |> range(start: date.sub(d: {int(fetch_hours)}h, from: now()), stop: now())
      |> filter(fn: (r) => r["_measurement"] == "{live_meas}")
      |> filter(fn: (r) => r["_field"] == "{live_field}")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    df_hist_y = db.query_dataframe(query_history_y)
    
    y_col = live_field
    
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

    if df_hist_y.empty:
        print("Error: No historical target data found (Live or Stats).")
        return pd.DataFrame()

    # Fetch historical regressors to match the target history
    reg_hist_bucket = settings['influxdb']['buckets']['regressor_history']
    reg_hist_meas = settings['influxdb']['measurements']['regressor_history']
    regressor_scale = settings['model']['preprocessing'].get('regressor_scale', 1.0)
    
    field_filters_hist = ' or '.join([f'r["_field"] == "{f}"' for f in regressor_fields])
    
    query_history_reg = f'''
    import "date"
    from(bucket: "{reg_hist_bucket}")
      |> range(start: date.sub(d: {int(fetch_hours)}h, from: now()), stop: now())
      |> filter(fn: (r) => r["_measurement"] == "{reg_hist_meas}")
      |> filter(fn: (r) => {field_filters_hist})
      |> map(fn: (r) => ({{ r with _value: r._value * {regressor_scale} }}))
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    df_hist_reg = db.query_dataframe(query_history_reg)
    
    # Process and merge
    df_history = prepare_prophet_dataframe(df_hist_y, freq='15min')
    
    if not df_hist_reg.empty:
        df_hist_reg = prepare_prophet_dataframe(df_hist_reg, freq='15min')
        df_history = pd.merge(df_history, df_hist_reg, on='ds', how='inner')
    else:
        print("Warning: No historical regressor data found. AR context may lack features.")
    
    # Rename target column
    if y_col in df_history.columns:
        df_history.rename(columns={y_col: 'y'}, inplace=True)
        
    # Resample to ensure continuous 15-minute intervals and fill gaps
    df_history_final = df_history.copy()
    df_history_final['ds'] = pd.to_datetime(df_history_final['ds'])
    
    # Determine the latest timestamp before resampling for logging
    last_raw_ts = df_history_final['ds'].max()
    print(f"Latest historical data point (raw): {last_raw_ts}")
    
    df_history_final = df_history_final.set_index('ds').resample('15min').mean()
    df_history_final = df_history_final.fillna(0)
    df_history_final = df_history_final.reset_index()
    
    # Build column list: ds, y, regressors
    # We want to keep all regressor fields plus 'y' and 'ds'
    columns_to_keep = ['ds', 'y'] 
    for r in regressor_fields:
        if r not in df_history_final.columns:
             print(f"Warning: Missing regressor {r} in history. Filling with 0.")
             df_history_final[r] = 0.0
        else:
             df_history_final[r] = df_history_final[r].interpolate(method='linear', limit_direction='both')
        columns_to_keep.append(r)

    # Filter columns
    df_history_final = df_history_final[[c for c in columns_to_keep if c in df_history_final.columns]].copy()
    
    last_resampled_ts = df_history_final['ds'].max()
    print(f"Loaded {len(df_history_final)} historical data points. Latest (resampled): {last_resampled_ts}")
    
    # Warn if data is old (e.g. > 45 mins)
    # InfluxDB returns UTC (often naive), so we compare against UTC
    now_check = pd.Timestamp.utcnow().replace(tzinfo=None)
    if last_resampled_ts is not pd.NaT:
         # If dataframe is tz-aware, make now_check aware too, otherwise keep naive
         if last_resampled_ts.tzinfo is not None:
             now_check = pd.Timestamp.utcnow()

         age = (now_check - last_resampled_ts).total_seconds() / 60
         if age > 45:
            print(f"WARNING: Historical data is {age:.1f} minutes old! Intraday correction may be outdated.")
            
    return df_history_final


def fetch_future_regressors(db: InfluxDBWrapper, forecast_days: int) -> pd.DataFrame:
    """
    Fetches future regressor data (weather forecast) for prediction.
    
    Args:
        db: InfluxDBWrapper instance
        forecast_days: Number of days to fetch forecast for
        
    Returns:
        pd.DataFrame: Prepared future dataframe (resampled to 15min, gap-filled, interpolated)
    """
    print(f"Fetching regressor data for next {forecast_days} days...")
    
    regressor_scale = settings['model']['preprocessing'].get('regressor_scale', 1.0)
    regressor_fields_config = settings['influxdb']['fields']['regressor_future']
    regressor_fields = regressor_fields_config if isinstance(regressor_fields_config, list) else [regressor_fields_config]
    
    # Build Flux query for future regressors
    field_filters = ' or '.join([f'r["_field"] == "{f}"' for f in regressor_fields])
    query_regressor = f'''
    import "date"
    from(bucket: "{settings['influxdb']['buckets']['regressor_future']}")
      |> range(start: now(), stop: date.add(d: {forecast_days}d, to: now()))
      |> filter(fn: (r) => r["_measurement"] == "{settings['influxdb']['measurements']['regressor_future']}")
      |> filter(fn: (r) => {field_filters})
      |> map(fn: (r) => ({{ r with _value: r._value * {regressor_scale} }}))
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    df_future = db.query_dataframe(query_regressor)
    
    if df_future.empty:
        print("Error: No future regressor data found.")
        return pd.DataFrame()

    df_future = prepare_prophet_dataframe(df_future, freq='15min')
    
    # Interpolate and prepare regressor columns
    regressor_names = []
    for field in regressor_fields:
        if field not in df_future.columns:
             print(f"Warning: Future regressor field '{field}' missing. Filling with 0.")
             df_future[field] = 0.0
        df_future[field] = df_future[field].interpolate(method='linear', limit_direction='both')
        regressor_names.append(field)
    
    # Return dataframe with ds and regressors only
    df_future_final = df_future[['ds'] + regressor_names].copy()
    
    return df_future_final

