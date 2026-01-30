"""
Centralized data loading for training, forecasting, and plotting.
All scaling configuration is applied here - consumers receive data in Watt units.
"""

import pandas as pd
from src.db import InfluxDBWrapper
from src.config import settings
from src.preprocess import preprocess_data, prepare_prophet_dataframe


# ============================================================================
# Configuration Constants (loaded once)
# ============================================================================

# InfluxDB Configuration
BUCKETS = settings['influxdb']['buckets']
MEASUREMENTS = settings['influxdb']['measurements']
FIELDS = settings['influxdb']['fields']

# Preprocessing Configuration
PREPROCESSING = settings['model']['preprocessing']
MAX_POWER_CLIP = PREPROCESSING.get('max_power_clip', 5400)
PRODUCED_SCALE = PREPROCESSING.get('produced_scale', 1.0)
LIVE_SCALE = PREPROCESSING.get('live_scale', 1.0)
REGRESSOR_SCALE = PREPROCESSING.get('regressor_scale', 1.0)
PRODUCED_OFFSET = PREPROCESSING.get('produced_offset', '0m')
REGRESSOR_OFFSET = PREPROCESSING.get('regressor_offset', '0m')
NIGHT_THRESHOLD = PREPROCESSING.get('night_threshold', 50)

# Model Configuration
TRAINING_DAYS = settings['model']['training_days']

# Regressor Fields
REG_CONFIG = FIELDS['regressor_history']
REGRESSOR_FIELDS = REG_CONFIG if isinstance(REG_CONFIG, list) else [REG_CONFIG]


# ============================================================================
# Helper Functions
# ============================================================================

def _build_flux_query(bucket, measurement, field, range_start, scale=1.0, offset='0m'):
    """
    Build a standard Flux query with scaling and time offset.
    
    Args:
        bucket: InfluxDB bucket name
        measurement: Measurement name
        field: Field name
        range_start: Time range start (e.g., '-30d')
        scale: Scaling factor to apply
        offset: Time offset to apply
    
    Returns:
        str: Flux query string
    """
    query = f'''
    from(bucket: "{bucket}")
      |> range(start: {range_start})
      |> filter(fn: (r) => r["_measurement"] == "{measurement}")
      |> filter(fn: (r) => r["_field"] == "{field}")
    '''
    
    if scale != 1.0:
        query += f'  |> map(fn: (r) => ({{ r with _value: r._value * {scale} }}))\n'
    
    if offset != '0m':
        query += f'  |> timeShift(duration: {offset})\n'
    
    query += '  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")\n'
    
    return query


def _fetch_and_prepare(db, query, value_column, error_context, verbose=True):
    """
    Execute query, check for empty result, and prepare dataframe.
    
    Args:
        db: InfluxDBWrapper instance
        query: Flux query string
        value_column: Column name to use as value
        error_context: Dict with context info for error messages
        verbose: Print progress messages
    
    Returns:
        pd.DataFrame or None if empty
    """
    df = db.query_dataframe(query)
    
    if df.empty:
        if verbose:
            print(f"Error: No data found for {error_context['name']}")
            print(f"  > Bucket: {error_context['bucket']}")
            print(f"  > Measurement: {error_context['measurement']}")
            print(f"  > Field: {error_context['field']}")
        return None
    
    df = preprocess_data(df, value_column=value_column, is_prophet_input=True)
    df = prepare_prophet_dataframe(df, freq='15min')
    return df


# ============================================================================
# Public Functions
# ============================================================================

def fetch_training_data(verbose=True):
    """
    Fetch and prepare training data from InfluxDB.
    
    Args:
        verbose: Print progress messages
    
    Returns:
        tuple: (df_prophet, regressor_names) where df_prophet contains 'ds', 'y',
               and regressor columns, all scaled and ready for training
        None: If data fetching fails
    """
    if verbose:
        print("Fetching training data...")
    
    db = InfluxDBWrapper()
    range_start = f"-{TRAINING_DAYS}d"
    
    # 1. Fetch Target Data (Produced)
    if verbose:
        print(f"  - Fetching target data ({TRAINING_DAYS} days)...")
    
    query_produced = _build_flux_query(
        bucket=BUCKETS['history_produced'],
        measurement=MEASUREMENTS['produced'],
        field=FIELDS['produced'],
        range_start=range_start,
        scale=PRODUCED_SCALE,
        offset=PRODUCED_OFFSET
    )
    
    df_prophet = _fetch_and_prepare(
        db, query_produced, FIELDS['produced'],
        error_context={
            'name': 'training data',
            'bucket': BUCKETS['history_produced'],
            'measurement': MEASUREMENTS['produced'],
            'field': FIELDS['produced']
        },
        verbose=verbose
    )
    
    if df_prophet is None:
        return None
    
    # 2. Fetch Regressor Data
    if verbose:
        print(f"  - Fetching {len(REGRESSOR_FIELDS)} regressors...")
    
    regressor_filter = " or ".join([f'r["_field"] == "{f}"' for f in REGRESSOR_FIELDS])
    
    query_regressors = f'''
    from(bucket: "{BUCKETS['regressor_history']}")
      |> range(start: {range_start})
      |> filter(fn: (r) => r["_measurement"] == "{MEASUREMENTS['regressor_history']}")
      |> filter(fn: (r) => {regressor_filter})
      |> map(fn: (r) => ({{ r with _value: r._value * {REGRESSOR_SCALE} }}))
      |> timeShift(duration: {REGRESSOR_OFFSET})
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    
    df_regressors = db.query_dataframe(query_regressors)
    
    if df_regressors.empty:
        print("Error: No regressor data found")
        return None
    
    # Prepare and merge
    df_regressors = prepare_prophet_dataframe(df_regressors, freq='15min')
    cols_to_keep = ['ds'] + [c for c in df_regressors.columns if c in REGRESSOR_FIELDS]
    df_regressors = df_regressors[cols_to_keep]
    
    df_prophet = pd.merge(df_prophet, df_regressors, on='ds', how='inner')
    
    if df_prophet.empty:
        print("Error: Training data empty after merging. Check time alignment.")
        return None
    
    if verbose:
        print(f"  ✓ Loaded {len(df_prophet)} samples successfully")
    
    return df_prophet, REGRESSOR_FIELDS


def validate_data_sufficiency(df, required_days, tolerance=0.9):
    """
    Validate that DataFrame contains sufficient historical data.
    
    Args:
        df: DataFrame with 'ds' column
        required_days: Number of days required
        tolerance: Minimum fraction of required days (default 0.9 = 90%)
    
    Returns:
        bool: True if sufficient data, False otherwise
    """
    if df.empty or 'ds' not in df.columns:
        return False
    
    date_range = (df['ds'].max() - df['ds'].min()).days
    return date_range >= (required_days * tolerance)


def fetch_intraday_data(db, fetch_hours, regressor_fields):
    """
    Fetch intraday historical data for autoregression context.
    
    Loads last N hours of target + regressors from live bucket (with fallback to stats).
    All data is scaled to Watt units.
    
    Args:
        db: InfluxDBWrapper instance
        fetch_hours: Number of hours of history to fetch
        regressor_fields: List of regressor field names
    
    Returns:
        pd.DataFrame: Historical dataframe with 'ds', 'y', and regressor columns
    """
    # Fetch target from live bucket (with live_scale)
    query_target = f'''
    import "date"
    from(bucket: "{BUCKETS['live']}")
      |> range(start: date.sub(d: {int(fetch_hours)}h, from: now()), stop: now())
      |> filter(fn: (r) => r["_measurement"] == "{MEASUREMENTS['live']}")
      |> filter(fn: (r) => r["_field"] == "{FIELDS['live']}")
      |> map(fn: (r) => ({{ r with _value: r._value * {LIVE_SCALE} }}))
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    
    df_target = db.query_dataframe(query_target)
    target_field = FIELDS['live']
    
    # Fallback to stats bucket if live is unavailable
    if df_target.empty:
        print(f"Warning: Live data unavailable. Using stats bucket.")
        
        query_target = _build_flux_query(
            bucket=BUCKETS['history_produced'],
            measurement=MEASUREMENTS['produced'],
            field=FIELDS['produced'],
            range_start=f'-{int(fetch_hours)}h',
            scale=PRODUCED_SCALE
        )
        
        df_target = db.query_dataframe(query_target)
        target_field = FIELDS['produced']
    
    if df_target.empty:
        print("Error: No historical target data found (Live or Stats).")
        return pd.DataFrame()
    
    # Prepare target
    df_hist = preprocess_data(df_target, value_column=target_field, is_prophet_input=True)
    df_hist = prepare_prophet_dataframe(df_hist, freq='15min')
    
    # Fetch regressors
    regressor_filter = " or ".join([f'r["_field"] == "{f}"' for f in regressor_fields])
    
    query_regressors = f'''
    import "date"
    from(bucket: "{BUCKETS['regressor_history']}")
      |> range(start: date.sub(d: {int(fetch_hours)}h, from: now()), stop: now())
      |> filter(fn: (r) => r["_measurement"] == "{MEASUREMENTS['regressor_history']}")
      |> filter(fn: (r) => {regressor_filter})
      |> map(fn: (r) => ({{ r with _value: r._value * {REGRESSOR_SCALE} }}))
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    
    df_regressors = db.query_dataframe(query_regressors)
    
    if not df_regressors.empty:
        df_regressors = prepare_prophet_dataframe(df_regressors, freq='15min')
        cols_to_keep = ['ds'] + [c for c in df_regressors.columns if c in regressor_fields]
        df_regressors = df_regressors[cols_to_keep]
        df_hist = pd.merge(df_hist, df_regressors, on='ds', how='outer')
    
    # Fill gaps
    df_hist = df_hist.set_index('ds').resample('15min').interpolate(method='linear').reset_index()
    df_hist = df_hist.sort_values('ds').reset_index(drop=True)
    
    return df_hist


def fetch_future_regressors(db, forecast_days):
    """
    Fetch future regressor data (weather forecast) for prediction.
    
    Args:
        db: InfluxDBWrapper instance
        forecast_days: Number of days to fetch forecast for
    
    Returns:
        pd.DataFrame: Future dataframe with 'ds' and regressor columns
    """
    regressor_filter = " or ".join([f'r["_field"] == "{f}"' for f in REGRESSOR_FIELDS])
    
    query_future = f'''
    from(bucket: "{BUCKETS['regressor_future']}")
      |> range(start: -1h, stop: {forecast_days}d)
      |> filter(fn: (r) => r["_measurement"] == "{MEASUREMENTS['regressor_future']}")
      |> filter(fn: (r) => {regressor_filter})
      |> map(fn: (r) => ({{ r with _value: r._value * {REGRESSOR_SCALE} }}))
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    
    df_future = db.query_dataframe(query_future)
    
    if df_future.empty:
        return pd.DataFrame()
    
    df_future = prepare_prophet_dataframe(df_future, freq='15min')
    cols_to_keep = ['ds'] + [c for c in df_future.columns if c in REGRESSOR_FIELDS]
    df_future = df_future[cols_to_keep]
    
    # Interpolate to ensure no gaps
    df_future = df_future.set_index('ds').resample('15min').interpolate(method='linear').reset_index()
    df_future = df_future.sort_values('ds').reset_index(drop=True)
    
    return df_future
