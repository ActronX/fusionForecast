"""
Centralized data loading for training, forecasting, and plotting.
All scaling configuration is applied here - consumers receive data in Watt units.
"""

import pandas as pd
from src.db import InfluxDBWrapper
from src.config import settings
from src.preprocess import preprocess_data, prepare_prophet_dataframe, apply_nighttime_zero, interpolate_regressors, preprocess_regressors
from src.flux_builder import _build_flux_query


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

# Model Configuration
TRAINING_DAYS = settings['model']['training_days']

# Regressor Fields (Primary)
REG_CONFIG = FIELDS['regressor_history']
REGRESSOR_FIELDS = REG_CONFIG if isinstance(REG_CONFIG, list) else [REG_CONFIG]

# Regressor Fields (Secondary / optional)
# If bucket is empty string, this source is disabled entirely.
_BUCKET_2 = BUCKETS.get('regressor_history_2', '')
HAS_REGRESSOR_2 = bool(_BUCKET_2)  # "" -> False, "home" -> True

if HAS_REGRESSOR_2:
    REG_CONFIG_2 = FIELDS.get('regressor_history_2', [])
    REGRESSOR_FIELDS_2 = REG_CONFIG_2 if isinstance(REG_CONFIG_2, list) else [REG_CONFIG_2]
else:
    REGRESSOR_FIELDS_2 = []

# Combined list used by train / forecast
ALL_REGRESSOR_FIELDS = REGRESSOR_FIELDS + REGRESSOR_FIELDS_2


# ============================================================================
# Helper Functions
# ============================================================================

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


def _fetch_merged_regressors(db, 
                             bucket1, meas1, fields1, 
                             bucket2, meas2, fields2, 
                             range_start, range_stop=None, 
                             offset='0m', interpolate=False, verbose=True):
    """
    Fetch both primary and (optional) secondary regressors and merge them 
    into a single preprocessed dataframe aligned on 'ds' with 15min frequency.
    """
    if verbose:
        print(f"  - Fetching {len(fields1)} regressors from {bucket1}...")
        
    query_regressors = _build_flux_query(
        bucket=bucket1,
        measurement=meas1,
        fields=fields1,
        range_start=range_start,
        range_stop=range_stop,
        scale=REGRESSOR_SCALE,
        offset=offset,
        interpolate=interpolate,
        verbose=verbose
    )
    
    df_regressors = db.query_dataframe(query_regressors)
    
    if df_regressors.empty:
        if verbose:
            print(f"Warning: No primary regressor data found in {bucket1}.")
        df_merged = pd.DataFrame(columns=['ds'] + fields1)
    else:
        df_merged = preprocess_regressors(df_regressors, fields1, freq='15min')

    # Fetch 2nd Regressor Source (optional)
    if HAS_REGRESSOR_2 and bucket2:
        if verbose:
            print(f"  - Fetching {len(fields2)} regressors from {bucket2}...")
            
        query_reg2 = _build_flux_query(
            bucket=bucket2,
            measurement=meas2,
            fields=fields2,
            range_start=range_start,
            range_stop=range_stop,
            scale=REGRESSOR_SCALE,
            offset=offset,
            interpolate=interpolate,
            verbose=verbose
        )
        
        df_reg2 = db.query_dataframe(query_reg2)
        
        if df_reg2.empty:
            if verbose:
                print(f"Warning: No data from {bucket2}. Continuing without it.")
        else:
            df_reg2 = preprocess_regressors(df_reg2, fields2, freq='15min')
            if df_merged.empty:
                df_merged = df_reg2
            else:
                df_merged = pd.merge(df_merged, df_reg2, on='ds', how='outer')
                
    # Sort and reset index
    if not df_merged.empty:
        df_merged = df_merged.sort_values('ds').reset_index(drop=True)
        
    return df_merged


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
        fields=FIELDS['produced'],
        range_start=range_start,
        scale=PRODUCED_SCALE,
        offset=PRODUCED_OFFSET,
        downsample=True,   # 10s raw data → mean per 15min window
        interpolate=True,  # then fill any remaining gaps
        verbose=verbose
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

    # Apply Nighttime Zeroing
    if verbose:
        print("  - Applying nighttime zeroing...")
    
    # Ensure ds is datetime for pvlib
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    
    df_prophet = apply_nighttime_zero(
        df_prophet, 
        time_col='ds', 
        value_col='y',
        verbose=verbose
    )

    # 2. Fetch Regressor Data (Merged)
    df_regs = _fetch_merged_regressors(
        db=db,
        bucket1=BUCKETS['regressor_history'], meas1=MEASUREMENTS['regressor_history'], fields1=REGRESSOR_FIELDS,
        bucket2=BUCKETS.get('regressor_history_2', ''), meas2=MEASUREMENTS.get('regressor_history_2', ''), fields2=REGRESSOR_FIELDS_2,
        range_start=range_start,
        offset=REGRESSOR_OFFSET,
        verbose=verbose
    )
    
    if df_regs.empty:
        print("Error: Training data regressors empty. Check time alignment.")
        return None
        
    df_prophet = pd.merge(df_prophet, df_regs, on='ds', how='inner')
    
    if df_prophet.empty:
        print("Error: Training data empty after merging target and regressors. Check time alignment.")
        return None

    # Fill NaN in regressor columns (from sampling mismatches or outer-joins between primary/secondary sources).
    # This is done in Pandas rather than InfluxDB to save DB performance on large training ranges.
    df_prophet = interpolate_regressors(df_prophet, ALL_REGRESSOR_FIELDS)
    
    if verbose:
        print(f"  [OK] Loaded {len(df_prophet)} samples successfully")
    
    return df_prophet, ALL_REGRESSOR_FIELDS


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
    # Fetch target from live bucket (with live_scale and interpolation)
    query_target = _build_flux_query(
        bucket=BUCKETS['live'],
        measurement=MEASUREMENTS['live'],
        fields=FIELDS['live'],
        range_start=f'date.sub(d: {int(fetch_hours)}h, from: now())',
        range_stop='now()',
        scale=LIVE_SCALE,
        interpolate=True,
        downsample=True,
        verbose=False
    )
    
    df_target = db.query_dataframe(query_target)
    target_field = FIELDS['live']
    
    # Fallback to stats bucket if live is unavailable
    if df_target.empty:
        print(f"Warning: Live data unavailable. Using stats bucket with interpolation.")
        
        query_target = _build_flux_query(
            bucket=BUCKETS['history_produced'],
            measurement=MEASUREMENTS['produced'],
            fields=FIELDS['produced'],
            range_start=f'-{int(fetch_hours)}h',
            scale=PRODUCED_SCALE,
            offset=PRODUCED_OFFSET,
            interpolate=True,
            verbose=False
        )
        
        df_target = db.query_dataframe(query_target)
        target_field = FIELDS['produced']
    
    if df_target.empty:
        print("Error: No historical target data found (Live or Stats).")
        return pd.DataFrame()
    
    # Prepare target
    df_hist = preprocess_data(df_target, value_column=target_field, is_prophet_input=True)
    df_hist = prepare_prophet_dataframe(df_hist, freq='15min')
    
    # Fetch regressors (with interpolation)
    df_regs = _fetch_merged_regressors(
        db=db,
        bucket1=BUCKETS['regressor_history'], meas1=MEASUREMENTS['regressor_history'], fields1=REGRESSOR_FIELDS,
        bucket2=BUCKETS.get('regressor_history_2', ''), meas2=MEASUREMENTS.get('regressor_history_2', ''), fields2=REGRESSOR_FIELDS_2,
        range_start=f'date.sub(d: {int(fetch_hours)}h, from: now())',
        range_stop='now()',
        interpolate=True,
        verbose=False
    )
    
    if not df_regs.empty:
        df_hist = pd.merge(df_hist, df_regs, on='ds', how='outer')
    
    # Final check: Ensure all columns exist in df_hist
    df_hist = interpolate_regressors(df_hist, regressor_fields)
            
    # Sort
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
    # Fetch future regressors with interpolation to ensure no gaps
    df_future = _fetch_merged_regressors(
        db=db,
        bucket1=BUCKETS['regressor_future'], meas1=MEASUREMENTS['regressor_future'], fields1=REGRESSOR_FIELDS,
        bucket2=BUCKETS.get('regressor_future_2', ''), meas2=MEASUREMENTS.get('regressor_future_2', ''), fields2=REGRESSOR_FIELDS_2,
        range_start='-1h',
        range_stop=f'{forecast_days}d',
        interpolate=True,
        verbose=False
    )
    
    if df_future.empty:
        print("Error: No future regressor data found")
        return pd.DataFrame()
    
    # Sort (interpolation already done in InfluxDB and preprocess_regressors)
    df_future = df_future.sort_values('ds').reset_index(drop=True)
    
    return df_future