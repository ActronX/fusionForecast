"""
Logic for checking available training history and data density.
Extracted from data_loader.py to check connections without circular imports.
"""

import pandas as pd
from src.db import InfluxDBWrapper
from src.config import settings

# InfluxDB Configuration Constants (needed for the validation logic)
BUCKETS = settings['influxdb']['buckets']
MEASUREMENTS = settings['influxdb']['measurements']
FIELDS = settings['influxdb']['fields']
TRAINING_DAYS = settings['model']['training_days']

# Regressor Fields
REG_CONFIG = FIELDS['regressor_history']
REGRESSOR_FIELDS = REG_CONFIG if isinstance(REG_CONFIG, list) else [REG_CONFIG]

import warnings
from influxdb_client.client.warnings import MissingPivotFunction
warnings.simplefilter("ignore", MissingPivotFunction)


def _get_available_history_start(db, bucket, measurement, field):
    """
    Finds the earliest timestamp in the database for the given field.
    """
    query = f'''
    from(bucket: "{bucket}")
      |> range(start: 0)
      |> filter(fn: (r) => r["_measurement"] == "{measurement}")
      |> filter(fn: (r) => r["_field"] == "{field}")
      |> first()
      |> keep(columns: ["_time"])
    '''
    df = db.query_dataframe(query)
    if df.empty:
        return None
    
    # InfluxDB returns _time column
    return pd.to_datetime(df['_time'].iloc[0]).tz_convert(None) # Make naive for comparison


def _check_data_density(db, bucket, measurement, field, start_date, end_date):
    """
    Checks if data coverage (days with data) is at least 90% in the given range.
    """
    # Count number of days that have at least one data point
    # Remove count() aggregation to get individual windows
    query = f'''
    from(bucket: "{bucket}")
      |> range(start: {start_date.isoformat()}Z, stop: {end_date.isoformat()}Z)
      |> filter(fn: (r) => r["_measurement"] == "{measurement}")
      |> filter(fn: (r) => r["_field"] == "{field}")
      |> aggregateWindow(every: 1d, fn: count, createEmpty: false)
    '''
    df = db.query_dataframe(query)
    
    expected_days = (end_date - start_date).days
    if expected_days <= 0:
        return 0.0, 0, 0

    if df.empty:
        return 0.0, 0, expected_days
    
    # Check if the first day actually has data
    # (Allow 1 day tolerance for timezone alignment issues)
    first_data_time = pd.to_datetime(df['_time'].iloc[0]).tz_convert(None)
    headers_gap = (first_data_time - start_date).days
    
    if headers_gap > 1:
        # If data starts more than 1 day after the requested start, count it as a failure
        # to force the window to shrink to the actual start.
        # Returning 0 density forces a fail.
        return 0.0, len(df), expected_days
    
    actual_days = len(df)
    
    # Cap actual at expected
    actual_days = min(actual_days, expected_days)
    
    density = actual_days / expected_days
    return density, int(actual_days), int(expected_days)


def _find_optimal_training_window(db, check_items, max_days, threshold=0.9, verbose=False):
    """
    Iteratively reduces the training window by 7 days until data density is >= threshold
    for ALL checked items.
    
    Args:
        db: InfluxDBWrapper
        check_items: List of dicts with keys 'name', 'bucket', 'measurement', 'field'
        max_days: Start window size
        ...
    """
    current_days = max_days
    today = pd.Timestamp.utcnow().replace(tzinfo=None)
    
    while current_days > 7: # Don't go below 1 week
        check_start = today - pd.Timedelta(days=current_days)
        all_pass = True
        
        if verbose:
            print(f"  > Checking {current_days} days...")
            
        for item in check_items:
            density, actual, expected = _check_data_density(
                db, item['bucket'], item['measurement'], item['field'], check_start, today
            )
            
            if density < threshold:
                if verbose:
                    print(f"    - {item['name']}: Low coverage {density:.1%} (FAIL)")
                all_pass = False
                break
            # Optional: detailed logging if needed, or just log pass
            # if verbose:
            #    print(f"    - {item['name']}: {density:.1%} (OK)")
        
        if all_pass:
            if verbose:
                print(f"    - All fields pass coverage check.")
            return current_days, 1.0, 0, 0 # Return generic success stats since we checked multiple variables
            
        current_days -= 7
        
    return current_days, 0.0, 0, 0


def get_max_available_training_days(max_days=None, verbose=False):
    """
    Determines the maximum available training days where ALL data (target + regressors)
    is available with >90% daily coverage.
    
    Args:
        max_days: Optional cap (default: settings.training_days)
        verbose: Print details
        
    Returns:
        tuple: (days_to_use, density_ratio, actual_count, expected_count, start_date)
    """
    db = InfluxDBWrapper()
    if max_days is None:
        max_days = TRAINING_DAYS
        
    if verbose:
        print("Analyzing available history for ALL fields...")

    # Define items to check
    check_items = []
    # 1. Target
    check_items.append({
        'name': 'Produced (Target)',
        'bucket': BUCKETS['history_produced'],
        'measurement': MEASUREMENTS['produced'],
        'field': FIELDS['produced']
    })
    # 2. Regressors
    for reg in REGRESSOR_FIELDS:
        check_items.append({
            'name': f'Regressor {reg}',
            'bucket': BUCKETS['regressor_history'],
            'measurement': MEASUREMENTS['regressor_history'],
            'field': reg
        })
        
    # Check max available history for EACH item to find the common start time
    start_times = []
    for item in check_items:
        t = _get_available_history_start(db, item['bucket'], item['measurement'], item['field'])
        if t is None:
            if verbose:
                print(f"  WARNING: No data found for {item['name']}")
            return max_days, 0.0, 0, 0, None
        
        if verbose:
             print(f"  - {item['name']}: Data starts {t}")
        start_times.append(t)
        
    # Valid history can only start when the *latest* of the start times occurs
    # (i.e., we need all columns to be present)
    common_start_time = max(start_times)
    
    today = pd.Timestamp.utcnow().replace(tzinfo=None)
    available_days = (today - common_start_time).days - 1 # Safety buffer
    
    if verbose:
        print(f"  > Earliest common start: {common_start_time} ({available_days + 1} days ago)")
        print(f"  > Applying 1 day safety buffer: {available_days} max days")
        
    # Start optimization from the lesser of available or requested
    start_days = min(available_days, max_days)
    
    if verbose:
        print(f"  > Optimizing window (starting at {start_days} days)...")
        
    optimized_days, density, actual, expected = _find_optimal_training_window(
        db, 
        check_items,
        start_days,
        threshold=0.9,
        verbose=verbose
    )
    
    if verbose:
        print(f"  > Optimized Window: {optimized_days} days")
            
    return optimized_days, density, actual, expected, common_start_time
