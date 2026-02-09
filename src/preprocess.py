
import pandas as pd
from src.weather_utils import get_solar_position
from src.config import settings

def apply_nighttime_zero(df: pd.DataFrame, time_col: str = 'ds', value_col: str = 'y', verbose: bool = False) -> pd.DataFrame:
    """
    Sets value_col to 0 when the sun is below the horizon using pvlib.
    """
    df = df.copy()
    
    # Ensure we have a datetime index or column for calculation
    if time_col in df.columns:
        times = pd.to_datetime(df[time_col])
    elif isinstance(df.index, pd.DatetimeIndex):
        times = df.index
    else:
        raise ValueError(f"Column {time_col} not found and index is not DatetimeIndex")
    
    solpos = get_solar_position(times)
    
    # elevation > 0 means day. <= 0 means night (civil twilight etc aside, or use a small threshold like 3 or -3)
    # Read threshold from settings, default to 2.0 if not found
    threshold = settings.get('model', {}).get('preprocessing', {}).get('nighttime_elevation_threshold', 2.0)
    
    is_night = solpos['elevation'] <= threshold
    
    n_modified = is_night.sum()
    if verbose and n_modified > 0:
        print(f"  [Nighttime Correction] Setting {n_modified} data points to 0 based on solar position.")

    df.loc[is_night.values, value_col] = 0.0
    
    return df

def preprocess_regressors(df: pd.DataFrame, regressor_names: list, freq: str = '15min') -> pd.DataFrame:
    """
    Standardizes time axis and fills regressor gaps in one step.
    
    Args:
        df: Input DataFrame (from DB or raw source)
        regressor_names: List of expected regressor columns
        freq: Target frequency (default 15min)
        
    Returns:
        pd.DataFrame: Cleaned, resampled, and interpolated DataFrame
    """
    # 1. Standardize Time (ds, timezone, frequency)
    df = prepare_prophet_dataframe(df, freq=freq)
    
    # 2. Fill Regressor Gaps (interpolation, numeric conversion, missing columns)
    df = interpolate_regressors(df, regressor_names)
    
    # 3. Final column enforcement
    cols_to_keep = ['ds'] + [c for c in df.columns if c in regressor_names]
    df = df[cols_to_keep]
    
    return df

def interpolate_regressors(df: pd.DataFrame, regressor_names: list) -> pd.DataFrame:
    """
    Ensures all regressor columns are present, interpolates gaps linearly, 
    and fills remaining NaNs with 0.0.
    
    Args:
        df: DataFrame with 'ds' and potentially regressor columns
        regressor_names: List of expected regressor column names
        
    Returns:
        pd.DataFrame: DataFrame with all regressors present and filled
    """
    df = df.copy()
    for col in regressor_names:
        if col not in df.columns:
            df[col] = 0.0
        else:
            # Ensure numeric type before interpolation
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].interpolate(method='linear', limit_direction='both').fillna(0.0)
    return df

def truncate_time_column(df: pd.DataFrame, freq: str = '1h') -> pd.DataFrame:
    """
    Resamples/Truncates the time index to a specific frequency (default 1h).
    Assumes df has a DatetimeIndex or a column named '_time' or 'ds'.
    """
    df = df.copy()
    
    # Ensure we are working with datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if '_time' in df.columns:
            df['_time'] = pd.to_datetime(df['_time'])
            df.set_index('_time', inplace=True)
        elif 'ds' in df.columns:
            df['ds'] = pd.to_datetime(df['ds'])
            df.set_index('ds', inplace=True)
        else:
            raise ValueError("DataFrame must have a DatetimeIndex or '_time'/'ds' column.")

    # Sort index
    df.sort_index(inplace=True)
    
    # Resample to hourly and take the mean
    # resample is more idiomatic than groupby(floor) + mean
    df = df.resample(freq).mean(numeric_only=True)
    
    return df

def preprocess_data(df, value_column='_value', is_prophet_input=True):
    """
    Cleans data, clips values to 0-6000, and prepares for Prophet.
    """
    df = df.copy()
    
    # Handle missing values
    df.dropna(subset=[value_column], inplace=True)
    
    # Clip values
    max_clip = settings['model']['preprocessing']['max_power_clip']
    df[value_column] = df[value_column].clip(lower=0, upper=max_clip)
    
    if is_prophet_input:
        # Prophet requires 'ds' and 'y' columns
        # If the index is datetime, move it to 'ds'
        if isinstance(df.index, pd.DatetimeIndex):
            df['ds'] = df.index
        elif '_time' in df.columns:
            df['ds'] = pd.to_datetime(df['_time'])
        else:
            raise ValueError("Could not determine datetime column for Prophet 'ds'.")

        # Remove timezone if present (Prophet requirement)
        if df['ds'].dt.tz is not None:
             df['ds'] = df['ds'].dt.tz_localize(None)
            
        df['y'] = df[value_column]
        
        # Ensure ds is tz-naive or tz-aware as per requirement:
        # "The entire data processing must keep the timezone unchanged"
        # Prophet usually prefers tz-naive, but handled recent versions handle tz-aware.
        # We will keep it as is from the DB (Influx usually returns UTC).
        
        return df[['ds', 'y']]
    else:
        return df

def prepare_prophet_dataframe(df, freq='15min'):
    """
    Standardizes a dataframe for Prophet use:
    1. Ensures 'ds' column exists (from _time, index, or ds).
    2. Strips timezone from 'ds'.
    3. Resamples/Truncates to the specified frequency (default 15min).
    4. Returns dataframe with 'ds' as column, ready for merge/training.
    """
    df = df.copy()
    
    # 1. Identify/Create 'ds'
    if 'ds' not in df.columns:
        if '_time' in df.columns:
            df['ds'] = pd.to_datetime(df['_time'])
        elif isinstance(df.index, pd.DatetimeIndex):
            df['ds'] = df.index
        else:
             # Fallback, maybe 'ds' is already index but not valid col?
             pass

    if 'ds' not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
         raise ValueError("DataFrame must have a DatetimeIndex or '_time'/'ds' column.")

    # 2. Strip Timezone
    # If ds is column
    if 'ds' in df.columns and df['ds'].dt.tz is not None:
         df['ds'] = df['ds'].dt.tz_localize(None)
    # If index is used below in truncate, it handles it, but let's be safe
    
    # 3. Resample
    # Set index to ds explicitly to ensure truncate uses the tz-stripped version (if we just stripped it)
    if 'ds' in df.columns:
        df.set_index('ds', inplace=True)
    
    # Use freq
    df = truncate_time_column(df, freq=freq)
    
    # 4. Reset Index to get 'ds' column back
    df.index.name = 'ds'
    df.reset_index(inplace=True)
    
    return df

def postprocess_forecast(forecast_df):
    """
    Clips forecast values to 0-6000 and sets negatives to 0.
    """
    # Prophet returns 'yhat'
    if 'yhat' in forecast_df.columns:
        max_clip = settings['model']['preprocessing']['max_power_clip']
        forecast_df['yhat'] = forecast_df['yhat'].clip(lower=0, upper=max_clip)
    
    return forecast_df

def prepare_forecast_input(model, chunk, current_history, n_lags, n_forecasts, regressor_names):
    """
    Prepares the input dataframe for a NeuralProphet prediction step.
    Handles interpolation, padding, and history concatenation.
    
    Args:
        model: Trained NeuralProphet model
        chunk: DataFrame containing the future regressors for this step
        current_history: DataFrame containing the recent history (y and regressors)
        n_lags: Number of lags in the model
        n_forecasts: Number of forecast steps
        regressor_names: List of regressor column names
        
    Returns:
        tuple: (step_input, chunk_padded, actual_len)
    """
    import numpy as np # Import locally to avoid top-level dependency if not needed elsewhere
    
    chunk = chunk.copy()
    
    # Fill any missing values in regressors to avoid model crash
    chunk = interpolate_regressors(chunk, regressor_names)

    actual_len = len(chunk)
    
    # Pad chunk if it's shorter than n_forecasts (to avoid 'negative dimensions' or shape mismatch in NP)
    if actual_len < n_forecasts:
        padding_len = n_forecasts - actual_len
        last_row = chunk.iloc[-1].copy()
        
        # Create dates for padding to ensure strict continuity and correct frequency
        padding_dates = pd.date_range(
            start=last_row['ds'] + pd.Timedelta(minutes=15),
            periods=padding_len,
            freq='15min'
        )
        
        # Create padding dataframe with correct columns and types
        padding_df = pd.DataFrame(index=range(padding_len), columns=chunk.columns)
        padding_df['ds'] = padding_dates
        for col in chunk.columns:
            if col != 'ds':
                padding_df[col] = last_row[col]
        
        chunk_padded = pd.concat([chunk, padding_df], ignore_index=True)
    else:
        chunk_padded = chunk
    
    # Use manual construction to avoid 'make_future_dataframe' bugs with regressors
    # especially in recursive loops where some columns might be missing or all NaN.
    chunk_padded['y'] = np.nan
    if n_lags > 0 and not current_history.empty:
        step_input = pd.concat([current_history.tail(n_lags), chunk_padded], ignore_index=True)
    else:
        step_input = chunk_padded.copy()
        step_input['y'] = 0.0
    
    step_input = step_input.drop_duplicates(subset='ds', keep='last')
    step_input['y'] = pd.to_numeric(step_input['y'], errors='coerce')
    
    return step_input, chunk_padded, actual_len
