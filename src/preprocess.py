
import pandas as pd
import pvlib
from src.config import settings

def apply_nighttime_zero(df: pd.DataFrame, lat: float, lon: float, time_col: str = 'ds', value_col: str = 'y', verbose: bool = False) -> pd.DataFrame:
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

    # pvlib expects localized times or UTC. 
    # If naive, assume UTC or local standard time? 
    # The project seems to operate on naive timestamps (assumed local or UTC from Influx).
    # get_solarposition handles naive as UTC by default if no tz is given, or local if provided.
    # To be safe and consistent with settings, we should probably assume the time is compatible with the lat/lon.
    
    solpos = pvlib.solarposition.get_solarposition(times, lat, lon)
    
    # elevation > 0 means day. <= 0 means night (civil twilight etc aside, or use a small threshold like 0 or -3)
    # Using 0 as strict horizon.
    is_night = solpos['elevation'] <= -3
    
    n_modified = is_night.sum()
    if verbose and n_modified > 0:
        print(f"  [Nighttime Correction] Setting {n_modified} data points to 0 based on solar position.")

    df.loc[is_night.values, value_col] = 0.0
    
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
