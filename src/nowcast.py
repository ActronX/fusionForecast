
import math
import time
from datetime import datetime, timedelta, timezone
import pandas as pd
from src.config import settings
from src.db import InfluxDBWrapper

def run_nowcast():
    """
    Executes the Nowcast logic to adjust short-term forecasts based on real-time production data.
    
    Concept: "Real-Time Forecast Correction"
    We compare the ACTUAL production vs. the FORECAST since sunrise to calculate a "Performance Factor" (Damping Factor).
    - If the sun is stronger than predicted (Factor > 1.0), we scale UP the future forecast.
    - If it is cloudier than predicted (Factor < 1.0), we scale DOWN the future forecast.
    
    This allows the system to react dynamically to intraday weather changes (e.g. fog clearing up earlier than predicted).
    
    The Logic consists of two parts:
    1. Calculate Damping Factor: Weighted average of (Actual / Forecast) for the past few hours. 
       Recent data is weighted higher (Half-Life: 1 hour).
    2. Apply to Future: The calculated factor is applied to the near-future forecast.
       The influence of this factor decays over time, merging back to the original forecast (Half-Life: 1 hour).
    """
    print(f"Starting nowcast at {datetime.now()}...")

    # 1. Load Configuration
    if 'nowcast' not in settings:
         print("Warning: [nowcast] section missing in settings.toml. Using defaults.")
    
    use_damping = settings.get('nowcast', {}).get('use_damping_factor', True)
    m_nowcast = settings.get('nowcast', {}).get('m_nowcast', 'nowcast')
    
    if not use_damping:
        print("Nowcast (Damping Factor) is disabled in settings.")
        return

    # DB Connection
    db = InfluxDBWrapper()

    # Time setup
    now_dt = datetime.now(timezone.utc)
    # Fetch data for the last 3 hours (relevant context for decay logic)
    start_time_history = now_dt - timedelta(hours=3) 
    
    # 2. Fetch Data
    
    # A. Actual Production
    bucket_live = settings.get('nowcast', {}).get('bucket_live', 'energy_meter')
    measurement_live = settings.get('nowcast', {}).get('measurement_live', 'production')
    field_live = settings.get('nowcast', {}).get('field_live', 'production')
    
    min_damping = settings.get('nowcast', {}).get('min_damping_factor', 0.75)
    max_damping = settings.get('nowcast', {}).get('max_damping_factor', 1.5)
    
    query_prod = f'''
    from(bucket: "{bucket_live}")
      |> range(start: {int(start_time_history.timestamp())})
      |> filter(fn: (r) => r["_measurement"] == "{measurement_live}")
      |> filter(fn: (r) => r["_field"] == "{field_live}")
      |> aggregateWindow(every: 15m, fn: mean, createEmpty: false)
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    
    df_prod = db.query_dataframe(query_prod)
    
    # B. Forecast Data (Original)
    # bucket: fusionForecastData (b_target_forecast), measurement: dwd (m_forecast)
    query_forecast = f'''
    from(bucket: "{settings['buckets']['b_target_forecast']}")
      |> range(start: {int(start_time_history.timestamp())})
      |> filter(fn: (r) => r["_measurement"] == "{settings['measurements']['m_forecast']}")
      |> filter(fn: (r) => r["_field"] == "{settings['fields']['f_forecast']}")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    df_forecast = db.query_dataframe(query_forecast)

    if df_prod.empty:
        print("No production data found. Using Damping Factor 1.0 (Default)")
        damping_factor = 1.0
    elif df_forecast.empty:
        print("No forecast data found. Cannot compute damping.")
        return
    else:
        # Align Data
        if '_time' in df_prod.columns:
            df_prod.rename(columns={'_time': 'time'}, inplace=True)
        if '_time' in df_forecast.columns:
            df_forecast.rename(columns={'_time': 'time'}, inplace=True)

        if 'time' in df_prod.columns:
            df_prod['time'] = pd.to_datetime(df_prod['time'])
            df_prod.set_index('time', inplace=True)
        if 'time' in df_forecast.columns:
            df_forecast['time'] = pd.to_datetime(df_forecast['time'])
            df_forecast.set_index('time', inplace=True)
            
        df_prod.sort_index(inplace=True)
        df_forecast.sort_index(inplace=True)
        
        # Rename columns for clarity (handle pivot result)
        if field_live in df_prod.columns:
            df_prod.rename(columns={field_live: 'production'}, inplace=True)
        
        f_forecast = settings['fields']['f_forecast']
        if f_forecast in df_forecast.columns:
            df_forecast.rename(columns={f_forecast: 'forecast'}, inplace=True)
        
        # Merge/Join to compare
        # tolerance=10m (match production to forecast within 10 minutes)
        df_merged = pd.merge_asof(
            df_prod[['production']], 
            df_forecast[['forecast']], 
            left_index=True, 
            right_index=True, 
            tolerance=pd.Timedelta('10m'), 
            direction='nearest'
        )
        
        # 3. Calculate Damping Factor (Past Performance)
        # Weight recent data more strongly: Exponential decay of "relevance" (Half-Life 1h)
        # Example:
        # - T-00 min: Weight 100% (1.00) -> Full Impact
        # - T-30 min: Weight ~71% (0.71)
        # - T-60 min: Weight  50% (0.50) -> Half Impact
        # - T-2h:     Weight  25% (0.25)
        #
        # Why?
        # We want to react fast to changing weather (e.g., fog clearing up).
        # Data from 2 hours ago is "historically correct" but "irrelevant" for the current trend.
        
        sum_forecast_weighted = 0.0
        sum_production_weighted = 0.0
        count = 0
        
        now_ts = now_dt.timestamp()
        
        # Get night threshold from settings (prophet.tuning.night_threshold)
        night_threshold = settings.get('prophet', {}).get('tuning', {}).get('night_threshold', 50)
        
        print("Calculating damping factor...")
        
        for time_idx, row in df_merged.iterrows():
            prod = row['production']
            fcst = row['forecast']
            
            if pd.isna(prod) or pd.isna(fcst):
                continue
                
            ts = time_idx.timestamp()
            if ts > now_ts:
                continue
            
            # Filter Logic:
            # Only compare when relevant power is expected.
            # We ignore values where forecast is too low (night/low light).
            # User requested 2x the normal night threshold for nowcast robustness.
            if fcst < (2 * night_threshold): 
                continue
                
            age_seconds = now_ts - ts
            age_hours = max(0, age_seconds / 3600.0)
            
            # Decay Formula: Weight = 0.5 ^ (age_hours / 1.0)
            weight = math.pow(0.5, age_hours / 1.0)
            
            sum_forecast_weighted += fcst * weight
            sum_production_weighted += prod * weight
            count += 1
            
        damping_factor = 1.0
        
        # Only apply if we have enough accumulated sun-hours data (weighted sum).
        # Threshold: 4 * night_threshold allows for robust base of data before modifying forecast.
        if sum_forecast_weighted > (6 * night_threshold): 
            damping_factor = sum_production_weighted / sum_forecast_weighted
            
            # Safety Clamping (configurable) to prevent extreme scaling 
            # based on short-term anomalies.
            damping_factor = max(min_damping, min(damping_factor, max_damping))
        
        print(f"Analysis: Matches:{count} | Prod:{sum_production_weighted:.0f} vs Fcst:{sum_forecast_weighted:.0f}")    
        print(f"Damping Factor: {damping_factor:.4f}")

    # 4. Apply to Future Forecast
    # Use Flux date functions and pivot to match forecast.py style
    # get forecast_days from settings, default to 14 if not found
    load_forecast_days = settings.get('forecast_parameters', {}).get('forecast_days', 14)

    query_future = f'''
    import "date"
    from(bucket: "{settings['buckets']['b_target_forecast']}")
      |> range(start: now(), stop: date.add(d: {load_forecast_days}d, to: now()))
      |> filter(fn: (r) => r["_measurement"] == "{settings['measurements']['m_forecast']}")
      |> filter(fn: (r) => r["_field"] == "{settings['fields']['f_forecast']}")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    df_future = db.query_dataframe(query_future)
    
    if df_future.empty:
        print("No future forecast to adjust.")
        return
        
    # Standardize column names (pivot results in field columns)
    f_forecast = settings['fields']['f_forecast']
    if f_forecast not in df_future.columns:
         print(f"Error: Field '{f_forecast}' not found in future query result.")
         return
         
    # Rename to generic 'forecast' for internal processing
    df_future.rename(columns={f_forecast: 'forecast'}, inplace=True)

    # Ensure time index
    if '_time' in df_future.columns:
        df_future.rename(columns={'_time': 'time'}, inplace=True)
    if 'time' in df_future.columns:
        df_future['time'] = pd.to_datetime(df_future['time'])
        df_future.set_index('time', inplace=True)
    
    pv_peak = settings.get('preprocessing', {}).get('max_power_clip', 10000) 

    adjusted_values = []
    
    now_ts = now_dt.timestamp()
    
    for time_idx, row in df_future.iterrows():
        fcst = row['forecast']
        if pd.isna(fcst):
            continue
            
        ts = time_idx.timestamp()
        
        diff_hours = max(0, (ts - now_ts) / 3600.0)
        
        # --- Future Decay Logic ---
        # We trust the current deviation (Damping Factor) fully for the immediate moment.
        # However, weather anomalies (clouds, fog) are often temporary.
        # We don't want to scale the forecast 10 hours away based on a cloud right now.
        #
        # Formula: Weight = 0.5 ^ (hours / 1.0)
        # This is an "Exponential Decay" with a Half-Life of 1 hour.
        # - T+0h: Weight 100% -> Full influence of Damping Factor
        # - T+1h: Weight  50% -> Half influence
        # - T+2h: Weight  25% -> Quarter influence
        # - T+4h: Weight   6% -> Nearly back to original Forecast (1.0)
        
        decay_weight = math.pow(0.5, diff_hours / 1.0)
        
        # Calculate Effective Factor for this specific time point
        # Blend the calculated dampingFactor with standard 1.0 based on decay.
        effective_factor = 1.0 + ((damping_factor - 1.0) * decay_weight)
        
        # Clamp to physical maximum (max_power_clip from settings)
        adjusted_val = min(fcst * effective_factor, pv_peak)
        adjusted_values.append(adjusted_val)
        
    df_future['adjusted'] = adjusted_values
    
    # 5. Write Result
    print(f"Writing {len(df_future)} nowcast points to measurement '{m_nowcast}'...")
    
    df_to_write = df_future[['adjusted']].copy()
    df_to_write.rename(columns={'adjusted': settings['fields']['f_forecast']}, inplace=True)
    
    db.write_dataframe(
        df=df_to_write,
        bucket=settings['buckets']['b_target_forecast'],
        measurement=m_nowcast
    )
    
    print("Nowcast complete.")

if __name__ == "__main__":
    run_nowcast()
