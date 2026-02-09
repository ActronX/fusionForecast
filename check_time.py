import pandas as pd
from src.data_loader import fetch_training_data
from src.weather_utils import get_solar_position
from src.config import settings

def check_time_alignment():
    print("Fetching RAW training data for time check...")
    data = fetch_training_data(verbose=False)
    
    if not data:
        print("Error: No data found.")
        return

    df, _ = data
    df['ds'] = pd.to_datetime(df['ds']) + pd.Timedelta(hours=1) # Try shifting +1h
    
    # Check if tz-aware
    print(f"Timestamp Timezone: {df['ds'].dt.tz}")
    
    # Calculate solar noon for a few days
    # Solar noon should be around 12:00-13:00 local time (depending on DST)
    # If it's at 11:00 or 10:00, we have an offset.
    
    # Get peak production hour for each day
    df['date'] = df['ds'].dt.date
    daily_peaks = df.loc[df.groupby('date')['y'].idxmax()]
    
    print("\nDaily Peak Production Times (Top 10):")
    print(daily_peaks[['ds', 'y']].head(10))
    
    # Calculate theoretical solar noon
    print("\nTheoretical Solar Noon vs Peak Production:")
    for _, row in daily_peaks.head(5).iterrows():
        # Solar noon via get_solar_position
        noon_pos = get_solar_position(pd.DatetimeIndex([row['ds']]))
        
        # We need to find the actual solar noon time for this day, not just position at peak
        # Approximation: Peak elevation
        times = pd.date_range(start=row['ds'].floor('D'), end=row['ds'].ceil('D'), freq='10min')
        solpos = get_solar_position(times)
        solar_noon_time = solpos['elevation'].idxmax()
        
        print(f"Date: {row['ds'].date()} | Peak Prod: {row['ds'].time()} | Solar Noon: {solar_noon_time.time()} | Diff: {(row['ds'] - solar_noon_time).total_seconds()/3600:.1f}h")

if __name__ == "__main__":
    check_time_alignment()
