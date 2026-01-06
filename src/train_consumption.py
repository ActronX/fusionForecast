
import os
import pickle
import pandas as pd
import logging
logging.getLogger('cmdstanpy').disabled = True
logging.getLogger('cmdstanpy').propagate = False
logging.getLogger('cmdstanpy').setLevel(logging.CRITICAL)
from prophet import Prophet
from src.config import settings
from src.db import InfluxDBWrapper
from src.preprocess import preprocess_data, prepare_prophet_dataframe

def train_consumption_model():
    print("Starting consumption training pipeline...")
    
    # Initialize DB
    db = InfluxDBWrapper()
    
    # Time range: last 30 days
    # Time range: last 30 days
    # Allow overriding training_days specifically for consumption
    p_config = settings['prophet'].get('consumption', {})
    training_days = p_config.get('training_days', settings['forecast_parameters']['training_days'])
    
    print(f"Training with data from last {training_days} days.")
    range_start = f"-{training_days}d"
    
    # 1. Fetch Consumption Data (Target 'y')
    # Using 'b_history_produced' as default bucket if not specified for consumption
    bucket = settings['buckets'].get('b_consumption', settings['buckets']['b_history_produced'])
    measurement = settings['measurements']['m_consumption']
    field = settings['fields']['f_consumption']
    
    print(f"Fetching consumption data from {bucket}...")
    
    # Reuse produced_scale or check for specific consumption scale
    # Defaulting to 1.0 if not found, as consumption might already be in correct units or configured similarly
    # Check if there is a specific scale for consumption
    scale = settings['preprocessing'].get('consumption_scale', 1.0)
    
    # Offset
    offset = settings['preprocessing'].get('consumption_offset', '0m')
    
    query = f'''
    from(bucket: "{bucket}")
      |> range(start: {range_start})
      |> filter(fn: (r) => r["_measurement"] == "{measurement}")
      |> filter(fn: (r) => r["_field"] == "{field}")
      |> map(fn: (r) => ({{ r with _value: r._value * {scale} }}))
      |> timeShift(duration: {offset})
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    df_consumption = db.query_dataframe(query)
    
    if df_consumption.empty:
        print(f"Error: No consumption data found for training.")
        print(f"  > Bucket: {bucket}")
        print(f"  > Measurement: {measurement}")
        print(f"  > Field: {field}")
        print("  > Possible causes: Data missing in time range, or incorrect measurement/field names.")
        return

    # Preprocess
    # Clean and conform to Prophet (ds, y)
    df_prophet = preprocess_data(df_consumption, value_column=field, is_prophet_input=True)
    
    # Resample
    df_prophet = prepare_prophet_dataframe(df_prophet, freq='30min')
    
    print(f"Training data shape: {df_prophet.shape}")

    # 3. Configure and Train Model
    print("Configuring Prophet model for consumption (No Regressors)...")
    
    # Load consumption specific settings if available, else fallback or default
    p_config = settings['prophet'].get('consumption', {})
    
    yearly_seasonality = p_config.get('yearly_seasonality', True)
    weekly_seasonality = p_config.get('weekly_seasonality', True)
    daily_seasonality = p_config.get('daily_seasonality', True)
    seasonality_mode = p_config.get('seasonality_mode', 'multiplicative')
    
    changepoint_prior = p_config.get('changepoint_prior_scale', 0.05)
    seasonality_prior = p_config.get('seasonality_prior_scale', 10.0)
    holidays_prior = p_config.get('holidays_prior_scale', 10.0)

    # Holidays (Germany, Bavaria)
    holidays_df = _generate_holidays_dataframe(df_prophet['ds'].dt.year.unique().tolist())
    
    if holidays_df is not None and not holidays_df.empty:
         print(f"Total holiday entries (Public + School): {len(holidays_df)}")
    else:
         holidays_df = None

    print(f"Model Parameters:")
    print(f"  > Yearly Seasonality: {yearly_seasonality}")
    print(f"  > Weekly Seasonality: {weekly_seasonality}")
    print(f"  > Daily Seasonality:  {daily_seasonality}")
    print(f"  > Seasonality Mode:   {seasonality_mode}")
    print(f"  > Changepoint Prior:  {changepoint_prior}")
    print(f"  > Seasonality Prior:  {seasonality_prior}")
    print(f"  > Holidays Prior:     {holidays_prior}")

    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        daily_seasonality=daily_seasonality,
        weekly_seasonality=weekly_seasonality,
        changepoint_prior_scale=changepoint_prior,
        seasonality_prior_scale=seasonality_prior, 
        holidays_prior_scale=holidays_prior,
        seasonality_mode=seasonality_mode,
        holidays=holidays_df
    )
    
    # No regressors added
    
    print("Fitting model...", flush=True)
    model.fit(df_prophet)
    
    # 4. Save Model
    model_path = settings['model']['model_path_consumption']
    # Ensure directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    print(f"Saving model to {model_path}...")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
        
    print("Consumption training complete.")

def _generate_holidays_dataframe(years):
    """
    Generates a DataFrame of Public and School holidays for Bavaria (DE-BY).
    """
    all_holidays = []
    
    # 0. Prepare years
    if years:
        years.append(max(years) + 1)
    else:
        # Fallback if no years found
        import datetime
        current_year = datetime.datetime.now().year
        years = [current_year, current_year + 1]

    # 1. Public Holidays using 'holidays' library
    try:
        import holidays
        print(f"Adding Public Holidays for years: {years} (DE-BY)")
        bavaria_holidays = holidays.country_holidays('DE', subdiv='BY', years=years)
        
        for date, name in bavaria_holidays.items():
            all_holidays.append({'ds': pd.to_datetime(date), 'holiday': f"Public: {name}"})

    except ImportError:
        print("Warning: 'holidays' library not installed. Skipping public holidays.")
    except Exception as e:
         print(f"Warning: Error getting public holidays: {e}")

    # 2. School Holidays (using ferien-api.de)
    try:
        import requests
        print(f"Fetching School Holidays for years: {years} (DE-BY) from ferien-api.de...")
        
        for year in years:
            try:
                url = f"https://ferien-api.de/api/v1/holidays/BY/{year}"
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    for item in data:
                        start = pd.to_datetime(item['start']).date()
                        end = pd.to_datetime(item['end']).date()
                        name = item.get('name', 'School Holiday')
                        
                        date_range = pd.date_range(start=start, end=end)
                        for d in date_range:
                            all_holidays.append({'ds': d, 'holiday': f"School: {name}"})
                else:
                    print(f"  > Failed to fetch {year}: Status {response.status_code}")
            except Exception as e_req:
                print(f"  > Error fetching {year}: {e_req}")

    except ImportError:
        print("Warning: 'requests' library not installed. Skipping school holidays.")

    if not all_holidays:
        return None
        
    df = pd.DataFrame(all_holidays)
    # Sort and remove duplicates (keep first found)
    df.sort_values('ds', inplace=True)
    df.drop_duplicates(subset=['ds'], keep='first', inplace=True)
    
    return df

if __name__ == "__main__":
    train_consumption_model()
