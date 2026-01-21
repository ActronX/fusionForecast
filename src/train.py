
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

def train_model():
    print("Starting training pipeline...")
    
    # Initialize DB
    db = InfluxDBWrapper()
    
    # Time range: last 30 days
    training_days = settings['forecast_parameters']['training_days']
    range_start = f"-{training_days}d"
    
    # 1. Fetch Produced Data (Target 'y')
    print(f"Fetching produced data from {settings['buckets']['b_history_produced']}...")
    produced_scale = settings['preprocessing'].get('produced_scale', 1.0)
    produced_offset = settings['preprocessing'].get('produced_offset', '0m')
    query_produced = f'''
    from(bucket: "{settings['buckets']['b_history_produced']}")
      |> range(start: {range_start})
      |> filter(fn: (r) => r["_measurement"] == "{settings['measurements']['m_produced']}")
      |> filter(fn: (r) => r["_field"] == "{settings['fields']['f_produced']}")
      |> map(fn: (r) => ({{ r with _value: r._value * {produced_scale} }}))
      |> timeShift(duration: {produced_offset})
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    df_produced = db.query_dataframe(query_produced)
    
    if df_produced.empty:
        print(f"Error: No produced data found for training.")
        print(f"  > Bucket: {settings['buckets']['b_history_produced']}")
        print(f"  > Measurement: {settings['measurements']['m_produced']}")
        print(f"  > Field: {settings['fields']['f_produced']}")
        print("  > Possible causes: Data missing in time range, or incorrect measurement/field names.")
        return

    # Preprocess Produced
    # Data is retrieved from InfluxDB (raw or pre-aggregated)
    # Clean and conform to Prophet (ds, y)
    df_prophet = preprocess_data(df_produced, value_column=settings['fields']['f_produced'], is_prophet_input=True)
    
    # Ensure produced data is resampled to 30min to match regressor
    # Use helper to standardize to 30min (handling index, tz, etc.)
    df_prophet = prepare_prophet_dataframe(df_prophet, freq='30min')

    # 2. Fetch Regressor Data (Solarcast - History)
    print(f"Fetching regressor data from {settings['buckets']['b_regressor_history']}...")
    regressor_offset = settings['preprocessing'].get('regressor_offset', '0m')
    regressor_scale = settings['preprocessing'].get('regressor_scale', 1.0)
    
    # Check if we should use Perez POA / Effective
    use_pvlib = settings.get('prophet', {}).get('use_pvlib', False)
    
    if use_pvlib:
        regressor_fields = [
            settings['fields'].get('f_effective_irradiance', 'effective_irradiance'),
            settings['fields'].get('f_temp_cell', 'temperature_cell')
        ]
    else:
        regressor_fields = [settings['fields']['f_regressor_history']]
    
    print(f"Using regressor fields: {regressor_fields}")
    
    # Filter for all requested regressor fields
    regressor_filter = " or ".join([f'r["_field"] == "{f}"' for f in regressor_fields])
    
    query_regressor = f'''
    from(bucket: "{settings['buckets']['b_regressor_history']}")
      |> range(start: {range_start})
      |> filter(fn: (r) => r["_measurement"] == "{settings['measurements']['m_regressor_history']}")
      |> filter(fn: (r) => {regressor_filter})
      |> map(fn: (r) => ({{ r with _value: r._value * {regressor_scale} }}))
      |> timeShift(duration: {regressor_offset})
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    df_regressor = db.query_dataframe(query_regressor)

    if df_regressor.empty:
        print("Warning: No regressor data found. Model might fail if regressor is mandatory.")
        print(f"  > Bucket: {settings['buckets']['b_regressor_history']}")
        print(f"  > Measurement: {settings['measurements']['m_regressor_history']}")
        print(f"  > Field: {settings['fields']['f_regressor_history']}")
    else:
        # Preprocess Regressor
        # Standardize regressor data using helper
        df_regressor = prepare_prophet_dataframe(df_regressor, freq='30min')
        
        # Interpolate and rename each regressor field
        regressor_names = []
        for field in regressor_fields:
            # We name the regressor for Prophet based on the field name
            reg_name = field 
            df_regressor[reg_name] = df_regressor[reg_name].interpolate(method='linear', limit_direction='both')
            regressor_names.append(reg_name)
        
        # Merge on 'ds'
        df_prophet = pd.merge(df_prophet, df_regressor[['ds'] + regressor_names], on='ds', how='inner')
        
        # Drop rows with NaNs (Prophet doesn't like NaNs in regressors)
        df_prophet.dropna(inplace=True)
    
    print(f"Training data shape after merge: {df_prophet.shape}")
    
    if df_prophet.empty:
        print("Error: Training data empty after merging regressor. Check time alignment.")
        return

    # Check if we have enough data as requested in settings
    data_duration_days = (df_prophet['ds'].max() - df_prophet['ds'].min()).days
    if data_duration_days < (training_days * 0.9): # Allow 10% tolerance for missing chunks/gaps
        print(f"Error: Insufficient historical data.")
        print(f"  > Requested: {training_days} days")
        print(f"  > Available: {data_duration_days} days")
        print("  > Please ensure buckets contain enough history or reduce 'training_days' in settings.toml")
        return

    # 3. Configure and Train Model
    print("Configuring Prophet model...")
    yearly_seasonality = settings['prophet'].get('yearly_seasonality', False)
    changepoint_prior = settings['prophet'].get('changepoint_prior_scale', 0.05)
    seasonality_prior = settings['prophet'].get('seasonality_prior_scale', 10.0)
    seasonality_mode = settings['prophet'].get('seasonality_mode', 'multiplicative')
    regressor_mode = settings['prophet'].get('regressor_mode', 'multiplicative')

    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        daily_seasonality=True,
        weekly_seasonality=False,
        changepoint_prior_scale=changepoint_prior,
        seasonality_prior_scale=seasonality_prior, 
        seasonality_mode = seasonality_mode
    )
    
    # Add Regressors
    prior_scale = settings['prophet']['regressor_prior_scale']
    for reg_name in regressor_names:
        print(f"Adding regressor: {reg_name}")
        model.add_regressor(reg_name, mode=regressor_mode, prior_scale=prior_scale)
    
    print("Fitting model...")
    model.fit(df_prophet)
    
    # 4. Save Model
    model_path = settings['model']['model_path']
    # Ensure directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    print(f"Saving model to {model_path}...")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
        
    print("Training complete.")

if __name__ == "__main__":
    train_model()
