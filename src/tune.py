import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import itertools
import sys
import os
import concurrent.futures
import traceback

import logging
# Ensure src can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Silence cmdstanpy logs
logging.getLogger('cmdstanpy').disabled = True
logging.getLogger('cmdstanpy').propagate = False
logging.getLogger('cmdstanpy').setLevel(logging.CRITICAL)

from src.db import InfluxDBWrapper
from src.config import settings
from src.preprocess import preprocess_data, prepare_prophet_dataframe

def fetch_and_prepare_data():
    """Duplicates logic from train.py to get the training dataframe"""
    print("Fetching training data...")
    db = InfluxDBWrapper()
    training_days = settings['model']['training_days']
    range_start = f"-{training_days}d"

    # 1. Produced
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
    if df_produced.empty: return None

    df_prophet = preprocess_data(df_produced, value_column=settings['influxdb']['fields']['produced'], is_prophet_input=True)
    df_prophet = prepare_prophet_dataframe(df_prophet, freq='30min')

    # 2. Regressor
    regressor_offset = settings['model']['preprocessing'].get('regressor_offset', '0m')
    regressor_scale = settings['model']['preprocessing'].get('regressor_scale', 1.0)
    
    # Check if we should use Perez POA / Effective
    use_pvlib = settings['model'].get('prophet', {}).get('use_pvlib', False)
    
    if use_pvlib:
        regressor_fields = [
            settings['influxdb']['fields'].get('effective_irradiance', 'effective_irradiance'),
            settings['influxdb']['fields'].get('temp_cell', 'temperature_cell')
        ]
    else:
        regressor_fields = [settings['influxdb']['fields']['regressor_history']]
    
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
    if df_regressor.empty: return None

    df_regressor = prepare_prophet_dataframe(df_regressor, freq='30min')
    
    regressor_names = []
    for field in regressor_fields:
        reg_name = field
        df_regressor[reg_name] = df_regressor[reg_name].interpolate(method='linear', limit_direction='both')
        regressor_names.append(reg_name)

    # Merge
    df_prophet = pd.merge(df_prophet, df_regressor[['ds'] + regressor_names], on='ds', how='inner')
    df_prophet.dropna(inplace=True)
    
    return df_prophet, regressor_names

def evaluate_combination(params, df, regressor_names, initial, period, horizon):
    """
    Evaluates a single combination of hyperparameters.
    Must be a top-level function for pickling in multiprocessing.
    """
    try:
        m = Prophet(
            yearly_seasonality=settings['model']['prophet'].get('yearly_seasonality', False),
            daily_seasonality=True,
            weekly_seasonality=False,
            changepoint_prior_scale=params['changepoint_prior_scale'],
            seasonality_prior_scale=params['seasonality_prior_scale'],
            seasonality_mode = params.get('seasonality_mode', 'multiplicative')
        )
        for reg_name in regressor_names:
            m.add_regressor(reg_name, mode=params.get('regressor_mode', 'multiplicative'), prior_scale=params['regressor_prior_scale'], standardize=False)
        
        m.fit(df)
        
        # Disable parallel here to avoid nested parallelism issues
        df_cv = cross_validation(m, initial=initial, period=period, horizon=horizon, disable_tqdm=True, parallel=None)
        # Calculate metrics manually to apply consistent filtering
        y_true = df_cv['y']
        y_pred = df_cv['yhat']
        
        # Filter out low values (threshold configurable)
        threshold = settings['model'].get('tuning', {}).get('night_threshold', 50)
        valid_mask = y_true > threshold
        
        if valid_mask.sum() > 0:
             mae = np.mean(np.abs(y_true[valid_mask] - y_pred[valid_mask]))
             rmse = np.sqrt(np.mean((y_true[valid_mask] - y_pred[valid_mask])**2))
             mape = np.mean(np.abs((y_true[valid_mask] - y_pred[valid_mask]) / y_true[valid_mask]))
        else:
             mae = float('nan')
             rmse = float('nan')
             mape = float('nan')
        
        result = params.copy()
        result['rmse'] = rmse
        result['mae'] = mae
        result['mape'] = mape
        return result

    except Exception as e:
        # It's hard to print exception tracebacks from worker processes cleanly,
        # so we return the error as a string or special result
        print(f"Failed for {params}: {e}")
        traceback.print_exc()
        result = params.copy()
        result['rmse'] = float('inf')
        result['mae'] = float('inf')
        result['mape'] = float('inf')
        return result

def tune_hyperparameters():
    df, regressor_names = fetch_and_prepare_data()
    if df is None or df.empty:
        print("Data fetching failed.")
        return

    # Parameter grid
    param_grid = {
        'changepoint_prior_scale': settings['model']['tuning'].get('changepoint_prior_scale', [0.05]),
        'seasonality_prior_scale': settings['model']['tuning'].get('seasonality_prior_scale', [10.0]),
        'regressor_prior_scale': settings['model']['tuning'].get('regressor_prior_scale', [0.5]),
        'regressor_mode': settings['model']['tuning'].get('regressor_mode', ['multiplicative']),
        'seasonality_mode': settings['model']['tuning'].get('seasonality_mode', ['multiplicative']),
    }

    # Generate all combinations
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    
    process_count = settings['model']['tuning'].get('process_count', 4)
    print(f"Starting tuning with {len(all_params)} combinations using {process_count} cores...")
    
    # We need at least enough data for CV. 
    total_days = (df['ds'].max() - df['ds'].min()).days
    
    training_days = settings['model']['training_days']
    if total_days < (training_days * 0.9):
        print(f"Error: Insufficient historical data for tuning.")
        print(f"  > Requested: {training_days} days")
        print(f"  > Available: {total_days} days")
        print("  > Please ensure buckets contain enough history or reduce 'training_days' in settings.toml")
        return

    # Optimization for short-term forecast (2 days)
    if total_days > 720:
        # > 2 Years: Train on ~2 years, test every 2 weeks
        initial = '700 days'
        period = '14 days'
        horizon = '2 days'
    elif total_days > 400:
        # > 1 Year: Train on ~1 year, test weekly
        initial = '370 days'
        period = '7 days'
        horizon = '2 days'
    else:
        # Fallback for smaller datasets
        initial = f'{max(5, int(total_days * 0.6))} days'
        period = f'{max(2, int(total_days * 0.1))} days'
        horizon = '2 days'
    
    print(f"CV Params: initial={initial}, period={period}, horizon={horizon}")

    results = []
    
    # ProcessPoolExecutor for parallel execution
    # max_workers=process_count as requested
    with concurrent.futures.ProcessPoolExecutor(max_workers=process_count) as executor:
        # Submit all tasks
        future_to_params = {
            executor.submit(
                evaluate_combination, 
                params, 
                df, 
                regressor_names, 
                initial, 
                period, 
                horizon
            ): params for params in all_params
        }
        
        total_tasks = len(future_to_params)
        for i, future in enumerate(concurrent.futures.as_completed(future_to_params), 1):
            params = future_to_params[future]
            try:
                # Add a timeout of 180 seconds (3 minutes) per model
                result = future.result(timeout=180)
                results.append(result)
                print(f"[{i}/{total_tasks}] ({i/total_tasks:.1%}) Finished {params} -> RMSE: {result['rmse']:.4f}, MAE: {result['mae']:.4f}, MAPE: {result['mape']:.4f}", flush=True)
            except concurrent.futures.TimeoutError:
                print(f"[{i}/{total_tasks}] ({i/total_tasks:.1%}) TIMED OUT {params} (skipping)", flush=True)
            except Exception as exc:
                print(f'{params} generated an exception: {exc}')

    # DataFrame of results
    results_df = pd.DataFrame(results)
    results_df.sort_values(by='rmse', inplace=True)

    print("\n----------------------------------------------------------------")
    print("TUNING RESULTS")
    print("----------------------------------------------------------------")
    print(results_df.to_string(index=False))
    print("----------------------------------------------------------------")

    if not results_df.empty:
        best_params = results_df.iloc[0]
        best_rmse = best_params['rmse']
        
        print(f"BEST PARAMETERS (RMSE: {best_rmse:.4f})")
        print("----------------------------------------------------------------")
        print(f"changepoint_prior_scale = {best_params['changepoint_prior_scale']}")
        print(f"seasonality_prior_scale = {best_params['seasonality_prior_scale']}")
        print(f"regressor_prior_scale = {best_params['regressor_prior_scale']}")
        print(f"regressor_mode = \"{best_params.get('regressor_mode', 'multiplicative')}\"")
        print(f"seasonality_mode = \"{best_params.get('seasonality_mode', 'multiplicative')}\"")
        print("----------------------------------------------------------------")
        print("Update these values in your settings.toml [prophet] section!")

if __name__ == "__main__":
    tune_hyperparameters()
