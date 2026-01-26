import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation
import sys
import os
import traceback
import logging
import optuna
from functools import partial
import multiprocessing

# Ensure src can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Silence cmdstanpy/optuna logs
logging.getLogger('cmdstanpy').disabled = True
logging.getLogger('cmdstanpy').propagate = False
logging.getLogger('cmdstanpy').setLevel(logging.CRITICAL)
optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.config import settings
from src.data_loader import fetch_training_data, validate_data_sufficiency

def evaluate_combination(params, df, regressor_names, cv_params):
    """
    Evaluates a single combination of hyperparameters.
    """
    try:
        m = Prophet(
            growth='flat',
            yearly_seasonality=settings['model']['prophet'].get('yearly_seasonality', True),
            daily_seasonality=True,
            weekly_seasonality=False,
            changepoint_prior_scale=params['changepoint_prior_scale'],
            seasonality_prior_scale=params['seasonality_prior_scale'],
            seasonality_mode=params.get('seasonality_mode', 'multiplicative')
        )
        for reg_name in regressor_names:
            # We can customize prior scales per regressor if we want, 
            # but for now we stick to the global setting.
            # However, if we have Clear Sky, we might want it to be additive or multiplicative depending on physics.
            # Usually PV = ALLSKY * Efficiency.
            # But the user specifically wanted these features.
            # "CLRSKY" is a base, "ALLSKY" is the actual driver.
            # We let Prophet figure it out with the mode in settings.
            
            m.add_regressor(
                reg_name, 
                mode=params.get('regressor_mode', 'multiplicative'), 
                prior_scale=params['regressor_prior_scale'], 
                standardize=False
            )
        
        m.fit(df)
        
        # Cross-validation
        df_cv = cross_validation(
            m, 
            initial=cv_params['initial'], 
            period=cv_params['period'], 
            horizon=cv_params['horizon'], 
            disable_tqdm=True, 
            parallel=None
        )
        
        # Calculate metrics manually to apply consistent filtering
        y_true = df_cv['y']
        y_pred = df_cv['yhat'].clip(lower=0)
        
        threshold = settings['model'].get('tuning', {}).get('night_threshold', 50)
        valid_mask = y_true > threshold
        
        if valid_mask.sum() > 0:
             rmse = np.sqrt(np.mean((y_true[valid_mask] - y_pred[valid_mask])**2))
             mape = np.mean(np.abs((y_true[valid_mask] - y_pred[valid_mask]) / y_true[valid_mask]))
             # Combined score: RMSE + weighted MAPE (MAPE is 0.0-1.0, so we scale it)
             # This formula is specifically chosen for PV forecasting:
             # 1. RMSE (primary): Ensures total energy volume is correct (critical for battery charging).
             # 2. MAPE (weighting): Penalizes "nervous" predictions in low-light conditions (morning/evening),
             #    which improves self-consumption planning accuracy.
             # Result is a balanced metric that values both peak power accuracy and curve shape.
             # LOWER IS BETTER (0 = Perfect prediction)
             score = rmse * (1 + mape)
        else:
             score = float('inf')
             rmse = float('nan')
             mape = float('nan')
        
        return score, rmse, mape

    except Exception as e:
        print(f"Failed for {params}: {e}")
        traceback.print_exc()
        return float('inf'), float('inf'), float('inf')

def objective(trial, df, regressor_names, cv_params):
    # Suggest parameters
    params = {
        'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.001, 0.5, log=True),
        'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.001, 10.0, log=True),
        'regressor_prior_scale': trial.suggest_float('regressor_prior_scale', 0.001, 10.0, log=True),
        'regressor_mode': trial.suggest_categorical('regressor_mode', ['additive', 'multiplicative']),
        'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative']),
    }
    
    score, rmse, mape = evaluate_combination(params, df, regressor_names, cv_params)
    
    # Store additional metrics in trial
    trial.set_user_attr("rmse", rmse)
    trial.set_user_attr("mape", mape)
    
    return score 

def logging_callback(study, trial, n_trials, counter, lock):
    with lock:
        counter.value += 1
        current_step = counter.value
    
    # Extract metrics and parameters for clearer display
    score = trial.value
    rmse = trial.user_attrs.get('rmse', float('nan'))
    mape = trial.user_attrs.get('mape', float('nan'))
    params = trial.params
    
    # Format parameters into a concise string
    params_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in params.items()])
    
    print(f"[{current_step}/{n_trials}] "
          f"Score: {score:.4f} (RMSE: {rmse:.4f}, MAPE: {mape:.4f}) - "
          f"Params: {{{params_str}}}")

def tune_hyperparameters():
    result = fetch_training_data(verbose=True)
    if result is None:
        print("Data fetching failed.")
        return
    
    df, regressor_names = result
    
    # Validation
    total_days = (df['ds'].max() - df['ds'].min()).days
    training_days = settings['model']['training_days']
    if not validate_data_sufficiency(df, training_days):
        return

    # CV Parameters
    target_horizon = '1 days'
    if total_days > 720:
        initial, period, horizon = '700 days', '14 days', target_horizon
    elif total_days > 400:
        initial, period, horizon = '370 days', '7 days', target_horizon
    else:
        initial = f'{max(5, int(total_days * 0.6))} days'
        period = f'{max(2, int(total_days * 0.1))} days'
        horizon = target_horizon
    
    cv_params = {'initial': initial, 'period': period, 'horizon': horizon}
    print(f"CV Settings: {cv_params}")

    # Optuna Study
    n_trials = settings['model']['tuning'].get('trials', 30)
    process_count = settings['model']['tuning'].get('process_count', 4)
    
    print(f"Starting Optuna optimization with {n_trials} trials on {process_count} parallel cores...")
    
    # Shared counter for parallel progress tracking
    counter = multiprocessing.Value('i', 0)
    lock = multiprocessing.Lock()

    # We use functools.partial instead of lambda for better pickling support on Windows
    # when using n_jobs > 1.
    objective_with_args = partial(objective, df=df, regressor_names=regressor_names, cv_params=cv_params)
    callback = partial(logging_callback, n_trials=n_trials, counter=counter, lock=lock)
    
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(multivariate=True)
    )
    study.optimize(
        objective_with_args, 
        n_trials=n_trials,
        n_jobs=process_count,
        callbacks=[callback]
    )

    print("\n----------------------------------------------------------------")
    print("TOP 10 TRIALS")
    print("----------------------------------------------------------------")
    # Sort trials by value and get top 10 finished ones
    finished_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top_trials = sorted(finished_trials, key=lambda t: t.value)[:10]
    
    for i, t in enumerate(top_trials):
        rmse_val = t.user_attrs.get('rmse', float('nan'))
        mape_val = t.user_attrs.get('mape', float('nan'))
        params_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in t.params.items()])
        print(f"#{i+1:2}: Trial {t.number:2} - Score: {t.value:.4f} (RMSE: {rmse_val:.4f}, MAPE: {mape_val:.4f}) - Params: {{{params_str}}}")
    
    print("----------------------------------------------------------------")
    print(f"BEST PARAMETERS (Trial {study.best_trial.number})")
    print("----------------------------------------------------------------")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"{key} = {value:.6f}")
        else:
            print(f"{key} = \"{value}\"")
    print("----------------------------------------------------------------")
    print("Update these values in your settings.toml [model.prophet] section!")

if __name__ == "__main__":
    tune_hyperparameters()

