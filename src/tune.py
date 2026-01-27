
import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
import sys
import os
import traceback
import logging
import optuna


# Ensure src can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Silence logs
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("neuralprophet").setLevel(logging.ERROR)
logging.getLogger("NP").setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.WARNING)
import warnings
# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", ".*Trying to infer the `batch_size`.*")
warnings.filterwarnings("ignore", ".*DataFrameGroupBy.apply operated on the grouping columns.*")
warnings.filterwarnings("ignore", ".*Series.view is deprecated.*")
warnings.filterwarnings("ignore", ".*Argument ``multivariate`` is an experimental feature.*")
warnings.filterwarnings("ignore", ".*is deprecated, use `isinstance.*")
warnings.filterwarnings("ignore", ".*Defined frequency .* is different than major frequency.*")
warnings.filterwarnings("ignore", ".*You called .*but have no logger configured.*")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)



from src.config import settings
from src.data_loader import fetch_training_data, validate_data_sufficiency

def evaluate_combination(params, df, regressor_names):
    """
    Evaluates a single combination of hyperparameters using a simple train/test split.
    """
    try:
        # Configuration
        train_days = settings['model']['training_days']
        
        # Setup NeuralProphet
        m = NeuralProphet(
            growth='linear',
            yearly_seasonality=settings['model']['neuralprophet'].get('yearly_seasonality', False),
            weekly_seasonality=settings['model']['neuralprophet'].get('weekly_seasonality', False),
            daily_seasonality=settings['model']['neuralprophet'].get('daily_seasonality', True),
            seasonality_mode=params.get('seasonality_mode', 'additive'),
            learning_rate=None,  # Use NeuralProphet's auto learning rate range test
            epochs=None,  # Auto-set based on dataset size (1000-4000 steps)
            n_lags=settings['model']['neuralprophet'].get('n_lags', 0),
            n_forecasts=settings['model']['neuralprophet'].get('n_forecasts', 96),
            batch_size=settings['model']['neuralprophet'].get('batch_size', 128),
            # d_hidden/num_hidden_layers removed due to incompatibility
            ar_layers=settings['model']['neuralprophet'].get('ar_layers', []),
            trend_reg=0,  # No trend regularization (growth='off' for PV)
            seasonality_reg=params['seasonality_reg'],
            ar_reg=params.get('ar_reg', 0.0),
            collect_metrics=False,
            accelerator=settings['model']['neuralprophet'].get('accelerator', 'auto'),
            drop_missing=True
        )
        
        reg_mode = params.get('regressor_mode', 'additive')
        
        for reg_name in regressor_names:
            m.add_future_regressor(
                name=reg_name, 
                mode=reg_mode
            )
        
        # Split Data
        freq = '15min'
        df_train, df_val = m.split_df(df, freq=freq, valid_p=0.2)
        
        # Fit
        m.fit(df_train, freq=freq, progress=None)
        
        # Predict on validation
        forecast = m.predict(df_val)
        
        # Validate Forecast
        
        # Evaluate over all forecast steps (horizon)
        yhat_cols = [c for c in forecast.columns if c.startswith('yhat')]
        y_true = forecast['y']
        
        threshold = settings['model'].get('tuning', {}).get('night_threshold', 50)
        
        abs_errors = []
        y_true_sum = 0
        
        for col in yhat_cols:
            y_pred = forecast[col].clip(lower=0)
            
            # Check for NaNs and apply night threshold
            valid_mask = ~y_true.isna() & ~y_pred.isna() & (y_true > threshold)
            
            if valid_mask.sum() > 0:
                # Accumulate errors for WMAPE
                abs_diff = np.abs(y_true[valid_mask] - y_pred[valid_mask])
                abs_errors.extend(abs_diff)
                y_true_sum += np.sum(y_true[valid_mask])
        
        # DEBUG: Print statistics if we have issues
        if y_true_sum == 0 or not abs_errors:
            print(f"DEBUG: Validation failed. y_true_sum={y_true_sum}, len(abs_errors)={len(abs_errors)}")
            # Check a few things
            total_rows = len(forecast)
            nan_preds = forecast[yhat_cols].isna().sum().sum()
            above_threshold = (forecast['y'] > threshold).sum()
            print(f"  Total Rows: {total_rows}")
            print(f"  NaN Predictions: {nan_preds}")
            print(f"  Points > Threshold ({threshold}): {above_threshold}")
            if total_rows > 0:
                 print(f"  Sample Y: {forecast['y'].head().tolist()}")
                 print(f"  Sample Pred: {forecast[yhat_cols[0]].head().tolist()}")

        if y_true_sum > 0:
            wmape = np.sum(abs_errors) / y_true_sum
            # Auxiliary metrics
            rmse = np.sqrt(np.mean(np.array(abs_errors)**2))
            # Requested combined score formula
            score = rmse * (1 + wmape)
        else:
            score = float('inf')
            wmape = float('inf')
            rmse = float('nan')
        
        return score, wmape, rmse

    except Exception as e:
        print(f"Failed for {params}: {e}")
        import traceback
        traceback.print_exc()
        return float('inf'), float('inf'), float('inf')

def objective(trial, df, regressor_names):
    # Simplified tuning: focus on PV-critical parameters only
    # Use NeuralProphet's auto-tuning for learning_rate and epochs
    params = {
        # Regularization: small values to prevent overfitting
        'seasonality_reg': trial.suggest_float('seasonality_reg', 0.0, 0.1),  # NP default: 0
        'ar_reg': trial.suggest_float('ar_reg', 0.0, 0.1),  # NP default: 0
        # Mode: critical for PV (additive vs multiplicative weather impact)
        'regressor_mode': trial.suggest_categorical('regressor_mode', ['additive', 'multiplicative']),  # NP default: None
        # Fixed values (using defaults or known-good settings)
        'seasonality_mode': 'additive',  # Standard for normalized PV data
    }
    
    score, wmape, rmse = evaluate_combination(params, df, regressor_names)
    
    # Store additional metrics in trial
    trial.set_user_attr("wmape", wmape)
    trial.set_user_attr("rmse", rmse)
    
    print(f"Trial {trial.number}: Score={score:.4f} | WMAPE={wmape:.4f} | RMSE={rmse:.4f}")
    print(f"  Params: {params}")
    
    return score 


def tune_hyperparameters():
    import torch
    
    result = fetch_training_data(verbose=True)
    
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: GPU not detected. Training will be slow on CPU!")
        
    if result is None:
        print("Data fetching failed.")
        return
    
    df, regressor_names = result
    
    # Ensure continuous time index and fill gaps with 0 (MATCHING TRAIN.PY LOGIC)
    print("Preprocessing data (resample 15min, fillna 0)...")
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.set_index('ds').resample('15min').mean()
    df = df.fillna(0)
    df = df.reset_index()
    
    # Validation
    training_days = settings['model']['training_days']
    if not validate_data_sufficiency(df, training_days):
        return

    # Algorithm settings
    n_trials = settings['model']['tuning'].get('trials', 30)
    process_count = settings['model']['tuning'].get('process_count', 4)
    
    print(f"Starting Optuna optimization with {n_trials} trials on {process_count} parallel cores...")
    
    # Shared counter for progress tracking (simplified for sequential)
    # n_jobs=1 performs sequential execution
    
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(multivariate=True)
    )
    
    try:
        # Evaluate objective directly
        study.optimize(
            lambda trial: objective(trial, df, regressor_names), 
            n_trials=n_trials,
            n_jobs=1,  # Sequential execution
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTuning interrupted by user.")
    except Exception as e:
        print(f"Optimization failed: {e}")
        traceback.print_exc()

    print("\n----------------------------------------------------------------")
    print("TOP 10 TRIALS")
    print("----------------------------------------------------------------")
    finished_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top_trials = sorted(finished_trials, key=lambda t: t.value)[:10]
    
    for i, t in enumerate(top_trials):
        rmse_val = t.user_attrs.get('rmse', float('nan'))
        wmape_val = t.user_attrs.get('wmape', float('nan'))
        params_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in t.params.items()])
        print(f"#{i+1:2}: Trial {t.number:2} - Score: {t.value:.4f} (WMAPE: {wmape_val:.4f}, RMSE: {rmse_val:.4f}) - Params: {{{params_str}}}")
    
    if study.best_trial:
        print("----------------------------------------------------------------")
        print(f"BEST PARAMETERS (Trial {study.best_trial.number})")
        print(f"Best Score: {study.best_value:.4f}")
        print("----------------------------------------------------------------")
        for key, value in study.best_params.items():
            if isinstance(value, float):
                print(f"{key} = {value:.6f}")
            else:
                print(f"{key} = \"{value}\"")
        print("----------------------------------------------------------------")
        print("Update these values in your settings.toml [model.neuralprophet] section!")

if __name__ == "__main__":
    tune_hyperparameters()
