
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
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger("neuralprophet").setLevel(logging.WARNING)
logging.getLogger("NP").setLevel(logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.WARNING)
import warnings
# Suppress specific warnings
warnings.filterwarnings("ignore", ".*Trying to infer the `batch_size`.*")
warnings.filterwarnings("ignore", ".*DataFrameGroupBy.apply operated on the grouping columns.*")
warnings.filterwarnings("ignore", ".*Series.view is deprecated.*")
warnings.filterwarnings("ignore", ".*Argument ``multivariate`` is an experimental feature.*")
warnings.filterwarnings("ignore", ".*is deprecated, use `isinstance.*")
warnings.filterwarnings("ignore", ".*Defined frequency .* is different than major frequency.*")
warnings.filterwarnings("ignore", ".*You called .*but have no logger configured.*")



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
            learning_rate=params['learning_rate'],
            epochs=params['epochs'],
            n_lags=settings['model']['neuralprophet'].get('n_lags', 0),
            d_hidden=settings['model']['neuralprophet'].get('d_hidden', 16),
            num_hidden_layers=settings['model']['neuralprophet'].get('num_hidden_layers', 0),
            ar_layers=settings['model']['neuralprophet'].get('ar_layers', []),
            trend_reg=params['trend_reg'],
            seasonality_reg=params['seasonality_reg'],
            ar_reg=params.get('ar_reg', 0.0),
            collect_metrics=False,
            accelerator=settings['model']['neuralprophet'].get('accelerator', 'auto')
        )
        
        reg_mode = params.get('regressor_mode', 'additive')
        
        for reg_name in regressor_names:
            m.add_future_regressor(
                name=reg_name, 
                mode=reg_mode
            )
        
        # Split Data
        # We use the last 20% for validation, or simpler: last 14 days if possible.
        # NP split_df uses fractions.
        freq = '15min'
        df_train, df_val = m.split_df(df, freq=freq, valid_p=0.2)
        
        # Fit
        m.fit(df_train, freq=freq, progress=None)
        
        # Predict on validation
        forecast = m.predict(df_val)
        
        # Evaluate
        # NP forecast columns: ds, y, yhat1
        if 'yhat1' in forecast.columns:
            y_pred_col = 'yhat1'
        else:
            y_pred_col = 'yhat'
            
        y_true = forecast['y']
        y_pred = forecast[y_pred_col].clip(lower=0)
        
        threshold = settings['model'].get('tuning', {}).get('night_threshold', 50)
        valid_mask = y_true > threshold
        
        if valid_mask.sum() > 0:
             rmse = np.sqrt(np.mean((y_true[valid_mask] - y_pred[valid_mask])**2))
             mape = np.mean(np.abs((y_true[valid_mask] - y_pred[valid_mask]) / y_true[valid_mask]))
             # Combined score formula
             score = rmse * (1 + mape)
        else:
             score = float('inf')
             rmse = float('nan')
             mape = float('nan')
        
        return score, rmse, mape

    except Exception as e:
        print(f"Failed for {params}: {e}")
        return float('inf'), float('inf'), float('inf')

def objective(trial, df, regressor_names):
    # Suggest parameters for NeuralProphet
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'epochs': trial.suggest_int('epochs', 20, 100),
        'trend_reg': trial.suggest_float('trend_reg', 0.0, 10.0),
        'seasonality_reg': trial.suggest_float('seasonality_reg', 0.0, 10.0),
        'ar_reg': trial.suggest_float('ar_reg', 0.0, 10.0),
        'future_regressor_regularization': trial.suggest_float('future_regressor_regularization', 0.0, 10.0),
        'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative']),
        'regressor_mode': trial.suggest_categorical('regressor_mode', ['additive', 'multiplicative']),
    }
    
    score, rmse, mape = evaluate_combination(params, df, regressor_names)
    
    # Store additional metrics in trial
    trial.set_user_attr("rmse", rmse)
    trial.set_user_attr("mape", mape)
    
    return score 



def tune_hyperparameters():
    result = fetch_training_data(verbose=True)
    if result is None:
        print("Data fetching failed.")
        return
    
    df, regressor_names = result
    
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
        mape_val = t.user_attrs.get('mape', float('nan'))
        params_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in t.params.items()])
        print(f"#{i+1:2}: Trial {t.number:2} - Score: {t.value:.4f} (RMSE: {rmse_val:.4f}, MAPE: {mape_val:.4f}) - Params: {{{params_str}}}")
    
    if study.best_trial:
        print("----------------------------------------------------------------")
        print(f"BEST PARAMETERS (Trial {study.best_trial.number})")
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
