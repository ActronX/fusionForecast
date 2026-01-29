import os
import sys
import pandas as pd
import warnings
import logging

# Silence logs
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("neuralprophet").setLevel(logging.ERROR)
logging.getLogger("NP").setLevel(logging.ERROR)

sys.path.append(os.getcwd())

# Suppress common warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", message=".*DataFrameGroupBy.apply operated on the grouping columns.*")
warnings.filterwarnings("ignore", message=".*DataFrame is highly fragmented.*")
warnings.filterwarnings("ignore", message=".*Trying to infer the `batch_size`.*")

import neuralprophet
from neuralprophet import NeuralProphet
from src.config import settings
from src.data_loader import fetch_training_data, validate_data_sufficiency


def train_model():
    import torch
    print("Starting training pipeline... (NeuralProphet)")
    
    # Check GPU
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: GPU not detected. Training will be slow on CPU!")

    # Fetch and prepare data using shared loader
    result = fetch_training_data(verbose=True)
    if result is None:
        sys.exit(1)
    
    df_prophet, regressor_names = result
    
    print("---------------------------------------------------")
    print(f"Active Regressors ({len(regressor_names)}):")
    for name in regressor_names:
        print(f"  - {name}")
    print("---------------------------------------------------")

    # Validate data sufficiency
    training_days = settings['model']['training_days']
    if not validate_data_sufficiency(df_prophet, training_days):
        sys.exit(1)

    print("Configuring NeuralProphet model...")
    p_settings = settings['model']['neuralprophet']
    
    # Initialize NeuralProphet model with configured parameters
    model = NeuralProphet(
        growth=p_settings.get('growth', 'linear'),
        yearly_seasonality=p_settings.get('yearly_seasonality', False),
        weekly_seasonality=p_settings.get('weekly_seasonality', False), 
        daily_seasonality=p_settings.get('daily_seasonality', True),
        seasonality_mode=p_settings.get('seasonality_mode', 'additive'),
        learning_rate=p_settings.get('learning_rate'), # Defaults to None (auto) if null in settings
        epochs=p_settings.get('epochs'),               # Defaults to None (auto) if null in settings
        batch_size=p_settings.get('batch_size', 128),
        # Regularization parameters
        trend_reg=0, # Hardcoded to 0 as in tune.py (growth='off')
        seasonality_reg=p_settings.get('seasonality_reg', 0.0),
        ar_reg=p_settings.get('ar_reg', 0.0),
        # AutoRegressive architecture
        n_lags=p_settings.get('n_lags', 96),
        n_forecasts=p_settings.get('n_forecasts', 96),
        # d_hidden/num_hidden_layers removed for v0.9.0 compatibility
        ar_layers=p_settings.get('ar_layers', [64, 32]),
        accelerator=p_settings.get('accelerator', 'auto'),
        drop_missing=True
    )
    
    # Add future regressors (weather data)
    reg_mode = p_settings.get('regressor_mode', 'additive')

    for reg_name in regressor_names:
        print(f"Adding future regressor: {reg_name} (mode={reg_mode})")
        model.add_future_regressor(
            name=reg_name,
            mode=reg_mode
        )
    
    # Add lagged regressors (e.g., recent production data)
    lagged_regs = p_settings.get('lagged_regressors', {})
    for lag_name, n_lags_val in lagged_regs.items():
        if lag_name in df_prophet.columns:
            print(f"Adding lagged regressor: {lag_name} (n_lags={n_lags_val})")
            model.add_lagged_regressor(names=lag_name, n_lags=n_lags_val)
        else:
            print(f"Warning: Lagged regressor '{lag_name}' not found in training data. Skipping.")


    
    print(f"Training data summary before processing:\n{df_prophet.describe()}")
    
    # Ensure continuous time index and fill gaps with 0 (PV systems produce 0 at night)
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    df_prophet = df_prophet.set_index('ds').resample('15min').mean()
    df_prophet = df_prophet.fillna(0)
    df_prophet = df_prophet.reset_index()
    
    print(f"Training data shape after gap-filling: {df_prophet.shape}")
    print(f"Training data summary after gap-filling:\n{df_prophet.describe()}")

    print("Fitting model...")
    model.fit(df_prophet, freq='15min')
    
    # Save trained model
    model_path = settings['model']['path']
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    print(f"Saving model to {model_path}...")
    neuralprophet.save(model, model_path)
    
    size_mb = os.path.getsize(model_path) / 1024 / 1024
    print(f"Model saved successfully ({size_mb:.2f} MB)")
    print("Training complete.")

if __name__ == "__main__":
    train_model()
