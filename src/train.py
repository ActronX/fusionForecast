import os
import sys
import pandas as pd
import warnings

sys.path.append(os.getcwd())

# Suppress common warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)

import neuralprophet
from neuralprophet import NeuralProphet
from src.config import settings
from src.data_loader import fetch_training_data, validate_data_sufficiency


def train_model():
    print("Starting training pipeline... (NeuralProphet)")
    
    # Fetch and prepare data using shared loader
    result = fetch_training_data(verbose=True)
    if result is None:
        sys.exit(1)
    
    df_prophet, regressor_names = result
    
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
        learning_rate=p_settings.get('learning_rate', 1e-4),
        epochs=p_settings.get('epochs', 10),
        batch_size=p_settings.get('batch_size', 128),
        # Regularization parameters
        trend_reg=p_settings.get('trend_reg', 0.0),
        seasonality_reg=p_settings.get('seasonality_reg', 0.0),
        ar_reg=p_settings.get('ar_reg', 0.0),
        # AutoRegressive architecture
        n_lags=p_settings.get('n_lags', 0),
        n_forecasts=p_settings.get('n_forecasts', 1),
        ar_layers=p_settings.get('ar_layers', []),
        accelerator=p_settings.get('accelerator', 'auto'),
        drop_missing=True
    )
    
    # Add future regressors (weather data)
    reg_mode = p_settings.get('regressor_mode', 'additive')
    reg_reg = p_settings.get('future_regressor_regularization', 0.0)

    for reg_name in regressor_names:
        print(f"Adding future regressor: {reg_name} (mode={reg_mode}, reg={reg_reg})")
        model.add_future_regressor(
            name=reg_name,
            mode=reg_mode,
            regularization=reg_reg
        )
    
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
