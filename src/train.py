
import os
import sys
import pickle
import logging
logging.getLogger('cmdstanpy').disabled = True
logging.getLogger('cmdstanpy').propagate = False
logging.getLogger('cmdstanpy').setLevel(logging.CRITICAL)
from prophet import Prophet
from src.config import settings
from src.data_loader import fetch_training_data, validate_data_sufficiency

def train_model():
    print("Starting training pipeline...")
    
    # Fetch and prepare data using shared loader
    result = fetch_training_data(verbose=True)
    if result is None:
        sys.exit(1)
    
    df_prophet, regressor_names = result
    
    # Validate data sufficiency
    training_days = settings['model']['training_days']
    if not validate_data_sufficiency(df_prophet, training_days):
        sys.exit(1)

    # 3. Configure and Train Model
    print("Configuring Prophet model...")
    yearly_seasonality = settings['model']['prophet'].get('yearly_seasonality', False)
    changepoint_prior = settings['model']['prophet'].get('changepoint_prior_scale', 0.05)
    seasonality_prior = settings['model']['prophet'].get('seasonality_prior_scale', 10.0)
    seasonality_mode = settings['model']['prophet'].get('seasonality_mode', 'multiplicative')
    regressor_mode = settings['model']['prophet'].get('regressor_mode', 'multiplicative')

    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        daily_seasonality=True,
        weekly_seasonality=False,
        changepoint_prior_scale=changepoint_prior,
        seasonality_prior_scale=seasonality_prior, 
        seasonality_mode = seasonality_mode
    )
    
    # Add Regressors
    prior_scale = settings['model']['prophet']['regressor_prior_scale']
    for reg_name in regressor_names:
        print(f"Adding regressor: {reg_name}")
        model.add_regressor(reg_name, mode=regressor_mode, prior_scale=prior_scale, standardize=False)
    
    print("Fitting model...")
    model.fit(df_prophet)
    
    # 4. Save Model
    model_path = settings['model']['path']
    # Ensure directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    print(f"Saving model to {model_path}...")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
        
    print("Training complete.")

if __name__ == "__main__":
    train_model()
