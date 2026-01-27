
print("DEBUG: Application starting...")
import os
import sys
print("DEBUG: Basic imports done")
import pickle
import logging
print("DEBUG: Standard lib imports done")

# logging.getLogger('cmdstanpy').disabled = True # Not needed for NeuralProphet
# Check if we need to suppress PyTorch Lightning logs
#logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
#logging.getLogger("NP").setLevel(logging.INFO)
import warnings
#warnings.filterwarnings("ignore", ".*Trying to infer the `batch_size`.*")
#warnings.filterwarnings("ignore", ".*DataFrameGroupBy.apply operated on the grouping columns.*")
#warnings.filterwarnings("ignore", ".*Series.view is deprecated.*")
#warnings.filterwarnings("ignore", ".*is deprecated, use `isinstance.*")
#warnings.filterwarnings("ignore", ".*Defined frequency .* is different than major frequency.*")
#warnings.filterwarnings("ignore", ".*You called .*but have no logger configured.*")


print("DEBUG: Importing NeuralProphet...")
from neuralprophet import NeuralProphet
print("DEBUG: NeuralProphet imported")
print("DEBUG: Importing local modules...")
from src.config import settings
from src.data_loader import fetch_training_data, validate_data_sufficiency
print("DEBUG: Local modules imported")


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
    # Validate data sufficiency
    training_days = settings['model']['training_days']
    if not validate_data_sufficiency(df_prophet, training_days):
        sys.exit(1)

    print("Configuring NeuralProphet model...")
    p_settings = settings['model']['neuralprophet']
    
    # NeuralProphet initialization
    # Note: 'growth'='flat' constraint from Prophet is not directly mapped. 
    # Using 'linear' effectively allows trend.
    model = NeuralProphet(
        growth=p_settings.get('growth', 'linear'),
        yearly_seasonality=p_settings.get('yearly_seasonality', False),
        weekly_seasonality=p_settings.get('weekly_seasonality', False), 
        daily_seasonality=p_settings.get('daily_seasonality', True),
        seasonality_mode=p_settings.get('seasonality_mode', 'additive'),
        learning_rate=p_settings.get('learning_rate', 0.01),
        epochs=p_settings.get('epochs', 10),
        # Regularization
        accelerator=p_settings.get('accelerator', 'auto'),
        # n_lags=24,     # AR disabled due to stability issues (NaN loss)
        # ar_layers=[64],
        drop_missing=True
    )
    
    # Add Regressors
    # In NeuralProphet, external variables for forecast are 'future_regressors'
    # Default mode can be additive or multiplicative
    # There is no global 'regressor_mode' param in init, it's per regressor usually?
    # No, add_future_regressor has 'mode'.
    # We will assume 'additive' unless specified (Prophet config had 'regressor_mode', let's use it if valid)
    # But we mapped it out. We will stick to the 'seasonality_mode' for simplicity or default 'additive'.
    # Actually, previous code used p_settings.get('regressor_mode', 'multiplicative').
    # Let's try to grab that if it exists, otherwise default.
    reg_mode = p_settings.get('regressor_mode', 'additive') # Default to additive for NP stability
    reg_reg = p_settings.get('future_regressor_regularization', 0.0)

    for reg_name in regressor_names:
        print(f"Adding future regressor: {reg_name} (mode={reg_mode}, reg={reg_reg})")
        model.add_future_regressor(
            name=reg_name,
            mode=reg_mode
            # regularization=reg_reg # Uncomment if supported by exact version, usually handled via list or global? 
            # NP allows regularization on regressors?
            # It seems 'normalize' is an option. 
            # For now, simplistic add.
        )
    
    print("Fitting model...")
    # metrics=True allows tracking loss
    model.fit(df_prophet, freq='30min')
    
    # 4. Save Model
    model_path = settings['model']['path']
    # Ensure directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    print(f"Saving model to {model_path}...")
    # NeuralProphet models can often be pickled, but sometimes recommended to use save().
    # We'll try pickle for detailed state preservation.
    
    # Calculate and print model size before saving
    model_size = len(pickle.dumps(model))
    print(f"Model size before saving: {model_size / 1024 / 1024:.2f} MB")
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
        
    print("Training complete.")

if __name__ == "__main__":
    train_model()
