
import os
import sys
import neuralprophet
from src.config import settings

def check_model():
    model_path = settings['model']['path']
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    print(f"Loading model from: {model_path}")
    model = neuralprophet.load(model_path)
    
    # Check future regressors in model
    # NeuralProphet changed config structure in recent versions
    if hasattr(model, 'config_regressors'):
        conf = model.config_regressors
        if conf is not None:
             # Try common attributes
             reg_dict = {}
             if hasattr(conf, 'regressors'):
                 reg_dict = conf.regressors
             elif hasattr(conf, 'model_config'):
                 reg_dict = conf.model_config
             
             if reg_dict:
                 print(f"Model expects these regressors: {list(reg_dict.keys())}")
             else:
                 print(f"Model config object found ({type(conf)}), but could not extract regressor keys.")
                 print(f"Available attributes: {dir(conf)}")
        else:
            print("Model has no future regressors configured.")
    
    # Check current settings
    reg_history = settings['influxdb']['fields']['regressor_history']
    print(f"Current settings.toml regressors: {reg_history}")

if __name__ == "__main__":
    check_model()
