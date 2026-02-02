"""
Hyperparameter Tuning for FusionForecast

This script performs a grid search to optimize NeuralProphet hyperparameters,
specifically focusing on AR-Net architecture (ar_layers) and regularization (ar_reg).

It uses a time-series cross-validation approach to evaluate model performance.
"""

import os
import sys
import itertools
import pandas as pd
import numpy as np
import logging
import warnings
from neuralprophet import NeuralProphet, set_log_level

# Suppress warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(os.getcwd())

from src.config import settings
from src.db import InfluxDBWrapper
from src.data_loader import fetch_training_data

# Suppress excessive logging
set_log_level("ERROR")
logging.getLogger("NP.df_utils").setLevel(logging.ERROR)
logging.getLogger("NP.data.processing").setLevel(logging.ERROR)
logging.getLogger("NP").setLevel(logging.ERROR)

def run_tuning():
    print("----------------------------------------------------------------")
    print("Starting Hyperparameter Tuning")
    print("----------------------------------------------------------------")

    # 1. Load Data
    print("Fetching training data...")
    # db is instantiated inside fetch_training_data
    data_result = fetch_training_data(verbose=False)
    
    if data_result is None:
        print("Error: No training data found.")
        return
        
    df, regressor_names = data_result # Unpack tuple

    if df.empty:
        print("Error: Training dataframe is empty.")
        return
        
    # Handle missing values robustly
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.interpolate(method='linear', limit_direction='both')
    df = df.fillna(method='bfill').fillna(method='ffill')

    print(f"Stats: {len(df)} rows, Range: {df['ds'].min()} to {df['ds'].max()}")
    print(f"Regressors: {regressor_names}")

    # 2. Define Parameter Grid
    # Testing interaction between Model Capacity (ar_layers) and Regularization (ar_reg)
    param_grid = {
        'n_lags': [96],
        'n_forecasts': [96],
        'ar_layers': [[]],                  # Fixed: Linear Model
        'ar_reg': [0.05],                   # Fixed: Good baseline
        'learning_rate': [0.01],            # Fixed: Good baseline
        'seasonality_mode': ['additive', 'multiplicative'],
    }

    # Generate all combinations
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Testing {len(combinations)} model configurations...")
    print("----------------------------------------------------------------")

    results = []

    # 3. Grid Search Loop
    for i, params in enumerate(combinations):
        print(f"Run {i+1}/{len(combinations)}: {params}")
        
        try:
            # Initialize Model
            m = NeuralProphet(
                n_lags=params['n_lags'],
                n_forecasts=params['n_forecasts'],
                ar_layers=params['ar_layers'],
                ar_reg=params['ar_reg'],
                learning_rate=params['learning_rate'],
                
                # Standard settings from config
                yearly_seasonality=settings['model']['neuralprophet']['yearly_seasonality'],
                weekly_seasonality=settings['model']['neuralprophet']['weekly_seasonality'],
                daily_seasonality=settings['model']['neuralprophet']['daily_seasonality'],
                seasonality_mode=params['seasonality_mode'],
                loss_func="Huber",
                drop_missing=True, # Safety fallback
            )
            
            # Add Regressors
            for reg in regressor_names:
                m.add_future_regressor(name=reg, mode=params['seasonality_mode'])

            # Train-Test Split (Last 7 days for validation)
            # Use split_df for simple validation to save time. 
            # Could use crossvalidation_split_df for more robustness but slower.
            df_train, df_val = m.split_df(df, freq='15min', valid_p=0.1)

            # Train
            metrics = m.fit(df_train, freq='15min', validation_df=df_val, progress='bar')
            
            # Get best validation loss
            val_loss = metrics['Loss_val'].min()
            
            print(f"  > Best Val Loss: {val_loss:.5f}")
            
            results.append({
                **params,
                'val_loss': val_loss
            })

        except Exception as e:
            print(f"  > Error: {e}")

    # 4. Analysis
    print("----------------------------------------------------------------")
    print("Tuning Completed. Top 3 Configurations:")
    
    if not results:
        print("No results generated.")
        return

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('val_loss')
    
    print(results_df.head(3))
    
    # Save results
    results_df.to_csv("tuning_results.csv", index=False)
    print("\nFull results saved to 'tuning_results.csv'")

if __name__ == "__main__":
    run_tuning()
