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
        'n_lags': [4, 8, 12, 24, 96],      # Test: 1h, 2h, 3h, 6h, 24h context
        'n_forecasts': [96],
        'ar_layers': [[]],                  # Fixed: Linear Model
        'ar_reg': [0.05],                   # Fixed: Good baseline
        'learning_rate': [0.01],            # Fixed: Good baseline
        'seasonality_mode': ['additive'],   # Fixed: Validated as better
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

            # 3-Fold Cross-Validation for robust evaluation
            # Test on different time periods to avoid statistical outliers
            n_folds = 3
            fold_scores = []
            fold_rmse = []
            fold_mae = []
            
            total_len = len(df)
            fold_size = total_len // (n_folds + 1)  # Reserve space for each fold
            
            for fold in range(n_folds):
                # Create different validation windows
                # Fold 0: Early period, Fold 1: Middle period, Fold 2: Late period
                val_start = fold_size * (fold + 1) - fold_size // 2
                val_end = val_start + fold_size
                
                df_val_fold = df.iloc[val_start:val_end].copy()
                df_train_fold = pd.concat([df.iloc[:val_start], df.iloc[val_end:]]).copy()
                
                # Reinitialize model for each fold
                m_fold = NeuralProphet(
                    n_lags=params['n_lags'],
                    n_forecasts=params['n_forecasts'],
                    ar_layers=params['ar_layers'],
                    ar_reg=params['ar_reg'],
                    learning_rate=params['learning_rate'],
                    yearly_seasonality=settings['model']['neuralprophet']['yearly_seasonality'],
                    weekly_seasonality=settings['model']['neuralprophet']['weekly_seasonality'],
                    daily_seasonality=settings['model']['neuralprophet']['daily_seasonality'],
                    seasonality_mode=params['seasonality_mode'],
                    loss_func="Huber",
                    drop_missing=True,
                )
                
                for reg in regressor_names:
                    m_fold.add_future_regressor(name=reg, mode=params['seasonality_mode'])
                
                # Train
                m_fold.fit(df_train_fold, freq='15min', progress='bar')
                
                # Predict
                forecast = m_fold.predict(df_val_fold)
                
                # Merge on 'ds' to handle length mismatch
                merged = pd.merge(df_val_fold[['ds', 'y']], forecast[['ds', 'yhat1']], on='ds', how='inner')
                y_true = merged['y'].values
                y_pred = merged['yhat1'].values
                
                mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & (y_true != 0)
                y_true_clean = y_true[mask]
                y_pred_clean = y_pred[mask]
                
                if len(y_true_clean) == 0:
                    continue
                
                rmse = np.sqrt(np.mean((y_true_clean - y_pred_clean) ** 2))
                mae = np.mean(np.abs(y_true_clean - y_pred_clean))
                max_power = settings['model']['preprocessing']['max_power_clip']
                nrmse = rmse / max_power
                nmae = mae / max_power
                score = 0.5 * nrmse + 0.5 * nmae
                
                fold_scores.append(score)
                fold_rmse.append(rmse)
                fold_mae.append(mae)
                print(f"    Fold {fold+1}: Score={score:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}")
            
            if len(fold_scores) == 0:
                print(f"  > Warning: No valid predictions for metrics")
                continue
            
            # Average across folds
            avg_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)
            avg_rmse = np.mean(fold_rmse)
            avg_mae = np.mean(fold_mae)
            
            print(f"  > AVG Score: {avg_score:.4f} ± {std_score:.4f}, RMSE: {avg_rmse:.2f}, MAE: {avg_mae:.2f}")
            
            results.append({
                **params,
                'rmse': avg_rmse,
                'mae': avg_mae,
                'nrmse': avg_rmse / max_power,
                'nmae': avg_mae / max_power,
                'score': avg_score,
                'score_std': std_score
            })

        except Exception as e:
            print(f"  > Error: {e}")

    # 4. Analysis
    print("----------------------------------------------------------------")
    print("Tuning Completed. All Configurations (sorted by Score, lower is better):")
    
    if not results:
        print("No results generated.")
        return

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('score').reset_index(drop=True)
    
    print(results_df.to_string())
    
    # Save results
    results_df.to_csv("tuning_results.csv", index=False)
    print("\nFull results saved to 'tuning_results.csv'")

if __name__ == "__main__":
    run_tuning()
