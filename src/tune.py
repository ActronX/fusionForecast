"""
Hyperparameter Tuning for FusionForecast

This script performs a grid search to optimize NeuralProphet hyperparameters,
using native NeuralProphet cross-validation for robust evaluation.

Based on NeuralProphet documentation:
- https://neuralprophet.com/how-to-guides/feature-guides/test_and_crossvalidate.html
- https://neuralprophet.com/how-to-guides/feature-guides/hyperparameter-selection.html
- https://neuralprophet.com/how-to-guides/feature-guides/sparse_autoregression_yosemite_temps.html
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

from src.config import settings
from src.db import InfluxDBWrapper
from src.data_loader import fetch_training_data

# Suppress excessive logging
set_log_level("ERROR")
logging.getLogger("NP.df_utils").setLevel(logging.ERROR)
logging.getLogger("NP.data.processing").setLevel(logging.ERROR)
logging.getLogger("NP").setLevel(logging.ERROR)


def run_tuning():
    print("=" * 70)
    print("NeuralProphet Hyperparameter Tuning")
    print("=" * 70)

    # 1. Load Data
    print("\n[1/4] Fetching training data...")
    data_result = fetch_training_data(verbose=False)
    
    if data_result is None:
        print("Error: No training data found.")
        return
        
    df, regressor_names = data_result

    if df.empty:
        print("Error: Training dataframe is empty.")
        return
        
    # Handle missing values robustly
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.interpolate(method='linear', limit_direction='both')
    df = df.fillna(method='bfill').fillna(method='ffill')

    print(f"  → {len(df)} rows, Range: {df['ds'].min()} to {df['ds'].max()}")
    print(f"  → Regressors: {regressor_names}")

    # 2. Define Parameter Grid
    # Based on NeuralProphet documentation recommendations:
    # - n_lags should be >= n_forecasts for AR-Net
    # - ar_reg controls sparsity (0.1=light, 1.0=moderate, 10=extreme)
    # - ar_layers: [] = linear, [64] = 1 hidden layer
    print("\n[2/4] Defining parameter grid...")
    
    param_grid = {
        # AR-Net Configuration
        # n_lags >= n_forecasts recommended by docs
        'n_lags': [4, 8,48, 96, 192, 288],       # 1 day, 2 days, 3 days context (at 15-min intervals)
        'n_forecasts': [96],             # Fixed: 1 day ahead
        
        # AR-Net Architecture
        # [] = linear model, [64] = single hidden layer with 64 neurons
        'ar_layers': [[]],
        
        # AR Regularization (Sparsity)
        # From docs: 0.1 = light sparsity, 1.0 = moderate, 10 = extreme (unstable)
        'ar_reg': [0.1, 0.5, 1.0],
        
        # Training - learning_rate is now dynamic based on ar_layers
        # (set in model creation, not in grid)
        
        # Seasonality
        'seasonality_mode': ['additive'],
    }

    # Generate all combinations
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"  → Testing {len(combinations)} configurations")
    
    # Print parameter ranges
    for key, vals in param_grid.items():
        if len(vals) > 1:
            print(f"  → {key}: {vals}")

    # 3. Cross-Validation Grid Search
    print("\n[3/4] Running Cross-Validation Grid Search...")
    print("-" * 70)

    results = []
    max_power = settings['model']['preprocessing']['max_power_clip']

    for i, params in enumerate(combinations):
        print(f"\nConfig {i+1}/{len(combinations)}:")
        print(f"  n_lags={params['n_lags']}, ar_layers={params['ar_layers']}, ar_reg={params['ar_reg']}")
        
        try:
            # Dynamic learning rate based on ar_layers
            # Per Solar PV docs: lower LR (0.003) for deep AR-Net, higher (0.01) for linear
            ar_layers = params['ar_layers']
            learning_rate = 0.003 if ar_layers else 0.01
            
            # Create model with current parameters
            m = NeuralProphet(
                # Growth: 'off' recommended for pure AR-based PV forecasting
                growth='off',
                n_lags=params['n_lags'],
                n_forecasts=params['n_forecasts'],
                ar_layers=ar_layers,
                ar_reg=params['ar_reg'],
                learning_rate=learning_rate,
                
                # Standard settings from config
                yearly_seasonality=settings['model']['neuralprophet']['yearly_seasonality'],
                weekly_seasonality=settings['model']['neuralprophet']['weekly_seasonality'],
                daily_seasonality=settings['model']['neuralprophet']['daily_seasonality'],
                seasonality_mode=params['seasonality_mode'],
                loss_func="Huber",
                drop_missing=True,
            )
            
            # Add Regressors
            for reg in regressor_names:
                m.add_future_regressor(name=reg, mode=params['seasonality_mode'])

            # Use NeuralProphet's native cross-validation
            # k=5 folds, each fold uses 10% of data for validation
            # fold_overlap_pct=0 ensures distinct validation periods
            folds = m.crossvalidation_split_df(
                df, 
                freq='15min', 
                k=5,              # 5 folds
                fold_pct=0.10,    # 10% of data per fold
                fold_overlap_pct=0.0
            )
            
            fold_metrics = []
            
            for fold_idx, (df_train, df_val) in enumerate(folds):
                # Reinitialize model for each fold (required by NeuralProphet)
                m_fold = NeuralProphet(
                    growth='off',
                    n_lags=params['n_lags'],
                    n_forecasts=params['n_forecasts'],
                    ar_layers=ar_layers,
                    ar_reg=params['ar_reg'],
                    learning_rate=learning_rate,
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
                m_fold.fit(df_train, freq='15min', progress='bar')
                
                # Use native test() method for evaluation
                test_metrics = m_fold.test(df_val)
                
                # Extract metrics from test results
                if 'MAE_val' in test_metrics.columns and 'RMSE_val' in test_metrics.columns:
                    mae = test_metrics['MAE_val'].iloc[-1]
                    rmse = test_metrics['RMSE_val'].iloc[-1]
                else:
                    # Fallback: manual calculation
                    forecast = m_fold.predict(df_val)
                    merged = pd.merge(df_val[['ds', 'y']], forecast[['ds', 'yhat1']], on='ds', how='inner')
                    y_true = merged['y'].values
                    y_pred = merged['yhat1'].values
                    
                    mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & (y_true != 0)
                    y_true_clean = y_true[mask]
                    y_pred_clean = y_pred[mask]
                    
                    if len(y_true_clean) == 0:
                        continue
                    
                    rmse = np.sqrt(np.mean((y_true_clean - y_pred_clean) ** 2))
                    mae = np.mean(np.abs(y_true_clean - y_pred_clean))
                
                nrmse = rmse / max_power
                nmae = mae / max_power
                score = 0.5 * nrmse + 0.5 * nmae
                
                fold_metrics.append({
                    'fold': fold_idx + 1,
                    'rmse': rmse,
                    'mae': mae,
                    'nrmse': nrmse,
                    'nmae': nmae,
                    'score': score
                })
                
                print(f"    Fold {fold_idx+1}: RMSE={rmse:.2f}, MAE={mae:.2f}, Score={score:.4f}")
            
            if len(fold_metrics) == 0:
                print(f"  ⚠ Warning: No valid predictions")
                continue
            
            # Aggregate across folds
            avg_rmse = np.mean([m['rmse'] for m in fold_metrics])
            avg_mae = np.mean([m['mae'] for m in fold_metrics])
            avg_score = np.mean([m['score'] for m in fold_metrics])
            std_score = np.std([m['score'] for m in fold_metrics])
            
            print(f"  → AVG: RMSE={avg_rmse:.2f}, MAE={avg_mae:.2f}, Score={avg_score:.4f} ± {std_score:.4f}")
            
            results.append({
                **params,
                'rmse': avg_rmse,
                'mae': avg_mae,
                'nrmse': avg_rmse / max_power,
                'nmae': avg_mae / max_power,
                'score': avg_score,
                'score_std': std_score,
                'n_folds': len(fold_metrics)
            })

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()

    # 4. Results Analysis
    print("\n" + "=" * 70)
    print("[4/4] Results Summary")
    print("=" * 70)
    
    if not results:
        print("No results generated.")
        return

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('score').reset_index(drop=True)
    
    # Display top 5 configurations
    print("\nTop 5 Configurations (lower score = better):")
    print("-" * 70)
    
    display_cols = ['n_lags', 'ar_layers', 'ar_reg', 'rmse', 'mae', 'score', 'score_std']
    print(results_df[display_cols].head(10).to_string(index=False))
    
    # Best configuration recommendation
    best = results_df.iloc[0]
    print("\n" + "=" * 70)
    print("BEST CONFIGURATION:")
    print("=" * 70)
    print(f"  n_lags:     {best['n_lags']}")
    print(f"  n_forecasts: {best['n_forecasts']}")
    print(f"  ar_layers:  {best['ar_layers']}")
    print(f"  ar_reg:     {best['ar_reg']}")
    print(f"  Score:      {best['score']:.4f} ± {best['score_std']:.4f}")
    print(f"  RMSE:       {best['rmse']:.2f}")
    print(f"  MAE:        {best['mae']:.2f}")
    
    # Save results
    results_df.to_csv("tuning_results.csv", index=False)
    print(f"\nFull results saved to 'tuning_results.csv'")
    
    # Provide settings.toml update suggestion
    print("\n" + "-" * 70)
    print("Suggested settings.toml update:")
    print("-" * 70)
    print(f"""
[model.neuralprophet]
n_lags = {best['n_lags']}
n_forecasts = {best['n_forecasts']}
ar_layers = {best['ar_layers']}
ar_reg = {best['ar_reg']}
""")


if __name__ == "__main__":
    run_tuning()
