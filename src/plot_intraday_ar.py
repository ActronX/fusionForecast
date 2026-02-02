"""
Intraday AR Impact Visualization
Shows forecast with vs without autoregression to visualize AR component impact.
"""

import os
import sys
import logging
import warnings

# Suppress ALL logging immediately
os.environ['NEURALPROPHET_MINIMAL_LOGGING'] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Configure critical loggers
for logger_name in ['NP', 'NP.df_utils', 'NP.data.processing', 'neuralprophet', 'pytorch_lightning']:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.CRITICAL)
    logger.propagate = False
    logger.disabled = True

logging.basicConfig(level=logging.CRITICAL)
warnings.filterwarnings("ignore")

import torch
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import webbrowser

sys.path.append(os.getcwd())

from src.config import settings
from src.db import InfluxDBWrapper
from src.data_loader import fetch_intraday_data, fetch_future_regressors


# ============================================================================
# Configuration
# ============================================================================

# HISTORY_HOURS is now calculated dynamically from model's n_lags
FUTURE_DAYS = 1     # Predict 1 day ahead
PLOT_WINDOW_HOURS = 24  # Show +/- 24h from now


# ============================================================================
# Helper Functions
# ============================================================================

def load_model(model_path):
    """Load NeuralProphet model from file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = torch.load(model_path, weights_only=False)
    if hasattr(model, 'restore_trainer'):
        model.restore_trainer()
    
    return model


def prepare_full_dataframe(df_hist, df_future):
    """
    Merge historical and future data, removing duplicates.
    
    Args:
        df_hist: Historical data with 'y' column
        df_future: Future regressor data
    
    Returns:
        pd.DataFrame: Combined dataframe with 'y' set to NaN for future
    """
    # Remove timestamp overlap to prevent duplicates
    last_hist_time = df_hist['ds'].max()
    df_future = df_future[df_future['ds'] > last_hist_time].copy()
    
    if df_future.empty:
        print("Warning: No future data after filtering overlaps")
        return df_hist.copy()
    
    # Merge
    full_df = pd.concat([df_hist, df_future], ignore_index=True)
    full_df = full_df.sort_values('ds').reset_index(drop=True)
    
    # Mark future as NaN
    full_df.loc[len(df_hist):, 'y'] = np.nan
    
    return full_df


def simulate_future(model, full_df):
    """
    Recursively simulate future predictions.
    
    For AR models, we need to fill 'y' values step-by-step using predictions.
    
    Args:
        model: NeuralProphet model
        full_df: Combined historical + future dataframe
    
    Returns:
        pd.DataFrame: Dataframe with simulated 'y' values
    """
    simulated = full_df.copy()
    last_valid_idx = simulated[simulated['y'].notna()].index[-1]
    future_indices = simulated.index[last_valid_idx + 1:].tolist()
    
    print(f"Simulating {len(future_indices)} future steps...")
    
    for i in future_indices:
        try:
            # Predict using all data up to and including row i
            slice_df = simulated.iloc[:i+1].copy()
            fcst = model.predict(slice_df)
            
            # Extract prediction for row i
            y_pred = fcst.iloc[-1]['yhat1']
            simulated.loc[i, 'y'] = y_pred
            
        except Exception as e:
            print(f"Simulation failed at step {i}: {e}")
            break
    
    return simulated


def pad_for_truncation(df, n_forecasts):
    """
    Add padding rows to compensate for NeuralProphet's truncation behavior.
    
    NeuralProphet with n_forecasts > 1 may drop last rows. We pad to ensure
    we get predictions for the full range.
    
    Args:
        df: Dataframe to pad
        n_forecasts: Number of padding rows
    
    Returns:
        pd.DataFrame: Padded dataframe
    """
    last_ds = df['ds'].max()
    padding_rows = []
    
    for i in range(1, n_forecasts + 1):
        next_ds = last_ds + pd.Timedelta(f'{i * 15}min')
        padding_row = {'ds': next_ds, 'y': 0}
        
        # Forward-fill regressor values
        for col in df.columns:
            if col not in ['ds', 'y']:
                padding_row[col] = df[col].iloc[-1]
        
        padding_rows.append(padding_row)
    
    padding_df = pd.DataFrame(padding_rows)
    return pd.concat([df, padding_df], ignore_index=True)


def filter_plot_window(df, window_hours):
    """
    Filter dataframe to +/- window_hours from now.
    
    Args:
        df: Dataframe with 'ds' column
        window_hours: Hours before/after now
    
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    now = pd.Timestamp.utcnow().replace(tzinfo=None)
    start = now - pd.Timedelta(f'{window_hours}h')
    end = now + pd.Timedelta(f'{window_hours}h')
    
    mask = (df['ds'] >= start) & (df['ds'] <= end)
    return df.loc[mask].copy(), now


def calculate_baseline(df, ar_col):
    """
    Calculate baseline forecast (without AR).
    
    Baseline = Total Forecast - AR Component
    
    Args:
        df: Dataframe with 'yhat1' and AR column
        ar_col: Name of AR column
    
    Returns:
        pd.DataFrame: Dataframe with 'yhat_baseline' column added
    """
    if 'yhat1' in df.columns and ar_col in df.columns:
        df['yhat_baseline'] = df['yhat1'] - df[ar_col]
    return df


def create_plot(df_plot, ar_col, now):
    """
    Create Plotly visualization comparing forecasts with/without AR.
    
    Args:
        df_plot: Dataframe in plot window
        ar_col: Name of AR column
        now: Current timestamp
    
    Returns:
        go.Figure: Plotly figure
    """
    fig = go.Figure()
    
    # 1. AR Impact (filled area)
    fig.add_trace(go.Scatter(
        x=df_plot['ds'],
        y=df_plot[ar_col],
        name='AR Impact',
        line=dict(color='#FF5722', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 87, 34, 0.2)'
    ))
    
    # 2. Actual values (up to now)
    df_actual = df_plot[df_plot['ds'] <= now]
    if not df_actual.empty and 'y' in df_actual.columns:
        fig.add_trace(go.Scatter(
            x=df_actual['ds'],
            y=df_actual['y'],
            name='Actual',
            line=dict(color='black', width=2),
            opacity=0.8
        ))
    
    # 3. Baseline Forecast (No AR) - future only
    df_future = df_plot[df_plot['ds'] > now]
    if not df_future.empty and 'yhat_baseline' in df_plot.columns:
        fig.add_trace(go.Scatter(
            x=df_future['ds'],
            y=df_future['yhat_baseline'],
            name='Forecast (No AR)',
            line=dict(color='#9E9E9E', width=2, dash='dash'),
            opacity=0.6
        ))
    
    # 4. Full Forecast (With AR) - future only
    if not df_future.empty and 'yhat1' in df_plot.columns:
        fig.add_trace(go.Scatter(
            x=df_future['ds'],
            y=df_future['yhat1'],
            name='Forecast (With AR)',
            line=dict(color='#2196F3', width=3),
            opacity=0.9
        ))
    
    # Layout
    fig.update_layout(
        title="Intraday Forecast: With vs Without Autoregression",
        yaxis_title="Power (Watts)",
        xaxis_title="Time (UTC)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add "Now" marker
    fig.add_shape(
        type="line",
        x0=now, x1=now, y0=0, y1=1,
        yref="paper",
        line=dict(color="black", width=2, dash="dash")
    )
    
    fig.add_annotation(
        x=now, y=1.05, yref="paper",
        text="Now",
        showarrow=False,
        font=dict(size=12, color="black")
    )
    
    return fig


# ============================================================================
# Main Function
# ============================================================================

def plot_intraday_ar():
    """Main plotting pipeline."""
    print("=" * 70)
    print("Intraday AR Impact Visualization")
    print("=" * 70)
    
    # 1. Load Model
    model_path = settings['model']['path']
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    n_lags = getattr(model, 'n_lags', 0)
    n_forecasts = getattr(model, 'n_forecasts', 1)
    print(f"Model: n_lags={n_lags}, n_forecasts={n_forecasts}")
    
    if n_lags == 0:
        raise ValueError("Model does not use autoregression (n_lags=0)")
    
    # Calculate history hours dynamically from n_lags (with 2x safety buffer)
    hours_needed = (n_lags * 15) / 60  # n_lags * 15min -> hours
    history_hours = max(24, int(hours_needed * 2))  # At least 24h, with 2x buffer
    
    # 2. Fetch Data
    print(f"Fetching data (History: {history_hours}h based on n_lags={n_lags}, Future: {FUTURE_DAYS}d)...")
    db = InfluxDBWrapper()
    
    reg_config = settings['influxdb']['fields']['regressor_history']
    regressor_fields = reg_config if isinstance(reg_config, list) else [reg_config]
    
    df_hist = fetch_intraday_data(db, history_hours, regressor_fields)
    df_future = fetch_future_regressors(db, FUTURE_DAYS)
    
    if df_hist.empty or df_future.empty:
        raise ValueError("Failed to fetch historical or future data")
    
    # 3. Prepare Combined Dataframe
    print("Preparing combined dataframe...")
    full_df = prepare_full_dataframe(df_hist, df_future)
    
    # 4. Simulate Future (Recursive AR)
    simulated_df = simulate_future(model, full_df)
    
    # 5. Pad and Generate Forecast
    print(f"Padding with {n_forecasts} rows to prevent truncation...")
    padded_df = pad_for_truncation(simulated_df, n_forecasts)
    
    print("Generating forecast with decomposition...")
    forecast = model.predict(padded_df, decompose=True)
    print(f"Forecast complete: {len(forecast)} points")
    
    # 6. Identify AR Column
    ar_col = 'ar1' if 'ar1' in forecast.columns else 'ar' if 'ar' in forecast.columns else None
    if not ar_col:
        raise ValueError("AR component column not found in forecast")
    
    # 7. Filter to Plot Window
    df_plot, now = filter_plot_window(forecast, PLOT_WINDOW_HOURS)
    if df_plot.empty:
        raise ValueError(f"No data in plot window (+/- {PLOT_WINDOW_HOURS}h)")
    
    print(f"Plot window: {df_plot['ds'].min()} to {df_plot['ds'].max()}")
    
    # Debug AR values
    print(f"AR Column detected: {ar_col}")
    if ar_col is not None:
        ar_values = df_plot[ar_col].dropna()
        if ar_values.empty:
            print("WARNING: AR column is empty!")
        else:
            ar_mean = ar_values.abs().mean()
            ar_max = ar_values.abs().max()
            ar_min_val = ar_values.min()
            ar_max_val = ar_values.max()
            print(f"AR Stats in window - Mean Abs: {ar_mean:.4f}, Max Abs: {ar_max:.4f}")
            print(f"AR Range: {ar_min_val:.4f} to {ar_max_val:.4f}")
            print("First 5 AR values:")
            print(df_plot[['ds', ar_col]].head().to_string())

      
    # 8. Calculate Baseline
    df_plot = calculate_baseline(df_plot, ar_col)
    
    # 9. Create Visualization
    print("Creating visualization...")
    fig = create_plot(df_plot, ar_col, now)
    
    # Debug Baseline vs Forecast
    print(f"Baseline (No AR) Mean: {df_plot['yhat_baseline'].mean():.4f}")
    print(f"Forecast (With AR) Mean: {df_plot['yhat1'].mean():.4f}")
    
    # 10. Save and Open
    output_file = os.path.abspath("plot_intraday_ar.html")
    fig.write_html(output_file)
    print(f"[OK] Saved to: {output_file}")
    
    webbrowser.open(f"file://{output_file}")
    print("=" * 70)


if __name__ == "__main__":
    plot_intraday_ar()
