"""
Intraday AR Impact Visualization

Shows forecast with vs without autoregression to visualize AR component impact.
Improvements based on NeuralProphet best practices.

Features:
- Native NeuralProphet prediction (no slow step-by-step simulation)
- Multi-step AR visualization
- Performance metrics (RMSE, MAE)
- Interactive range slider
- Component breakdown
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

import neuralprophet
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import webbrowser

sys.path.append(os.getcwd())

from src.config import settings
from src.db import InfluxDBWrapper
from src.data_loader import fetch_intraday_data, fetch_future_regressors


# ============================================================================
# Configuration
# ============================================================================

FUTURE_DAYS = 1         # Predict 1 day ahead
PLOT_WINDOW_HOURS = 24  # Show +/- 24h from now


# ============================================================================
# Helper Functions
# ============================================================================

def load_model(model_path):
    """Load NeuralProphet model using native load function."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Use NeuralProphet's native load (safer than torch.load)
    model = neuralprophet.load(model_path)
    return model


def prepare_prediction_dataframe(model, df_hist, df_future):
    """
    Prepare dataframe for prediction using native NeuralProphet methods.
    
    This is MUCH faster than step-by-step simulation.
    """
    # Remove timestamp overlap
    last_hist_time = df_hist['ds'].max()
    df_future_clean = df_future[df_future['ds'] > last_hist_time].copy()
    
    if df_future_clean.empty:
        print("Warning: No future data after filtering overlaps")
        return None
    
    # Use NeuralProphet's native make_future_dataframe
    try:
        future = model.make_future_dataframe(
            df=df_hist,
            regressors_df=df_future_clean,
            periods=len(df_future_clean),
            n_historic_predictions=True
        )
        return future
    except Exception as e:
        print(f"Error creating future dataframe: {e}")
        return None


def extract_diagonal_forecast(forecast, n_lags, n_forecasts):
    """
    Extract diagonal yhat values for multi-step forecasting.
    
    NeuralProphet with n_forecasts>1 outputs yhat1..yhatN for each row.
    For proper future forecasting, we need to extract the "diagonal":
    - Row at n_lags+0: use yhat1 (1-step ahead)
    - Row at n_lags+1: use yhat2 (2-step ahead)
    - etc.
    
    This fills in the 'yhat_combined' column with the correct forecast value.
    """
    result = forecast.copy()
    result['yhat_combined'] = np.nan
    result['ar_combined'] = np.nan
    
    # For historical rows (with y values), use yhat1
    hist_mask = result['y'].notna()
    if 'yhat1' in result.columns:
        result.loc[hist_mask, 'yhat_combined'] = result.loc[hist_mask, 'yhat1']
    if 'ar1' in result.columns:
        result.loc[hist_mask, 'ar_combined'] = result.loc[hist_mask, 'ar1']
    
    # For future rows, extract diagonal values
    future_mask = result['y'].isna()
    future_indices = result.index[future_mask].tolist()
    
    for i, idx in enumerate(future_indices):
        step = (i % n_forecasts) + 1  # Which forecast step to use
        yhat_col = f'yhat{step}'
        ar_col = f'ar{step}'
        
        if yhat_col in result.columns:
            # Look back to find the row that predicted this timestamp
            source_idx = max(0, idx - step + 1)
            if source_idx in result.index and yhat_col in result.columns:
                result.loc[idx, 'yhat_combined'] = result.loc[source_idx, yhat_col]
        
        if ar_col in result.columns:
            source_idx = max(0, idx - step + 1)
            if source_idx in result.index:
                result.loc[idx, 'ar_combined'] = result.loc[source_idx, ar_col]
    
    return result


def filter_plot_window(df, window_hours):
    """Filter dataframe to +/- window_hours from now."""
    now = pd.Timestamp.utcnow().replace(tzinfo=None)
    start = now - pd.Timedelta(f'{window_hours}h')
    end = now + pd.Timedelta(f'{window_hours}h')
    
    mask = (df['ds'] >= start) & (df['ds'] <= end)
    return df.loc[mask].copy(), now


def calculate_metrics(df):
    """Calculate performance metrics for historical data."""
    # Filter to rows with actual values
    mask = df['y'].notna() & df['yhat1'].notna()
    df_valid = df.loc[mask]
    
    if len(df_valid) == 0:
        return None, None
    
    y_true = df_valid['y'].values
    y_pred = df_valid['yhat1'].values
    
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    
    return rmse, mae


def find_ar_columns(forecast):
    """Find all AR component columns in forecast."""
    ar_cols = []
    for col in forecast.columns:
        if col.startswith('ar') and col not in ['ar_layers', 'ar_reg']:
            ar_cols.append(col)
    return sorted(ar_cols)


def create_plot(df_plot, ar_cols, now, rmse, mae, n_forecasts):
    """
    Create comprehensive Plotly visualization.
    
    Features:
    - Actual vs Forecast comparison
    - AR Impact visualization
    - Baseline (without AR) comparison
    - Performance metrics
    - Interactive range slider
    """
    # Create subplots: main plot + AR breakdown
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=["Forecast: With vs Without AR", "AR Component Impact"],
        vertical_spacing=0.12,
        shared_xaxes=True
    )
    
    # ─────────────────────────────────────────────────────────────────
    # Row 1: Main Forecast Plot
    # ─────────────────────────────────────────────────────────────────
    
    # 1. Actual values (historical)
    df_actual = df_plot[df_plot['ds'] <= now]
    if not df_actual.empty and 'y' in df_actual.columns:
        fig.add_trace(go.Scatter(
            x=df_actual['ds'],
            y=df_actual['y'],
            name='Actual',
            line=dict(color='black', width=2),
            opacity=0.9
        ), row=1, col=1)
    
    # 2. Baseline Forecast (Total - AR component)
    # Use yhat_combined (which has diagonal values for future rows)
    if 'yhat_combined' in df_plot.columns and 'ar_combined' in df_plot.columns:
        df_plot['yhat_baseline'] = df_plot['yhat_combined'] - df_plot['ar_combined'].fillna(0)
        
        # Debug info
        print(f"  Debug: yhat_combined valid: {df_plot['yhat_combined'].notna().sum()}")
        print(f"  Debug: baseline valid: {df_plot['yhat_baseline'].notna().sum()}")
        
        # Show baseline for future only (cleaner visualization)
        df_future = df_plot[df_plot['ds'] > now]
        
        if not df_future.empty and df_future['yhat_baseline'].notna().any():
            fig.add_trace(go.Scatter(
                x=df_future['ds'],
                y=df_future['yhat_baseline'],
                name='Forecast (No AR)',
                line=dict(color='#9E9E9E', width=2, dash='dash'),
                opacity=0.7
            ), row=1, col=1)
        else:
            print("  Warning: No valid baseline data for future period")
    
    # 3. Full Forecast (With AR) - use yhat_combined for continuous line
    if 'yhat_combined' in df_plot.columns:
        fig.add_trace(go.Scatter(
            x=df_plot['ds'],
            y=df_plot['yhat_combined'],
            name='Forecast (With AR)',
            line=dict(color='#2196F3', width=3),
            opacity=0.9
        ), row=1, col=1)
    
    # ─────────────────────────────────────────────────────────────────
    # Row 2: AR Component Breakdown
    # ─────────────────────────────────────────────────────────────────
    
    # Show multiple AR steps if available
    colors = ['#FF5722', '#E91E63', '#9C27B0', '#673AB7']
    
    # Select representative AR columns (1st, 25%, 50%, last)
    if len(ar_cols) > 1:
        step_indices = [0]
        if len(ar_cols) >= 4:
            step_indices.append(len(ar_cols) // 4)
        if len(ar_cols) >= 2:
            step_indices.append(len(ar_cols) // 2)
        step_indices.append(len(ar_cols) - 1)
        step_indices = list(dict.fromkeys(step_indices))  # Remove duplicates
        
        selected_ar = [ar_cols[i] for i in step_indices if i < len(ar_cols)]
    else:
        selected_ar = ar_cols
    
    for i, ar_col in enumerate(selected_ar[:4]):
        # Extract step number
        step_num = ar_col.replace('ar', '') if ar_col != 'ar' else '1'
        step_label = f"AR Step {step_num} ({int(step_num)*15}min)" if step_num.isdigit() else ar_col
        
        fig.add_trace(go.Scatter(
            x=df_plot['ds'],
            y=df_plot[ar_col],
            name=step_label,
            line=dict(color=colors[i % len(colors)], width=2),
            fill='tozeroy' if i == 0 else None,
            fillcolor=f'rgba(255, 87, 34, 0.15)' if i == 0 else None
        ), row=2, col=1)
    
    # ─────────────────────────────────────────────────────────────────
    # Layout & Annotations
    # ─────────────────────────────────────────────────────────────────
    
    # Add "Now" marker (vertical line across full figure)
    # Convert timestamp to string to avoid pandas compatibility issues
    now_str = now.isoformat()
    fig.add_shape(
        type="line",
        x0=now_str, x1=now_str,
        y0=0, y1=1,
        yref="paper",
        line=dict(color="red", width=2, dash="dash")
    )
    fig.add_annotation(
        x=now_str, y=1.05, yref="paper",
        text="Now",
        showarrow=False,
        font=dict(size=12, color="red")
    )
    
    # Performance metrics annotation
    if rmse is not None:
        metrics_text = f"RMSE: {rmse:.0f}W | MAE: {mae:.0f}W"
        fig.add_annotation(
            x=0.02, y=0.98,
            xref="paper", yref="paper",
            text=metrics_text,
            showarrow=False,
            font=dict(size=14, color="black"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1,
            borderpad=4
        )
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f"Intraday AR Forecast (n_forecasts={n_forecasts})",
            font=dict(size=18)
        ),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=700
    )
    
    # Y-axis labels
    fig.update_yaxes(title_text="Power (W)", row=1, col=1)
    fig.update_yaxes(title_text="AR Impact (W)", row=2, col=1)
    
    # Interactive range slider
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=6, label="6h", step="hour", stepmode="backward"),
                dict(count=12, label="12h", step="hour", stepmode="backward"),
                dict(count=24, label="24h", step="hour", stepmode="backward"),
                dict(step="all", label="All")
            ]),
            font=dict(size=11)
        ),
        rangeslider=dict(visible=True, thickness=0.05),
        row=2, col=1
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
    print(f"\n[1/5] Loading model from {model_path}...")
    model = load_model(model_path)
    
    n_lags = getattr(model, 'n_lags', 0)
    n_forecasts = getattr(model, 'n_forecasts', 1)
    print(f"  Model: n_lags={n_lags}, n_forecasts={n_forecasts}")
    
    if n_lags == 0:
        raise ValueError("Model does not use autoregression (n_lags=0)")
    
    # Calculate history needed (n_lags with 2x buffer)
    hours_needed = (n_lags * 15) / 60
    history_hours = max(24, int(hours_needed * 2))
    
    # 2. Fetch Data
    print(f"\n[2/5] Fetching data (History: {history_hours}h, Future: {FUTURE_DAYS}d)...")
    db = InfluxDBWrapper()
    
    reg_config = settings['influxdb']['fields']['regressor_history']
    regressor_fields = reg_config if isinstance(reg_config, list) else [reg_config]
    
    df_hist = fetch_intraday_data(db, history_hours, regressor_fields)
    df_future = fetch_future_regressors(db, FUTURE_DAYS)
    
    if df_hist.empty:
        raise ValueError("Failed to fetch historical data")
    if df_future.empty:
        raise ValueError("Failed to fetch future regressor data")
    
    print(f"  Historical: {len(df_hist)} rows ({df_hist['ds'].min()} to {df_hist['ds'].max()})")
    print(f"  Future: {len(df_future)} rows")
    
    # 3. Prepare Prediction Dataframe (FAST native method)
    print("\n[3/5] Preparing prediction dataframe (native method)...")
    future_df = prepare_prediction_dataframe(model, df_hist, df_future)
    
    if future_df is None:
        raise ValueError("Failed to prepare prediction dataframe")
    
    print(f"  Prediction dataframe: {len(future_df)} rows")
    
    # 4. Generate Forecast
    print("\n[4/5] Generating forecast with decomposition...")
    try:
        forecast = model.predict(future_df, decompose=True)
        print(f"  Forecast complete: {len(forecast)} rows")
    except Exception as e:
        print(f"  Decomposition failed: {e}")
        forecast = model.predict(future_df, decompose=False)
    
    # 4b. Extract diagonal forecast values for multi-step predictions
    forecast = extract_diagonal_forecast(forecast, n_lags, n_forecasts)
    print(f"  yhat_combined NaN count: {forecast['yhat_combined'].isna().sum()}")
    
    # 5. Identify AR Columns
    ar_cols = find_ar_columns(forecast)
    if not ar_cols:
        raise ValueError("No AR component columns found in forecast")
    print(f"  AR columns found: {ar_cols[:5]}{'...' if len(ar_cols) > 5 else ''}")
    
    # 6. Filter to Plot Window
    df_plot, now = filter_plot_window(forecast, PLOT_WINDOW_HOURS)
    if df_plot.empty:
        raise ValueError(f"No data in plot window (+/- {PLOT_WINDOW_HOURS}h)")
    
    print(f"  Plot window: {df_plot['ds'].min()} to {df_plot['ds'].max()}")
    
    # 7. Calculate Metrics
    rmse, mae = calculate_metrics(df_plot)
    if rmse:
        print(f"  Performance: RMSE={rmse:.0f}W, MAE={mae:.0f}W")
    
    # 8. Create Visualization
    print("\n[5/5] Creating visualization...")
    fig = create_plot(df_plot, ar_cols, now, rmse, mae, n_forecasts)
    
    # 9. Save and Open
    output_file = os.path.abspath("plot_intraday_ar.html")
    fig.write_html(output_file)
    print(f"\n[OK] Saved to: {output_file}")
    
    webbrowser.open(f"file://{output_file}")
    print("=" * 70)


if __name__ == "__main__":
    plot_intraday_ar()
