import os
import sys
import torch
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import webbrowser
import warnings
import logging

# Add project root to path
sys.path.append(os.getcwd())

from src.config import settings
from src.db import InfluxDBWrapper
from src.data_loader import fetch_intraday_data, fetch_future_regressors

# Suppress logs and warnings
logging.getLogger("NP.df_utils").setLevel(logging.ERROR)
logging.getLogger("NP.data.processing").setLevel(logging.ERROR)
logging.getLogger("NP").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

def plot_intraday_ar():
    print("----------------------------------------------------------------")
    print("Starting Intraday AR Plot (Standalone)")
    print("----------------------------------------------------------------")

    # 1. Load Model
    model_path = settings['model']['path']
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    model = torch.load(model_path, weights_only=False)
    if hasattr(model, 'restore_trainer'):
        model.restore_trainer()

    n_lags = getattr(model, 'n_lags', 0)
    n_forecasts = getattr(model, 'n_forecasts', 1)
    print(f"Model Config: n_lags={n_lags}, n_forecasts={n_forecasts}")

    if n_lags == 0:
        print("Error: Model does not use autoregression (n_lags=0). Cannot plot AR impact.")
        return

    # 2. Data Fetching
    db = InfluxDBWrapper()
    
    # We want +/- 1 day (24h)
    # Fetch more history (48h) to ensure sufficient context for NeuralProphet
    history_hours = 48
    future_days = 1
    
    print(f"Fetching data (History: {history_hours}h, Future: {future_days}d)...")

    # A. Fetch Intraday History
    # Note: We need regressor names to fetch correct history columns
    reg_config = settings['influxdb']['fields']['regressor_history']
    regressor_fields = reg_config if isinstance(reg_config, list) else [reg_config]
    
    df_hist = fetch_intraday_data(db, fetch_hours=history_hours, regressor_fields=regressor_fields)
    
    if df_hist.empty:
        print("Error: No historical data found.")
        return

    # B. Fetch Future Regressors
    df_future = fetch_future_regressors(db, forecast_days=future_days)
    
    if df_future.empty:
        print("Error: No future regressor data found.")
        return

    # 3. Recursive Simulation (Gap Filling & Future Prediction)
    print("Preparing recursive simulation...")
    
    # Combined dataframe
    # History has 'y', Future has Regressors but no 'y'
    full_df = pd.concat([df_hist, df_future], ignore_index=True).sort_values('ds').reset_index(drop=True)
    
    # Determine simulation range
    # We need to simulate from the end of valid history up to the end of our future window
    last_hist_idx = df_hist.index[-1] # Index in df_hist, but we need index in full_df
    
    # Correctly find where history ends in full_df
    # Since we concat, history is at the beginning. 
    # Valid history ends where 'y' is not null (assuming future 'y' is NaN or we force it)
    
    # Ensure future 'y' is NaN
    full_df.loc[len(df_hist):, 'y'] = np.nan
    
    # Verify last valid index
    last_valid_idx = full_df[full_df['y'].notna()].index[-1]
    
    print(f"History ends at index {last_valid_idx} ({full_df.loc[last_valid_idx, 'ds']})")
    
    # Simulation loop
    simulated_df = full_df.copy()
    future_indices = simulated_df.index[last_valid_idx + 1:].tolist()
    
    print(f"Simulating {len(future_indices)} future steps...")
    
    for i in future_indices:
        # Input slice: up to i (INCLUSIVE) to force prediction for row i
        slice_df = simulated_df.iloc[:i+1].copy()
        
        try:
            fcst = model.predict(slice_df)
            y_pred = fcst.iloc[-1]['yhat1'] # Prediction for step i
            simulated_df.loc[i, 'y'] = y_pred
                 
        except Exception as e:
            print(f"Simulation failed at step {i}: {e}")
            import traceback
            traceback.print_exc()
            break
            
    # 4. Workaround for model truncation
    # NeuralProphet with n_forecasts > 1 drops the last n_forecasts rows
    # We pad with dummy rows to compensate
    print(f"Padding dataframe with {n_forecasts} dummy rows to compensate for model truncation...")
    
    # Create padding rows
    last_ds = simulated_df['ds'].max()
    padding_rows = []
    for i in range(1, n_forecasts + 1):
        next_ds = last_ds + pd.Timedelta(f'{i * 15}min')
        padding_row = {'ds': next_ds, 'y': 0}  # y value doesn't matter, will be dropped
        # Add regressor columns with forward-filled or dummy values
        for col in simulated_df.columns:
            if col not in ['ds', 'y']:
                padding_row[col] = simulated_df[col].iloc[-1]  # Forward fill last value
        padding_rows.append(padding_row)
    
    padding_df = pd.DataFrame(padding_rows)
    padded_df = pd.concat([simulated_df, padding_df], ignore_index=True)
    
    print(f"Padded dataframe shape: {padded_df.shape}")

    # 5. Generate Components (Decomposition)
    print("Generating decomposition...")
    forecast = model.predict(padded_df, decompose=True)
    
    # The forecast should now have the rows we need (truncated back to original range)
    print(f"Forecast generated with {len(forecast)} data points")

    # 6. Plotting
    print("Creating AR Impact Plot...")
    
    # Identify AR column
    ar_col_name = None
    if 'ar1' in forecast.columns: ar_col_name = 'ar1'
    elif 'ar' in forecast.columns: ar_col_name = 'ar'
    
    if not ar_col_name:
        print("Error: AR component column not found in forecast dataframe.")
        return

    # Filter Plot Data (+/- 24h from now)
    now = pd.Timestamp.utcnow().replace(tzinfo=None)
    plot_start = now - pd.Timedelta('24h')
    plot_end = now + pd.Timedelta('24h')
    
    mask = (forecast['ds'] >= plot_start) & (forecast['ds'] <= plot_end)
    df_plot = forecast.loc[mask].copy()
    
    if df_plot.empty:
        print("Warning: No data in plot window (+/- 24h).")
        return
        
    future_count = len(df_plot[df_plot['ds'] > now])
    print(f"Plot range: {df_plot['ds'].min()} to {df_plot['ds'].max()} ({len(df_plot)} points, {future_count} future)")

    fig = go.Figure()

    # AR Impact Trace
    fig.add_trace(go.Scatter(
        x=df_plot['ds'], 
        y=df_plot[ar_col_name],
        name='Autoregression Impact',
        line=dict(color='#FF5722', width=2), # Vibrancy
        fill='tozeroy',
        yaxis='y'
    ))
    
    # Context (Actual + Simulated up to now)
    # Use the 'y' column from forecast, which includes:
    # - Database values (historical)
    # - Simulated values from recursive loop (gap-fill up to now)
    df_context = df_plot[df_plot['ds'] <= now].copy()
    
    if not df_context.empty and 'y' in df_context.columns:
        fig.add_trace(go.Scatter(
             x=df_context['ds'], 
             y=df_context['y'], 
             name='Context (Actual)', 
             line=dict(color='black', width=2),
             opacity=0.7
         ))

    # Forecast (Total)
    if 'yhat1' in df_plot.columns:
         # We want to distinguish history fit vs future forecast
         # Future starts at 'now'
         df_fcst_future = df_plot[df_plot['ds'] > now]
         df_fcst_past = df_plot[df_plot['ds'] <= now]
         
         # Plot Past Fit (optional, maybe faint)
         # fig.add_trace(go.Scatter(x=df_fcst_past['ds'], y=df_fcst_past['yhat1'], name='Model Fit', line=dict(color='gray', width=1, dash='dot'), opacity=0.3))
         
         # Plot Future Forecast
         if not df_fcst_future.empty:
             fig.add_trace(go.Scatter(
                 x=df_fcst_future['ds'], 
                 y=df_fcst_future['yhat1'], 
                 name='Forecast (Future)', 
                 line=dict(color='#2196F3', width=3), 
                 opacity=0.9
             ))

    # Layout
    fig.update_layout(
        title="Intraday Autoregression Impact (+/- 24h)",
        yaxis_title="Correction (Watts)",
        xaxis_title="Time (UTC)",
        template="plotly_white",
        hovermode="x unified"
    )
    
    # Add "Now" vertical line as a shape manually to avoid plotly annotation issues
    fig.add_shape(
        type="line",
        x0=now, x1=now,
        y0=0, y1=1,
        yref="paper",
        line=dict(color="black", width=2, dash="dash")
    )
    
    # Add "Now" annotation
    fig.add_annotation(
        x=now,
        y=1.05,
        yref="paper",
        text="Now",
        showarrow=False,
        font=dict(size=12, color="black")
    )

    # Save
    output_file = os.path.abspath("plot_intraday_ar.html")
    fig.write_html(output_file)
    print(f"Plot saved to: {output_file}")
    
    # Open
    webbrowser.open(f"file://{output_file}")

if __name__ == "__main__":
    plot_intraday_ar()
