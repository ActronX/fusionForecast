"""
Forecast Data Visualization Script for FusionForecast

Reads the exported forecast results CSV and generates interactive Plotly charts:
- PV Production (yhat) over time
- History production (y) as comparison if available
- Future regressors overlaid

Usage:
    python -m src.plot_forecast_data
    python -m src.plot_forecast_data --csv exports/forecast_data.csv
"""

import os
import sys
import argparse
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import webbrowser

sys.path.append(os.getcwd())
from src.config import settings


def plot_forecast_data(csv_path=None):
    """Load forecast CSVs and generate interactive plots."""

    # Resolve CSV base path
    if csv_path is None:
        csv_path = settings['model'].get('export_forecast_csv', 'exports/forecast_data.csv')

    base, ext = os.path.splitext(csv_path)
    results_path = f"{base}_results{ext}"
    history_path = f"{base}_history{ext}"

    if not os.path.exists(results_path):
        print(f"Error: Forecast results file not found at '{results_path}'")
        print("Run forecasting with 'export_forecast_csv' enabled first.")
        return

    # Load data
    print(f"Loading forecast results from: {results_path}")
    df_res = pd.read_csv(results_path, parse_dates=['ds'])
    
    df_hist = None
    if os.path.exists(history_path):
        print(f"Loading history context from: {history_path}")
        df_hist = pd.read_csv(history_path, parse_dates=['ds'])

    # Identify regressor columns (everything except ds, yhat, y)
    exclude = ('ds', 'yhat', 'y', 'time')
    regressor_cols = [col for col in df_res.columns if col not in exclude]

    print(f"  Forecast rows: {len(df_res)}")
    if df_hist is not None:
        print(f"  History rows: {len(df_hist)}")
    print(f"  Regressors: {regressor_cols}")

    # ── Chart: Forecast Overview ────────────────────────────────────
    n_regressors = len(regressor_cols)
    fig = make_subplots(
        rows=1 + n_regressors, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=['PV Production [W]'] + [f'{col}' for col in regressor_cols],
        row_heights=[0.4] + [0.6 / max(n_regressors, 1)] * n_regressors
    )

    # History Trace (if available) - plotted on first row
    if df_hist is not None:
        fig.add_trace(
            go.Scatter(
                x=df_hist['ds'], y=df_hist['y'],
                name='History (y)',
                line=dict(color='#888888', width=1.5, dash='dot'),
                opacity=0.7
            ),
            row=1, col=1
        )

    # Forecast Trace
    fig.add_trace(
        go.Scatter(
            x=df_res['ds'], y=df_res['yhat'],
            name='Forecast (yhat)',
            line=dict(color='#FF6B35', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 107, 53, 0.15)'
        ),
        row=1, col=1
    )

    # Regressor traces
    colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
    for i, col in enumerate(regressor_cols):
        color = colors[i % len(colors)]
        
        # Plot future regressor
        fig.add_trace(
            go.Scatter(
                x=df_res['ds'], y=df_res[col],
                name=f'{col} (Forecast)',
                line=dict(color=color, width=1),
                fill='tozeroy',
                fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1) '
            ),
            row=2 + i, col=1
        )
        
        # Plot historical regressor if available
        if df_hist is not None and col in df_hist.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_hist['ds'], y=df_hist[col],
                    name=f'{col} (History)',
                    line=dict(color=color, width=1, dash='dot'),
                    opacity=0.5,
                    showlegend=False
                ),
                row=2 + i, col=1
            )

    fig.update_layout(
        title=dict(text='FusionForecast – Forecast Results Overview', font=dict(size=20)),
        height=300 + 200 * n_regressors,
        template='plotly_dark',
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified'
    )

    # ── Save & Open ──────────────────────────────────────────────────
    output_dir = 'exports'
    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(output_dir, 'forecast_data_overview.html')
    fig.write_html(out_path)
    print(f"Chart saved: {out_path}")

    # Open in browser
    webbrowser.open(f'file://{os.path.abspath(out_path)}')
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot FusionForecast forecast data')
    parser.add_argument('--csv', type=str, default=None, help='Base path to forecast CSV')
    args = parser.parse_args()
    plot_forecast_data(csv_path=args.csv)
