"""
Training Data Visualization Script for FusionForecast

Reads the exported training data CSV and generates interactive Plotly charts:
- Production (y) over time
- Each regressor overlaid on a secondary axis
- Correlation scatter plots between production and regressors

Usage:
    python -m src.plot_training_data
    python -m src.plot_training_data --csv path/to/custom.csv
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


def plot_training_data(csv_path=None):
    """Load training CSV and generate interactive plots."""

    # Resolve CSV path
    if csv_path is None:
        csv_path = settings['model'].get('export_training_csv', 'exports/training_data.csv')

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at '{csv_path}'")
        print("Run training with 'export_training_csv' enabled first.")
        return

    # Load data
    print(f"Loading training data from: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=['ds'])

    # Identify regressor columns (everything except ds and y)
    regressor_cols = [col for col in df.columns if col not in ('ds', 'y')]

    print(f"  Rows: {len(df)}")
    print(f"  Range: {df['ds'].min()} → {df['ds'].max()}")
    print(f"  Regressors: {regressor_cols}")

    # Filter to daytime only (y > 0 or any regressor > 0) for cleaner plots
    daytime_mask = (df['y'] > 0) | (df[regressor_cols].sum(axis=1) > 0)

    # ── Chart 1: Overview (Production + Regressors) ──────────────────
    n_regressors = len(regressor_cols)
    fig = make_subplots(
        rows=1 + n_regressors, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=['Production (y) [W]'] + [f'{col}' for col in regressor_cols],
        row_heights=[0.4] + [0.6 / max(n_regressors, 1)] * n_regressors
    )

    # Production trace
    fig.add_trace(
        go.Scatter(
            x=df['ds'], y=df['y'],
            name='Production (y)',
            line=dict(color='#FF6B35', width=1),
            fill='tozeroy',
            fillcolor='rgba(255, 107, 53, 0.15)'
        ),
        row=1, col=1
    )

    # Regressor traces
    colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
    for i, col in enumerate(regressor_cols):
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=df['ds'], y=df[col],
                name=col,
                line=dict(color=color, width=1),
                fill='tozeroy',
                fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.15)'
            ),
            row=2 + i, col=1
        )

    fig.update_layout(
        title=dict(text='FusionForecast – Training Data Overview', font=dict(size=20)),
        height=300 + 200 * n_regressors,
        template='plotly_dark',
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified'
    )

    # ── Chart 2: Correlation Scatter Plots ───────────────────────────
    if regressor_cols:
        df_day = df[daytime_mask].copy()

        fig_corr = make_subplots(
            rows=1, cols=n_regressors,
            subplot_titles=[f'y vs {col}' for col in regressor_cols]
        )

        for i, col in enumerate(regressor_cols):
            color = colors[i % len(colors)]
            correlation = df_day['y'].corr(df_day[col])
            fig_corr.add_trace(
                go.Scatter(
                    x=df_day[col], y=df_day['y'],
                    mode='markers',
                    name=f'{col} (r={correlation:.3f})',
                    marker=dict(color=color, size=3, opacity=0.4)
                ),
                row=1, col=1 + i
            )
            fig_corr.update_xaxes(title_text=col, row=1, col=1 + i)
            fig_corr.update_yaxes(title_text='Production [W]', row=1, col=1 + i)

        fig_corr.update_layout(
            title=dict(text='Regressor Correlation (Daytime Only)', font=dict(size=20)),
            height=500,
            width=500 * max(n_regressors, 1),
            template='plotly_dark',
            showlegend=True
        )

    # ── Save & Open ──────────────────────────────────────────────────
    output_dir = 'exports'
    os.makedirs(output_dir, exist_ok=True)

    overview_path = os.path.join(output_dir, 'training_data_overview.html')
    fig.write_html(overview_path)
    print(f"Overview chart saved: {overview_path}")

    if regressor_cols:
        corr_path = os.path.join(output_dir, 'training_data_correlation.html')
        fig_corr.write_html(corr_path)
        print(f"Correlation chart saved: {corr_path}")

    # Open in browser
    webbrowser.open(f'file://{os.path.abspath(overview_path)}')
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot FusionForecast training data')
    parser.add_argument('--csv', type=str, default=None, help='Path to training data CSV')
    args = parser.parse_args()
    plot_training_data(csv_path=args.csv)
