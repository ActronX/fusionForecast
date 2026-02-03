"""
Model Visualization Script for FusionForecast

Generates comprehensive plots for NeuralProphet model analysis:
- Forecast Overview
- Component Decomposition
- Regressor Impact
- AR Parameters (per forecast step)
- Residuals Analysis
- Autocorrelation (optional)

Based on NeuralProphet documentation:
- https://neuralprophet.com/how-to-guides/feature-guides/plotly.html
- https://neuralprophet.com/tutorials/tutorial04.html
"""

import os
import pandas as pd
import numpy as np
import neuralprophet
from src.config import settings
from src.db import InfluxDBWrapper
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import webbrowser
import time

def plot_model():
    print("=" * 70)
    print("NeuralProphet Model Visualization")
    print("=" * 70)
    
    # 1. Load Model
    model_path = settings['model']['path']
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    print(f"\n[1/5] Loading model from {model_path}...")
    # Use NeuralProphet's native load function (safer than torch.load)
    model = neuralprophet.load(model_path)
    
    # Set plotting backend once
    model.set_plotting_backend("plotly")
    
    # Extract model configuration
    n_lags = getattr(model, 'n_lags', 0)
    n_forecasts = getattr(model, 'n_forecasts', 1)
    print(f"  Model: n_lags={n_lags}, n_forecasts={n_forecasts}")

    # 2. Configuration & Data Fetching
    db = InfluxDBWrapper()
    
    # User Request: 14 days history, 7 days future
    plot_history_days = 14 
    forecast_days = 7
    
    reg_config = settings['influxdb']['fields']['regressor_history']
    regressor_fields = reg_config if isinstance(reg_config, list) else [reg_config]
    
    regressor_scale = settings['model']['preprocessing'].get('regressor_scale', 1.0)
    produced_scale = settings['model']['preprocessing'].get('produced_scale', 1.0)
    
    print(f"\n[2/5] Fetching data (History: {plot_history_days}d, Future: {forecast_days}d)...")

    # A. Fetch History (Target + Regressors)
    range_start = f"-{plot_history_days + 5}d"
    
    # Fetch Target (Produced)
    print("  - Fetching Target History...")
    query_target = f'''
    from(bucket: "{settings['influxdb']['buckets']['history_produced']}")
      |> range(start: {range_start})
      |> filter(fn: (r) => r["_measurement"] == "{settings['influxdb']['measurements']['produced']}")
      |> filter(fn: (r) => r["_field"] == "{settings['influxdb']['fields']['produced']}")
      |> map(fn: (r) => ({{ r with _value: r._value * {produced_scale} }}))
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    df_result_target = db.query_dataframe(query_target)
    
    if df_result_target.empty:
        print("Error: No target history found.")
        return

    df_result_target['ds'] = pd.to_datetime(df_result_target['_time']).dt.tz_localize(None)
    target_col = settings['influxdb']['fields']['produced']
    if target_col in df_result_target.columns:
         df_result_target.rename(columns={target_col: 'y'}, inplace=True)
    df_hist = df_result_target[['ds', 'y']].copy()

    # Fetch Regressors History
    print(f"  - Fetching Regressors History ({len(regressor_fields)} fields)...")
    regressor_filter = " or ".join([f'r["_field"] == "{f}"' for f in regressor_fields])
    
    query_hist_reg = f'''
    from(bucket: "{settings['influxdb']['buckets']['regressor_history']}")
      |> range(start: {range_start})
      |> filter(fn: (r) => r["_measurement"] == "{settings['influxdb']['measurements']['regressor_history']}")
      |> filter(fn: (r) => {regressor_filter})
      |> map(fn: (r) => ({{ r with _value: r._value * {regressor_scale} }}))
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    df_hist_reg = db.query_dataframe(query_hist_reg)
    
    if not df_hist_reg.empty:
        df_hist_reg['ds'] = pd.to_datetime(df_hist_reg['_time']).dt.tz_localize(None)
        cols_to_keep = ['ds'] + [c for c in df_hist_reg.columns if c in regressor_fields]
        df_hist_reg = df_hist_reg[cols_to_keep]
        df_hist = pd.merge(df_hist, df_hist_reg, on='ds', how='outer')
    
    df_hist = df_hist.sort_values('ds').reset_index(drop=True)
    df_hist = df_hist.fillna(0)
    
    # Fetch lagged regressor data if configured
    lagged_reg_config = settings['model']['neuralprophet'].get('lagged_regressors', {})
    if 'Production_W' in lagged_reg_config:
        print("  - Fetching Production_W for lagged regressor...")
        query_production = f'''
        from(bucket: "{settings['influxdb']['buckets']['live']}")
          |> range(start: {range_start})
          |> filter(fn: (r) => r["_measurement"] == "{settings['influxdb']['measurements']['live']}")
          |> filter(fn: (r) => r["_field"] == "{settings['influxdb']['fields']['live']}")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        df_production = db.query_dataframe(query_production)
        
        if not df_production.empty:
            df_production['ds'] = pd.to_datetime(df_production['_time']).dt.tz_localize(None)
            live_field = settings['influxdb']['fields']['live']
            if live_field in df_production.columns:
                df_production.rename(columns={live_field: 'Production_W'}, inplace=True)
                df_hist = pd.merge(df_hist, df_production[['ds', 'Production_W']], on='ds', how='left')
                df_hist['Production_W'] = df_hist['Production_W'].fillna(df_hist['y'])
        else:
            df_hist['Production_W'] = df_hist['y']

    # B. Fetch Future Regressors
    print("  - Fetching Future Regressors...")
    query_fcst_reg = f'''
    from(bucket: "{settings['influxdb']['buckets']['regressor_future']}")
      |> range(start: -1h, stop: {forecast_days}d) 
      |> filter(fn: (r) => r["_measurement"] == "{settings['influxdb']['measurements']['regressor_future']}")
      |> filter(fn: (r) => {regressor_filter})
      |> map(fn: (r) => ({{ r with _value: r._value * {regressor_scale} }}))
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    df_fcst_reg = db.query_dataframe(query_fcst_reg)
    
    if df_fcst_reg.empty:
        print("Error: No future regressors found.")
        return
        
    df_fcst_reg['ds'] = pd.to_datetime(df_fcst_reg['_time']).dt.tz_localize(None)
    cols_to_keep_fut = ['ds'] + [c for c in df_fcst_reg.columns if c in regressor_fields]
    df_future_regressors = df_fcst_reg[cols_to_keep_fut].sort_values('ds')
    df_future_regressors = df_future_regressors.set_index('ds').resample('15min').interpolate(method='linear').reset_index()
    
    # 3. Construct Prediction Dataframe
    print("\n[3/5] Constructing prediction dataframe...")
    
    df_hist = df_hist.set_index('ds').resample('15min').mean().interpolate().reset_index()
    
    last_hist_date = df_hist['ds'].max()
    df_future_regressors = df_future_regressors[df_future_regressors['ds'] > last_hist_date]
    
    periods = len(df_future_regressors)
    print(f"  Predicting {periods} steps into future...")
    
    try:
        future = model.make_future_dataframe(
            df=df_hist, 
            regressors_df=df_future_regressors,
            periods=periods,
            n_historic_predictions=True
        )
        
        print(f"  Prediction dataframe ready. Shape: {future.shape}")
        
        # 4. Predict
        print("\n[4/5] Generating predictions...")
        try:
            forecast = model.predict(future, decompose=True)
            print("  Prediction with decomposition successful.")
        except Exception as e_decomp:
            print(f"  Decomposition failed ({e_decomp}). Retrying without...")
            forecast = model.predict(future, decompose=False)
        
        # 5. Plotting
        print("\n[5/5] Generating plots...")
        
        output_files = []
        
        # ─────────────────────────────────────────────────────────────────
        # Plot 1: Forecast Overview
        # ─────────────────────────────────────────────────────────────────
        print("  - Plot 1: Forecast Overview...")
        fig1 = model.plot(forecast)
        
        # Rename traces for clarity
        for trace in fig1.data:
            if trace.name == 'yhat1':
                trace.name = 'Forecast'
            if trace.name == 'y':
                trace.name = 'Actual'
                
        fig1.update_layout(title="Forecast Overview (History + Future)")
        fig1_path = os.path.abspath("plot_overview.html")
        fig1.write_html(fig1_path)
        output_files.append(fig1_path)
        
        # ─────────────────────────────────────────────────────────────────
        # Plot 2: Components
        # ─────────────────────────────────────────────────────────────────
        print("  - Plot 2: Components...")
        fig2 = model.plot_components(forecast)
        fig2_path = os.path.abspath("plot_components.html")
        fig2.write_html(fig2_path)
        output_files.append(fig2_path)
        
        # ─────────────────────────────────────────────────────────────────
        # Plot 3: Regressor Impact
        # ─────────────────────────────────────────────────────────────────
        reg_cols = [c for c in forecast.columns if c.startswith('future_regressor_')]
        if reg_cols:
            print("  - Plot 3: Regressor Impact...")
            fig3 = go.Figure()
            for col in reg_cols:
                clean_name = col.replace('future_regressor_', '').replace('_', ' ').title()
                fig3.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast[col],
                    name=clean_name,
                    mode='lines'
                ))
            fig3.update_layout(
                title="Regressor Influence (Impact on Production)",
                yaxis_title="Contribution (W)",
                template="plotly_white"
            )
            fig3_path = os.path.abspath("plot_regressor_impact.html")
            fig3.write_html(fig3_path)
            output_files.append(fig3_path)
        
        # ─────────────────────────────────────────────────────────────────
        # Plot 4: Parameters (AR Weights)
        # ─────────────────────────────────────────────────────────────────
        print("  - Plot 4: Model Parameters...")
        fig4 = model.plot_parameters()
        fig4_path = os.path.abspath("plot_parameters.html")
        fig4.write_html(fig4_path)
        output_files.append(fig4_path)
        
        # ─────────────────────────────────────────────────────────────────
        # Plot 5: AR Weights per Forecast Step (if n_forecasts > 1)
        # ─────────────────────────────────────────────────────────────────
        if n_forecasts > 1 and n_lags > 0:
            print("  - Plot 5: AR Weights per Forecast Step...")
            
            # Show AR weights for different forecast horizons
            steps_to_show = [1, n_forecasts // 4, n_forecasts // 2, n_forecasts]
            steps_to_show = [s for s in steps_to_show if 1 <= s <= n_forecasts]
            
            fig5 = make_subplots(
                rows=len(steps_to_show), cols=1,
                subplot_titles=[f"AR Weights for Step {s} ({s*15}min ahead)" for s in steps_to_show],
                vertical_spacing=0.08
            )
            
            for idx, step in enumerate(steps_to_show):
                model_highlighted = model.highlight_nth_step_ahead_of_each_forecast(step)
                try:
                    ar_fig = model_highlighted.plot_parameters(components=["autoregression"])
                    # Extract trace data
                    for trace in ar_fig.data:
                        trace_copy = go.Scatter(
                            x=trace.x,
                            y=trace.y,
                            name=f"Step {step}",
                            showlegend=(idx == 0)
                        )
                        fig5.add_trace(trace_copy, row=idx+1, col=1)
                except Exception:
                    pass  # Skip if no AR component
            
            fig5.update_layout(
                title="Autoregression Weights by Forecast Horizon",
                height=250 * len(steps_to_show),
                template="plotly_white"
            )
            fig5_path = os.path.abspath("plot_ar_steps.html")
            fig5.write_html(fig5_path)
            output_files.append(fig5_path)
        
        # ─────────────────────────────────────────────────────────────────
        # Plot 6: Residuals Analysis
        # ─────────────────────────────────────────────────────────────────
        if 'y' in forecast.columns and 'yhat1' in forecast.columns:
            print("  - Plot 6: Residuals Analysis...")
            
            # Calculate residuals
            df_res = forecast[['ds', 'y', 'yhat1']].dropna()
            df_res['residuals'] = df_res['y'] - df_res['yhat1']
            
            # Create subplot with residuals over time and histogram
            fig6 = make_subplots(
                rows=2, cols=1,
                subplot_titles=["Residuals Over Time (Actual - Predicted)", "Residual Distribution"],
                vertical_spacing=0.15,
                row_heights=[0.6, 0.4]
            )
            
            # Residuals over time
            fig6.add_trace(
                go.Scatter(
                    x=df_res['ds'],
                    y=df_res['residuals'],
                    mode='lines',
                    name='Residuals',
                    line=dict(color='steelblue')
                ),
                row=1, col=1
            )
            
            # Zero line
            fig6.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
            
            # Histogram
            fig6.add_trace(
                go.Histogram(
                    x=df_res['residuals'],
                    name='Distribution',
                    marker_color='steelblue',
                    nbinsx=50
                ),
                row=2, col=1
            )
            
            # Statistics annotation
            mean_res = df_res['residuals'].mean()
            std_res = df_res['residuals'].std()
            rmse = np.sqrt((df_res['residuals'] ** 2).mean())
            
            fig6.update_layout(
                title=f"Residuals Analysis (RMSE: {rmse:.2f}, Mean: {mean_res:.2f}, Std: {std_res:.2f})",
                template="plotly_white",
                height=600
            )
            
            fig6_path = os.path.abspath("plot_residuals.html")
            fig6.write_html(fig6_path)
            output_files.append(fig6_path)
        
        # ─────────────────────────────────────────────────────────────────
        # Plot 7: Forecast Focus (specific step)
        # ─────────────────────────────────────────────────────────────────
        if n_forecasts > 1:
            print("  - Plot 7: Forecast Focus (1st step)...")
            fig7 = model.plot(forecast, forecast_in_focus=1)
            fig7.update_layout(title="Forecast Focus: 1-Step Ahead (15 min)")
            fig7_path = os.path.abspath("plot_forecast_focus.html")
            fig7.write_html(fig7_path)
            output_files.append(fig7_path)
        
        # ─────────────────────────────────────────────────────────────────
        # Summary
        # ─────────────────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("Done. Files saved:")
        print("=" * 70)
        for f in output_files:
            print(f"  - {os.path.basename(f)}")
        
        # Open in browser
        print("\nOpening plots in browser...")
        for f in output_files:
            webbrowser.open(f"file://{f}")
            time.sleep(0.5)
        
    except Exception as e:
        print(f"\nError during prediction/plotting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    plot_model()
