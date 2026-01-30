
import os
import torch 
import pandas as pd
from src.config import settings
from src.db import InfluxDBWrapper
import plotly.graph_objs as go
import webbrowser
import time

def plot_model():
    print("Starting robust plotting pipeline... (NeuralProphet)")
    
    # 1. Load Model
    model_path = settings['model']['path']
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    model = torch.load(model_path, weights_only=False)
    if hasattr(model, 'restore_trainer'):
        model.restore_trainer()

    # 2. Configuration & Data Fetching
    db = InfluxDBWrapper()
    
    training_days = settings['model']['training_days'] 
    
    # User Request: 14 days history, 7 days future
    plot_history_days = 14 
    forecast_days = 7 # Override setting for plotting view
    
    # ... (rest of configuration)
    
    # ... (fetching logic uses plot_history_days and forecast_days, so skipping edit there if variable names match)
    
    # Note: I need to ensure the variables `plot_history_days` and `forecast_days` are used in fetching queries.
    # In my previous write_to_file, I used `forecast_days = settings...`. I will overwrite lines 28-30

    
    reg_config = settings['influxdb']['fields']['regressor_history']
    regressor_fields = reg_config if isinstance(reg_config, list) else [reg_config]
    
    regressor_scale = settings['model']['preprocessing'].get('regressor_scale', 1.0)
    produced_scale = settings['model']['preprocessing'].get('produced_scale', 1.0)
    
    print(f"Fetching data (History: {plot_history_days}d, Future: {forecast_days}d)...")

    # A. Fetch History (Target + Regressors)
    # We fetch enough history for the model to have context (n_lags) + the plot window.
    # Safe buffer: plot_history_days + 5 days
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
        # Drop non-numeric and _time
        cols_to_keep = ['ds'] + [c for c in df_hist_reg.columns if c in regressor_fields]
        df_hist_reg = df_hist_reg[cols_to_keep]
        
        # Merge into df_hist
        df_hist = pd.merge(df_hist, df_hist_reg, on='ds', how='outer') # Outer to keep all timestamps
    
    # Sort and fill
    df_hist = df_hist.sort_values('ds').reset_index(drop=True)
    df_hist = df_hist.fillna(0) # Simple fill for plotting gaps
    
    # Fetch lagged regressor data (Production_W) if configured
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
                df_hist['Production_W'] = df_hist['Production_W'].fillna(df_hist['y'])  # Use 'y' as fallback
                print(f"    Added Production_W column (lagged regressor)")
        else:
            # If no live data, use 'y' as Production_W
            df_hist['Production_W'] = df_hist['y']
            print("    No live data found, using 'y' as Production_W")

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
    
    # Interpolate Future Regressors to ensure no gaps for make_future_dataframe
    df_future_regressors = df_future_regressors.set_index('ds').resample('15min').interpolate(method='linear').reset_index()
    
    
    # 3. Construct Prediction Dataframe
    print("Constructing prediction dataframe...")
    
    # clean df_hist to have 15 min freq
    df_hist = df_hist.set_index('ds').resample('15min').mean().interpolate().reset_index()
    
    # Create the future dataframe using NeuralProphet's method
    # This automatically handles extending ‘ds’ and appending regressors if provided correctly
    # But for 'regressors_df', NP expects a dataframe with 'ds' and regressor columns covering the FUTURE period.
    
    # Ensure df_future_regressors covers the period after df_hist
    last_hist_date = df_hist['ds'].max()
    df_future_regressors = df_future_regressors[df_future_regressors['ds'] > last_hist_date]
    
    # We allow 'periods' to be controlled by the available future regressor data
    # (or strictly forecast_days * 96)
    periods = len(df_future_regressors)
    print(f"Predicting {periods} steps into future...")
    
    try:
        future = model.make_future_dataframe(
            df=df_hist, 
            regressors_df=df_future_regressors,
            periods=periods,
            n_historic_predictions=True # We want to plot history too
        )
        
        print(f"Prediction dataframe ready. Shape: {future.shape}")
        
        # 4. Predict
        # Try with decomposition first to get full components
        try:
            print("Predicting with decomposition...")
            forecast = model.predict(future, decompose=True)
            print("Prediction with decomposition successful.")
        except Exception as e_decomp:
            print(f"Prediction with decomposition failed ({e_decomp}). Retrying without decomposition...")
            forecast = model.predict(future, decompose=False)
        
        # Do NOT rename yhat1 to yhat, as model.plot expects yhat1
        # if 'yhat1' in forecast.columns:
        #    forecast.rename(columns={'yhat1': 'yhat'}, inplace=True)
            
        print("Forecast generated successfully.")
        
        # 5. Plotting
        print("Generating plots...")
        
        # Plot 1: Overview
        fig1 = model.plot(forecast, plotting_backend="plotly")
        
        # Rename traces for clarity (yhat1 -> Forecast, y -> Actual)
        for trace in fig1.data:
            if trace.name == 'yhat1':
                trace.name = 'Forecast'
            if trace.name == 'y':
                trace.name = 'Actual'
                
        fig1.update_layout(title="Forecast Overview (Last 14 Days + Future)")
        fig1_path = os.path.abspath("plot_overview.html")
        fig1.write_html(fig1_path)
        
        # Plot 2: Components (Standard)
        fig2 = model.plot_components(forecast, plotting_backend="plotly")
        fig2_path = os.path.abspath("plot_components.html")
        fig2.write_html(fig2_path)
        
        # Plot 3: Custom Regressor Impact Analysis
        # Extract columns starting with 'future_regressor_'
        reg_cols = [c for c in forecast.columns if c.startswith('future_regressor_')]
        if reg_cols:
            fig3 = go.Figure()
            for col in reg_cols:
                # Clean name: future_regressor_temperature_2m -> Temperature 2m
                clean_name = col.replace('future_regressor_', '').replace('_', ' ').title()
                fig3.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast[col],
                    name=clean_name,
                    mode='lines'
                ))
            fig3.update_layout(
                title="Regressor Influence (Impact on Production)",
                yaxis_title="Contribution (kW)", # Assuming kW
                template="plotly_white"
            )
            fig3_path = os.path.abspath("plot_regressor_impact.html")
            fig3.write_html(fig3_path)
        else:
            fig3_path = None
            
        # Plot 4: Parameters (Weights)
        fig4 = model.plot_parameters(plotting_backend="plotly")
        fig4_path = os.path.abspath("plot_parameters.html")
        fig4.write_html(fig4_path)
        
        print("Done. Files saved in project root:")
        print(f" - {fig1_path}")
        print(f" - {fig2_path}")
        if fig3_path: print(f" - {fig3_path}")
        print(f" - {fig4_path}")
        
        # Open in browser
        # Open in browser
        webbrowser.open(f"file://{fig1_path}")
        time.sleep(1)
        webbrowser.open(f"file://{fig2_path}")
        time.sleep(1)
        if fig3_path: webbrowser.open(f"file://{fig3_path}")
        time.sleep(1)
        webbrowser.open(f"file://{fig4_path}")

        
    except Exception as e:
        print(f"Error during prediction/plotting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    plot_model()
