
import os
import pickle
import pandas as pd
from neuralprophet import NeuralProphet
import plotly.graph_objs as go
from src.config import settings
from src.db import InfluxDBWrapper
import time
import webbrowser

def plot_model():
    print("Starting plotting pipeline... (NeuralProphet)")
    
    # 1. Load Model
    model_path = settings['model']['path']
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # 2. Setup Dataframe for Plotting
    # Unlike Prophet, NeuralProphet model object doesn't cache history.
    # We need to feed it history + future to plot the full context.
    
    db = InfluxDBWrapper()
    
    # Range: History (training_days) + Future (forecast_days)
    training_days = settings['model']['training_days']
    forecast_days = settings['model']['forecast_days']
    
    print(f"Fetching data for plotting (History: {training_days}d + Future: {forecast_days}d)...")
    
    range_start = f"-{training_days + 1}d"
    range_stop = f"{forecast_days + 1}d" # Relative to now
    
    # We reuse the logic from plot_model original: fetch regressors history & future.
    # But we ALSO need 'y' (target) history to show actuals.
    # Let's fetch Regressors first.
    
    regressor_scale = settings['model']['preprocessing'].get('regressor_scale', 1.0)
    # Get regressor names from model configuration or settings?
    # NeuralProphet model doesn't easily expose regressor names in a simple list always, but we can look at specs.
    # Safe bet: use user settings.
    # wait, forecast code relies on 'influxdb.fields.regressor_future'.
    # train code iterates 'regressor_names'.
    # We should iterate all potential regressors. 
    # Let's assume the user config is the source of truth.
    regressor_fields = [settings['influxdb']['fields']['regressor_history']] 
    # If there are sub-regressors (e.g. cloud cover), we need those too.
    # This logic was not fully explicit in original plot_model, it just listed model.extra_regressors.
    # We will try to inspect the model for regressors if possible.
    
    # Inspect model for regressors
    model_regressors = []
    if hasattr(model, 'future_regressors') and model.future_regressors:
        model_regressors = list(model.future_regressors.keys())
    
    print(f"Model uses regressors: {model_regressors}")
    
    regressor_filter = " or ".join([f'r["_field"] == "{f}"' for f in model_regressors]) if model_regressors else "r[\"_field\"] == \"none\""
    
    # Fetch Regressors (History + Future)
    # We can do one query or two. Let's do two to be safe with buckets.
    
    dfs_to_concat = []
    
    if model_regressors:
        # History Regressors
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
            dfs_to_concat.append(df_hist_reg)

        # Future Regressors
        # We need these for the forecast part.
        query_fcst_reg = f'''
        from(bucket: "{settings['influxdb']['buckets']['regressor_future']}")
          |> range(start: -1h, stop: {forecast_days}d) 
          |> filter(fn: (r) => r["_measurement"] == "{settings['influxdb']['measurements']['regressor_future']}")
          |> filter(fn: (r) => {regressor_filter})
          |> map(fn: (r) => ({{ r with _value: r._value * {regressor_scale} }}))
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        df_fcst_reg = db.query_dataframe(query_fcst_reg)
        if not df_fcst_reg.empty:
            df_fcst_reg['ds'] = pd.to_datetime(df_fcst_reg['_time']).dt.tz_localize(None)
            dfs_to_concat.append(df_fcst_reg)
    
    # Combine Regressors
    if dfs_to_concat:
        df_reg = pd.concat(dfs_to_concat).drop_duplicates(subset=['ds']).sort_values('ds')
        df_reg = df_reg.set_index('ds')
    else:
        df_reg = pd.DataFrame()

    # Now Fetch Target History (y)
    print("Fetching target history...")
    produced_scale = settings['model']['preprocessing'].get('produced_scale', 1.0)
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
        print("Warning: No target history found.")
        df_target = pd.DataFrame(columns=['ds', 'y'])
    else:
        df_result_target['ds'] = pd.to_datetime(df_result_target['_time']).dt.tz_localize(None)
        # Rename target field to 'y'
        target_field = settings['influxdb']['fields']['produced']
        if target_field in df_result_target.columns:
             df_result_target.rename(columns={target_field: 'y'}, inplace=True)
        df_target = df_result_target[['ds', 'y']].set_index('ds')

    # Merge Regressors and Target
    # We want a continuous dataframe at 15min resolution usually.
    # Start: min(target, regressor)
    # End: max(target, regressor) + forecast_days
    
    # To simplify, we reindex everything to a global timeline
    if not df_target.empty:
        start_date = df_target.index.min()
    elif not df_reg.empty:
        start_date = df_reg.index.min()
    else:
        start_date = pd.Timestamp.now() - pd.Timedelta(days=1)
        
    end_date = pd.Timestamp.now() + pd.Timedelta(days=forecast_days)
    
    full_index = pd.date_range(start=start_date, end=end_date, freq='30min') # Or 15min? train uses 15min.
    # Let's use 15min to match training frequency.
    
    df_combined = pd.DataFrame(index=full_index)
    df_combined.index.name = 'ds'
    df_combined = df_combined.reset_index()
    
    # Merge Target
    if not df_target.empty:
        df_combined = pd.merge_asof(
            df_combined.sort_values('ds'), 
            df_target.reset_index().sort_values('ds'), 
            on='ds', 
            direction='nearest', 
            tolerance=pd.Timedelta('14min')
        )
    else:
        df_combined['y'] = None

    # Merge Regressors
    if not df_reg.empty:
        # We need to act carefully, maybe interpolation is better.
        df_reg_resampled = df_reg.reindex(full_index).interpolate(method='time')
        df_reg_resampled.index.name = 'ds'
        df_combined = pd.merge(df_combined, df_reg_resampled.reset_index(), on='ds', how='left')
        
    # Fill NAs in Regressors (Linear Interpolation for gaps)
    for reg in model_regressors:
        if reg in df_combined.columns:
            df_combined[reg] = df_combined[reg].interpolate(limit_direction='both')
            df_combined[reg] = df_combined[reg].fillna(0) # Logic from before
        else:
            df_combined[reg] = 0.0

    print("Generating forecast...")
    # Make prediction
    forecast = model.predict(df_combined)
    
    # Rename yhat1 -> yhat if needed
    if 'yhat1' in forecast.columns:
        forecast.rename(columns={'yhat1': 'yhat'}, inplace=True)

    # 4. Plot
    print("Creating improved interactive plots...")
    
    # Define zoom range: Last 30 days + Future
    last_date = forecast['ds'].max()
    start_zoom = last_date - pd.Timedelta(days=30)
    
    # --- Plot 1: Main Overview (Zoomed) ---
    # Use NeuralProphet built-in plot if available?
    # model.plot(forecast) returns a plotly Figure if backend=plotly
    try:
        fig1 = model.plot(forecast, plotting_backend="plotly")
        fig1.update_layout(
            title="Forecast Overview (Zoomed: Last 30 Days)",
            xaxis_range=[start_zoom, last_date],
            xaxis_title="Date",
            yaxis_title="Energy Production",
            legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center")
        )
        fig1_path = os.path.abspath("plot_overview.html")
        fig1.write_html(fig1_path)
    except Exception as e:
        print(f"Standard plot failed: {e}. Falling back to manual...")
        # Fallback manual plot
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['y'], name='Actual', mode='markers'))
        fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Predicted'))
        fig1_path = os.path.abspath("plot_overview.html")
        fig1.write_html(fig1_path)
    
    # --- Plot 2: Components (Trend, Week, Day) ---
    try:
        fig2 = model.plot_components(forecast, plotting_backend="plotly")
        # fig2.update_layout(title="Model Components") # NP might not return a single Figure object easily for components? 
        # Actually plot_components returns a Figure.
        fig2_path = os.path.abspath("plot_components.html")
        fig2.write_html(fig2_path)
    except Exception as e:
        print(f"Component plot failed: {e}")
        fig2_path = "plot_components_failed.html"

    # --- Plot 3: Regressor Impact (If applicable) ---
    fig3_path = None
    if model_regressors:
        reg_names = model_regressors
        
        # Create a clearer impact plot
        fig3 = go.Figure()
        
        # Add Total Prediction as reference
        fig3.add_trace(go.Scatter(
            x=forecast['ds'], 
            y=forecast['yhat'], 
            name='Total Predicted',
            line=dict(color='black', width=2)
        ))
        
        # Add each regressor's contribution
        # In NP, getting exact contribution requires more work or looking at components.
        # But we can plot the Regressor INPUT values efficiently here to see correlation.
        # Or look for component columns in forecast if enabled?
        # Default predict doesn't return components unless decompose=True?
        # Let's skip complex decomposition and just plot input values scaled.
        
        for reg in reg_names:
            if reg in forecast.columns:
                fig3.add_trace(go.Scatter(
                    x=forecast['ds'], 
                    y=forecast[reg], 
                    name=f'Input: {reg}',
                    visible='legendonly' 
                ))
        
        fig3.update_layout(
            title="Regressor Inputs Analysis",
            xaxis_range=[start_zoom, last_date],
            yaxis_title="Value",
            template="plotly_white"
        )
        fig3_path = os.path.abspath("plot_regressors.html")
        fig3.write_html(fig3_path)

    print("Opening plots in browser...")
    webbrowser.open(f"file://{fig1_path}")
    time.sleep(1)
    if fig3_path:
        webbrowser.open(f"file://{fig3_path}")
    else:
        if os.path.exists(fig2_path):
            webbrowser.open(f"file://{fig2_path}")
    
    print("Done. Files saved:")
    print(f" - {fig1_path}")
    print(f" - {fig2_path}")
    if fig3_path:
        print(f" - {fig3_path}")

if __name__ == "__main__":
    plot_model()
