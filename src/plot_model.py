import os
import pickle
import pandas as pd
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objs as go
from src.config import settings
from src.db import InfluxDBWrapper
import time
import webbrowser

def plot_model():
    print("Starting plotting pipeline...")
    
    # 1. Load Model
    model_path = settings['model']['path']
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # 2. Setup Future Dataframe 
    # We want to see how it fits history + forecast
    future = model.make_future_dataframe(periods=48, freq='30min') # 24 hours of 30min chunks approx

    # 3. Handle Regressors
    if model.extra_regressors:
        print(f"Model uses regressors: {list(model.extra_regressors.keys())}. Fetching data...")
        db = InfluxDBWrapper()
        
        # We need data covering the entire 'future' range (history + future)
        # We fetch a bit more history to be safe
        range_start = f"-{settings['model']['training_days'] + 1}d"
        regressor_scale = settings['model']['preprocessing'].get('regressor_scale', 1.0)
        
        regressor_fields = list(model.extra_regressors.keys())
        regressor_filter = " or ".join([f'r["_field"] == "{f}"' for f in regressor_fields])
        
        # FETCH HISTORY REGRESSORS
        print("Fetching regressor history...")
        query_hist = f'''
        from(bucket: "{settings['influxdb']['buckets']['regressor_history']}")
          |> range(start: {range_start})
          |> filter(fn: (r) => r["_measurement"] == "{settings['influxdb']['measurements']['regressor_history']}")
          |> filter(fn: (r) => {regressor_filter})
          |> map(fn: (r) => ({{ r with _value: r._value * {regressor_scale} }}))
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        df_hist = db.query_dataframe(query_hist)
        
        # FETCH FORECAST REGRESSORS
        print("Fetching regressor forecast...")
        query_fcst = f'''
        from(bucket: "{settings['influxdb']['buckets']['regressor_future']}")
          |> range(start: -1h, stop: 48h) 
          |> filter(fn: (r) => r["_measurement"] == "{settings['influxdb']['measurements']['regressor_future']}")
          |> filter(fn: (r) => {regressor_filter})
          |> map(fn: (r) => ({{ r with _value: r._value * {regressor_scale} }}))
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        df_fcst = db.query_dataframe(query_fcst)
        
        # Combine
        dfs_to_concat = []
        if not df_hist.empty:
            if '_time' in df_hist.columns:
                df_hist['ds'] = pd.to_datetime(df_hist['_time']).dt.tz_localize(None)
            dfs_to_concat.append(df_hist)
        
        if not df_fcst.empty:
            if '_time' in df_fcst.columns:
                df_fcst['ds'] = pd.to_datetime(df_fcst['_time']).dt.tz_localize(None)
            dfs_to_concat.append(df_fcst)
            
        if not dfs_to_concat:
            print("Error: No regressor data found.")
        else:
            df_reg = pd.concat(dfs_to_concat).drop_duplicates(subset=['ds']).sort_values('ds')
            df_reg = df_reg.set_index('ds').sort_index()
            
            # Interpolate onto future
            combined_index = df_reg.index.union(future['ds']).sort_values().drop_duplicates()
            
            for reg_name in regressor_fields:
                if reg_name in df_reg.columns:
                    df_reg[reg_name] = pd.to_numeric(df_reg[reg_name], errors='coerce')
                    s_reg_interp = df_reg[reg_name].reindex(combined_index).interpolate(method='time')
                    future[reg_name] = s_reg_interp.reindex(future['ds']).values
                    future[reg_name] = future[reg_name].ffill().bfill()
                else:
                    future[reg_name] = 0.0

    print("Generating forecast...")
    forecast = model.predict(future)
    
    # 4. Plot
    print("Creating improved interactive plots...")
    
    # Define zoom range: Last 14 days + Future
    last_date = forecast['ds'].max()
    start_zoom = last_date - pd.Timedelta(days=14)
    
    # --- Plot 1: Main Overview (Zoomed) ---
    fig1 = plot_plotly(model, forecast)
    fig1.update_layout(
        title="Forecast Overview (Zoomed: Last 14 Days)",
        xaxis_range=[start_zoom, last_date],
        xaxis_title="Date",
        yaxis_title="Energy Production",
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center")
    )
    fig1_path = os.path.abspath("plot_overview.html")
    fig1.write_html(fig1_path)
    
    # --- Plot 2: Components (Trend, Week, Day) ---
    # Prophet Standard Components
    fig2 = plot_components_plotly(model, forecast)
    fig2.update_layout(title="Model Components (Trend & Seasonality)")
    fig2_path = os.path.abspath("plot_components.html")
    fig2.write_html(fig2_path)
    
    # --- Plot 3: Regressor Impact (If applicable) ---
    fig3_path = None
    if model.extra_regressors:
        reg_names = list(model.extra_regressors.keys())
        
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
        # Note: These values are the *effect* on y, not the raw input
        for reg in reg_names:
            if reg in forecast.columns:
                fig3.add_trace(go.Scatter(
                    x=forecast['ds'], 
                    y=forecast[reg], 
                    name=f'Impact: {reg}',
                    visible='legendonly' # Hidden by default to avoid clutter
                ))
        
        # Also add the raw input scaled? No, keeps it simple.
        
        fig3.update_layout(
            title="Regressor Impact Analysis (Select in Legend)",
            xaxis_range=[start_zoom, last_date],
            yaxis_title="Contribution to Prediction",
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
        webbrowser.open(f"file://{fig2_path}")
    
    print("Done. Files saved:")
    print(f" - {fig1_path}")
    print(f" - {fig2_path}")
    if fig3_path:
        print(f" - {fig3_path}")

if __name__ == "__main__":
    plot_model()
