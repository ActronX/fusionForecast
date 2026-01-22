import os
import pickle
import pandas as pd
from prophet.plot import plot_plotly, plot_components_plotly
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

    # 2. Setup Future Dataframe (similar to forecast.py but maybe just a short lookahead for viz)
    # We want to see how it fits history + forecast
    # Let's do 24 hours future
    future = model.make_future_dataframe(periods=24, freq='h')

    # 3. Handle Regressors
    # We need to add all extra regressor columns to 'future' df.
    if model.extra_regressors:
        print(f"Model uses regressors: {list(model.extra_regressors.keys())}. Fetching data...")
        db = InfluxDBWrapper()
        
        # We need data covering the entire 'future' range (history + 24h)
        range_start = f"-{settings['model']['training_days'] + 1}d"
        regressor_scale = settings['model']['preprocessing'].get('regressor_scale', 1.0)
        
        # Determine fields to fetch
        regressor_fields = list(model.extra_regressors.keys())
        # We need to map these regressor names to InfluxDB fields if they differ
        # In our pipeline, the regressor names in the model ARE the field names
        
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
        
        # FETCH FORECAST REGRESSORS (for the future 24h)
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
        
        # Prepare combined regressor dataframe
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
            return

        df_reg = pd.concat(dfs_to_concat).drop_duplicates(subset=['ds']).sort_values('ds')
        df_reg = df_reg.set_index('ds').sort_index()
        
        # Interpolate each regressor column
        combined_index = df_reg.index.union(future['ds']).sort_values().drop_duplicates()
        
        for reg_name in regressor_fields:
            if reg_name in df_reg.columns:
                df_reg[reg_name] = pd.to_numeric(df_reg[reg_name], errors='coerce')
                s_reg_interp = df_reg[reg_name].reindex(combined_index).interpolate(method='time')
                future[reg_name] = s_reg_interp.reindex(future['ds']).values
                future[reg_name] = future[reg_name].ffill().bfill()
            else:
                print(f"Warning: Regressor '{reg_name}' not found in fetched data. Filling with 0.")
                future[reg_name] = 0.0

    print("Generating forecast...")
    forecast = model.predict(future)
    
    # 4. Plot
    print("Creating interactive plots...")
    
    # Main forecast plot
    fig1 = plot_plotly(model, forecast)
    fig1_path = os.path.abspath("forecast_plot.html")
    fig1.write_html(fig1_path)
    print(f"Saved forecast plot to: {fig1_path}")
    
    # Components plot
    fig2 = plot_components_plotly(model, forecast)
    fig2_path = os.path.abspath("components_plot.html")
    fig2.write_html(fig2_path)
    print(f"Saved components plot to: {fig2_path}")
    
    # Open in browser
    print("Opening plots in browser...")
    webbrowser.open(f"file://{fig1_path}")
    time.sleep(1)
    webbrowser.open(f"file://{fig2_path}")
    
    print("Done.")

if __name__ == "__main__":
    plot_model()
