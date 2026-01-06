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
    model_path = settings['model']['model_path']
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

    # 3. Handle Regressors (if any)
    # We need to add regressor columns to 'future' df.
    # We will fetch recent history + future regressor data.
    # For simplicity in this test script, we might just re-fetch everything or try to align.
    
    regressor_name = settings['measurements']['m_regressor_history'] # 'solarcast'
    
    # Check if model has regressors
    if regressor_name in model.extra_regressors:
        print(f"Model uses regressor: {regressor_name}. Fetching data...")
        db = InfluxDBWrapper()
        
        # We need data covering the entire 'future' range (history + 24h)
        # However, getting exact history alignment can be tricky without re-querying everything.
        # Let's simplistic approach: fetch last X days history + forecast
        # Or easier: just fetch the regressor data for the timeframe of 'future' df
        
        start_date = future['ds'].min()
        end_date = future['ds'].max()
        
        # InfluxDB query for this range
        # We need to be careful with timezones. Prophet 'ds' is usually naive or consistent.
        # InfluxDB is UTC.
        
        # To make it robust for this test script, let's fetch a generous range of regressor data
        range_start = f"-{settings['forecast_parameters']['training_days'] + 1}d"
        
        # We need prediction data for the future part
        # And history data for the historical part
        
        # FETCH HISTORY REGRESSOR
        print("Fetching regressor history...")
        regressor_scale = settings['preprocessing'].get('regressor_scale', 1.0)
        query_hist = f'''
        from(bucket: "{settings['buckets']['b_regressor_history']}")
          |> range(start: {range_start})
          |> filter(fn: (r) => r["_measurement"] == "{settings['measurements']['m_regressor_history']}")
          |> filter(fn: (r) => r["_field"] == "{settings['fields']['f_regressor_history']}")
          |> map(fn: (r) => ({{ r with _value: r._value * {regressor_scale} }}))
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        df_hist = db.query_dataframe(query_hist)
        
        # FETCH FORECAST REGRESSOR (for the future 24h)
        print("Fetching regressor forecast...")
        query_fcst = f'''
        from(bucket: "{settings['buckets']['b_regressor_future']}")
          |> range(start: -1h, stop: 48h) 
          |> filter(fn: (r) => r["_measurement"] == "{settings['measurements']['m_regressor_future']}")
          |> filter(fn: (r) => r["_field"] == "{settings['fields']['f_regressor_future']}")
          |> map(fn: (r) => ({{ r with _value: r._value * {regressor_scale} }}))
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        df_fcst = db.query_dataframe(query_fcst)
        
        # Concat and formatting
        # We need one column 'ds' and one '{regressor_name}'
        
        # PRe-process history
        if not df_hist.empty:
            df_hist.rename(columns={settings['fields']['f_regressor_history']: regressor_name}, inplace=True)
            if '_time' in df_hist.columns:
                df_hist['ds'] = pd.to_datetime(df_hist['_time']).dt.tz_localize(None)
        
        # Pre-process forecast
        if not df_fcst.empty:
            df_fcst.rename(columns={settings['fields']['f_regressor_future']: regressor_name}, inplace=True)
            if '_time' in df_fcst.columns:
                df_fcst['ds'] = pd.to_datetime(df_fcst['_time']).dt.tz_localize(None)
                
        # Combine
        df_reg = pd.concat([df_hist, df_fcst]).drop_duplicates(subset=['ds']).sort_values('ds')
        
        # Interpolate regressor to match future timestamps
        # Set index
        df_reg = df_reg.set_index('ds').sort_index()
        
        # Combine indices to allow time-based interpolation
        combined_index = df_reg.index.union(future['ds']).sort_values().drop_duplicates()
        
        # Reindex and interpolate only the regressor column
        # Ensure it is numeric
        df_reg[regressor_name] = pd.to_numeric(df_reg[regressor_name], errors='coerce')
        s_reg_interp = df_reg[regressor_name].reindex(combined_index).interpolate(method='time')
        
        # Extract only the rows corresponding to future['ds']
        future[regressor_name] = s_reg_interp.reindex(future['ds']).values
        
        # Fill any remaining NaNs (e.g. at edges)
        future[regressor_name] = future[regressor_name].ffill().bfill()
        
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
