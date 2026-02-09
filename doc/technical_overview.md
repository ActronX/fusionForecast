# Technical Overview

This document provides a detailed description of the Python scripts and the data flow within InfluxDB.

## Script Overview

Here is a detailed description of the Python scripts located in `src/`:

### Core Pipeline
- **`src/train.py`**: Trains the **Production** (PV) model. Fetches historical production and regressor data, configures the model with **Linear AR** (`n_lags=96`, 24h context), trains NeuralProphet, and saves `prophet_model.pkl`.
- **`src/forecast.py`**: Generates **Production** forecasts using multi-step prediction with **AR correction**. Uses:
  - **AR-Net**: Autoregressive network initializes with the last 24h of live production.
  - **Recursive Forecasting**: Predicts 24 hours in chunks, feeding predictions back as history for subsequent steps.
  - Loads the model, fetches future weather data and recent production (via `data_loader.py`), predicts generation, and writes to InfluxDB.

### Data Fetching & Calculations
- **`src/fetch_future_weather.py`**: Fetches **current** weather forecasts from Open-Meteo. Uses `weather_utils.py` to calculate effective irradiance (GTI) and clearsky GHI.
- **`src/fetch_historic_weather.py`**: Fetches **historical** weather data from Open-Meteo. Uses `weather_utils.py` to calculate effective irradiance (GTI) and clearsky GHI.
- **`src/weather_utils.py`**: **Consolidated Physics Model**. Shared utility that handles `pvlib` calculations for solar position, plane of array (POA) irradiance, and clearsky GHI.

### Utilities & Maintenance

- **`src/tune.py`**: Performs **Hyperparameter Tuning** using **Grid Search** to find the optimal NeuralProphet parameters (e.g., `ar_layers`, `ar_reg`, `seasonality_mode`) for your specific data.

- **`src/plot_model.py`**: Generates interactive Plotly charts of the model components (trend, seasonality) for visual inspection.

## InfluxDB Data Flow

This section describes which data is read from and written to InfluxDB, and why this is necessary.

### 1. Training (Model Creation)
*Scripts: `src/train.py`*

- **Reads**:
    - **Production History** (`buckets.history_produced`): Actual historical PV generation data.
    - **Regressor History** (`buckets.regressor_history`): Historical weather data (e.g., solar irradiance) corresponding to the production history.
- **Why**: The NeuralProphet model needs to learn the relationship between the target variable (Production) and time/weather. The model uses:
  - **AutoRegressive (AR-Net)** with `n_lags=96`: Learns from the last 24 hours of target values to capture short-term patterns and initialize predictions.

### 2. Forecasting (Prediction)
*Scripts: `src/forecast.py`*

- **Reads**:
    - **Future Regressor** (`buckets.regressor_future`): The current weather forecast for the next few days.
- **Writes**:
    - **Target Forecast** (`buckets.target_forecast`): The predicted values for production.
- **Why**: To make a prediction for tomorrow, the model needs to know the expected weather (Regressor). The result is then stored so it can be visualized in Grafana or used by an energy management system (e.g., to charge a battery).

### 3. Data Fetching (External Sources)
*Scripts: `src/fetch_future_weather.py`, `src/fetch_historic_weather.py`*

- **Writes**:
    - **Historic Weather** (`buckets.regressor_history`): Stores historical weather data fetched from Open-Meteo.
    - **Future Weather** (`buckets.regressor_future`): Stores the latest weather forecast from Open-Meteo.
- **Why**: FusionForecast relies on external weather data (Irradiance) to make accurate PV predictions. This data must be actively fetched and stored in InfluxDB so the Training and Forecast scripts can access it.
