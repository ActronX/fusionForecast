# FusionForecast

FusionForecast is an ML-based tool for forecasting time series data (e.g., PV generation) using [**Prophet**](https://facebook.github.io/prophet/) and [**InfluxDB**](https://www.influxdata.com/). It trains a model based on historical data and external regressors (e.g., weather forecasts or Solcast) and writes the forecasts back into an InfluxDB.

## Features

- **Data Source**: Reads training data (target value and regressor) from InfluxDB.
- **Modeling**: Uses Facebook Prophet for time series forecasting.
- **Server-Side Aggregation**: Performs downsampling (e.g., to 1h means) directly in the database.
- **Configurable**: All settings (buckets, measurements, offsets) are defined in `settings.toml`.
- **Offset Support**: Supports time offsets for regressors (e.g., to adjust time zones or lead times).

## Prerequisites

- Python 3.9+
- InfluxDB v2
- Access to relevant InfluxDB buckets
- **Historical PV Data:** At least 30 days of historical generation data are required for training. Ideally, 1 to 2 years of data should be available for better accuracy.

## Installation

1. Clone or extract the repository.
2. Create virtual environment and install dependencies:

   **Windows:**
   ```batch
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

   **Linux/Mac:**
   ```bash
   chmod +x install_deps.sh
   ./install_deps.sh
   ```
   This script installs system dependencies, creates the venv, and necessary folders via `install_deps.sh`.

## Configuration

Configuration is managed via the `settings.toml` file.

> **Initial Setup:** Before starting, rename `settings.example.toml` to `settings.toml` and fill in your credentials.

**This file is fully documented with comments for every setting.** Please refer to it for detailed explanations of all parameters.

### Important Settings Sections

- **[influxdb]**: URL, Token, and Org for database connection.
- **[buckets]**: Names of source and target buckets.
- **[measurements] / [fields]**: Names of Measurements and Fields in InfluxDB.
- **[preprocessing]**:
    - `max_power_clip`: Upper limit for outlier clipping.
    - `regressor_offset`: Time offset for the regressor (e.g., `"-1h"`), applied via `timeShift()`.
- **[forecast_parameters]**:
    - `training_days`: Past period of training data (e.g., 731 days).
    - `forecast_days`: Forecast horizon into the future (e.g., 3 days).
- **[prophet]**:
    - `regressor_prior_scale`: Prior scale for the regressor (controls flexibility).
    - `seasonality_mode` / `regressor_mode`: Additive or Multiplicative.
- **[prophet.consumption]**:
    - Specific settings for the consumption model (e.g., `daily_seasonality`, `weekly_seasonality`).
- **[open_meteo]** / **[open_meteo.historic]** / **[open_meteo.forecast]**:
    - Settings for fetching weather data (location, API URLs, models).
    - `minutely_15`: Specifies the 15-minute weather variable to fetch (e.g., `global_tilted_irradiance_instant`).
- **[open_meteo]**:
    - `latitude` / `longitude`: GPS coordinates of your PV system.
    - `azimuth`: Orientation of the PV panels (0 = South, -90 = East, 90 = West).
    - `tilt`: Tilt angle of the PV panels.

## Usage Workflow

### 1. Verify Connection

Before proceeding, you can verify the connection to your InfluxDB instance.

**Start:**
- Windows: `test_connection.bat`
- Linux: `./test_connection.sh`

### 2. Fetch Historical Weather Data

Before training the model, you must fetch historical weather data for your location to train the regressor.

**Start:**
- Windows: `fetch_historic_dwd_data.bat` or `python -m src.fetch_historic_dwd_data`
- Linux: `./fetch_historic_dwd_data.sh` or `python3 -m src.fetch_historic_dwd_data`

### 3. Train Model

The training script loads historical data, trains the Prophet model, and saves it locally as a `.pkl` file.

**Start:**
- Windows: `train.bat` or `python -m src.train`
- Linux: `./train.sh` or `python3 -m src.train`

### 4. Fetch Future Weather Data

Before creating a forecast, you must fetch the latest weather forecast from DWD/Open-Meteo. This data serves as the regressor for the prediction.

**Start:**
- Windows: `update_dwd_data.bat` or `python -m src.update_dwd_data`
- Linux: `./update_dwd_data.sh` or `python3 -m src.update_dwd_data`

### 5. Create Forecast

The forecast script loads the saved model and future regressor data (e.g., weather forecast), calculates the forecast, and writes it to InfluxDB.

**Start:**
- Windows: `forecast.bat` or `python -m src.forecast`
- Linux: `./forecast.sh` or `python3 -m src.forecast`

### 6. Run Pipeline (Train + Forecast)

Executes both sequentially.

**Start:**
- Windows: `run_pipeline.bat`
- Linux: `./run_pipeline.sh`

### 7. Consumption Forecasting (optional)

A separate pipeline exists for forecasting electricity consumption (without external regressors, purely based on history and seasonality).

**Features:**
- Predicts consumption based on historical patterns (daily, weekly, yearly seasonality).
- **Holidays**: Automatically integrates holidays for **Germany/Bavaria (DE-BY)**:
    - **Public Holidays**: Using `holidays` library.
    - **School Holidays**: Fetched dynamically from `ferien-api.de`.
- Configurable via `[prophet.consumption]` in `settings.toml`.

**Start Training:**
- Windows: `train_consumption.bat`
- Linux: `./train_consumption.sh`

- Windows: `forecast_consumption.bat`
- Linux: `./forecast_consumption.sh`

### 8. Hyperparameter Tuning

To optimize the model's accuracy, you can tune the hyperparameters (e.g., `changepoint_prior_scale`, `seasonality_mode`) using a grid search. The parameters to be tested are defined in `settings.toml` under `[prophet.tuning]`.

**Start:**
- Windows: `tune.bat` or `python -m src.tune`
- Linux: `./tune.sh` or `python3 -m src.tune`

## Folder Structure

- `src/`: Source code (Python scripts).
- `models/`: Storage location for trained models (`prophet_model.pkl`).
- `settings.toml`: Configuration file.
- `requirements.txt`: Python dependencies.


## Script Overview

Here is a detailed description of the Python scripts located in `src/`:

### Core Pipeline
- **`src/train.py`**: Trains the **Production** (PV) model. Fetches historical production and regressor data, trains Prophet, and saves `prophet_model.pkl`.
- **`src/forecast.py`**: Generates **Production** forecasts. Loads the model, fetches future weather data, predicts generation, and writes to InfluxDB.
- **`src/train_consumption.py`**: Trains the **Consumption** model. Uses historical consumption data and holidays (Public & School) to train `prophet_model_consumption.pkl`.
- **`src/forecast_consumption.py`**: Generates **Consumption** forecasts. Similar to `forecast.py` but for household usage.

### Data Fetching
- **`src/update_dwd_data.py`**: Fetches **current** weather forecasts from Open-Meteo (DWD ICON-D2) for the next few days and stores them in InfluxDB (used for forecasting).
- **`src/fetch_historic_dwd_data.py`**: Fetches **historical** weather data from Open-Meteo for the past (used for training the regressor).

### Utilities & Maintenance
- **`src/evaluate.py`**: Calculates error metrics (RMSE, MAE, MAPE) by comparing past forecasts with actual values. helpful for checking model performance.
- **`src/tune.py`**: Performs **Hyperparameter Tuning** using a grid search to find the optimal Prophet parameters (e.g., `changepoint_prior_scale`) for your specific data.
- **`src/diagnose.py`**: Diagnostic tool to check data integrity, timestamps, and potential issues in InfluxDB data before training.
- **`src/backfill.py`**: Generates "historical forecasts" (hindcasts) using the current model to evaluate how it *would have* performed in the past.
- **`src/plot_model.py`**: Generates interactive Plotly charts of the model components (trend, seasonality) for visual inspection.


## InfluxDB Data Flow

This section describes which data is read from and written to InfluxDB, and why this is necessary.

### 1. Training (Model Creation)
*Scripts: `src/train.py`, `src/train_consumption.py`*

- **Reads**:
    - **Production History** (`b_history_produced`): Actual historical PV generation data.
    - **Consumption History** (`b_history_consumption`): Actual historical household electricity usage.
    - **Regressor History** (`b_regressor_history`): Historical weather data (e.g., solar irradiance) corresponding to the production history.
- **Why**: The Prophet model needs to learn the relationship between the target variable (Production/Consumption) and time/weather. For example, it learns that "high irradiance = high production" or "Monday morning = high consumption".

### 2. Forecasting (Prediction)
*Scripts: `src/forecast.py`, `src/forecast_consumption.py`*

- **Reads**:
    - **Future Regressor** (`b_regressor_future`): The current weather forecast for the next few days.
- **Writes**:
    - **Target Forecast** (`b_target_forecast`, `b_target_consumption`): The predicted values for production and consumption.
- **Why**: To make a prediction for tomorrow, the model needs to know the expected weather (Regressor). The result is then stored so it can be visualized in Grafana or used by an energy management system (e.g., to charge a battery).

### 3. Data Fetching (External Sources)
*Scripts: `src/update_dwd_data.py`, `src/fetch_historic_dwd_data.py`*

- **Writes**:
    - **Historic Weather** (`b_dwd_historic`): Stores historical weather data fetched from Open-Meteo.
    - **Future Weather** (`b_regressor_future`): Stores the latest weather forecast from Open-Meteo.
- **Why**: FusionForecast relies on external weather data (Irradiance) to make accurate PV predictions. This data must be actively fetched and stored in InfluxDB so the Training and Forecast scripts can access it.

## Automation (Linux Cron Jobs)

To run the system automatically, you can set up Cron jobs.
This ensures the model is regularly retrained (to learn current trends) and forecasts are continuously updated.

Open the Crontab configuration:
```bash
crontab -e
```

Add the following lines (adjust `/path/to/fusionForecast` to your installation path):

```cron
# Update DWD Weather Data (e.g. every hour at minute 15)
15 * * * * /path/to/fusionForecast/update_dwd_data.sh >> /path/to/fusionForecast/logs/update_dwd.log 2>&1

# Train the model every day at 03:00 AM
0 3 * * * /path/to/fusionForecast/train.sh >> /path/to/fusionForecast/logs/train.log 2>&1

# Train consumption model every day at 03:30 AM
30 3 * * * /path/to/fusionForecast/train_consumption.sh >> /path/to/fusionForecast/logs/train_consumption.log 2>&1

# Create a forecast every hour
0 * * * * /path/to/fusionForecast/forecast.sh >> /path/to/fusionForecast/logs/forecast.log 2>&1
0 * * * * /path/to/fusionForecast/forecast_consumption.sh >> /path/to/fusionForecast/logs/forecast_consumption.log 2>&1
```

Ensure the scripts are executable (`chmod +x *.sh`) and the path is absolute.
Logs will be stored in the `logs/` directory (created automatically by `install_deps.sh`).

## Evaluation Metrics

To assess the quality of the forecasts, the following metrics are used. They help to understand how much the model deviates from the actual values.

> **Note:** All metrics are calculated only for time periods where the actual production (`y_true`) exceeds a configurable threshold (default `50` units). This ensures that night hours or times of insignificant production do not distort the error metrics (e.g., by artificially lowering RMSE/MAE due to many zero-error data points at night).

### RMSE (Root Mean Squared Error)
- **Definition:** The square root of the average of squared errors.
- **Relevance for PV:** RMSE penalizes larger errors more heavily than smaller ones. In the context of PV forecasting, this is particularly relevant if large deviations (e.g., predicting full sun during a thunderstorm) are significantly more "expensive" or problematic for grid/battery management than many small deviations.

### MAE (Mean Absolute Error)
- **Definition:** The average of absolute errors.
- **Relevance for PV:** This gives a direct average of "how far off" the forecast is in the same unit as the data (e.g., Watts). It is easy to interpret: "On average, the forecast is off by X Watts."

### MAPE (Mean Absolute Percentage Error)
- **Definition:** The average percentage error.
- **Relevance for PV:** Indicates accuracy in percent.
    - **Challenge:** For PV systems, production is often 0 (at night) or very low. Division by zero (or near zero) leads to infinite or extremely high MAPE values that distort the overall picture.
    - **Solution:** Therefore, MAPE is usually only calculated for daylight hours or implied with a small offset, but remains difficult to interpret for night times.
