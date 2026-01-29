## Manual Installation (Alternative)

### Installation

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

The logic of FusionForecast is controlled via the `settings.toml` file. This file follows a hierarchical structure to group related settings logically.

> [!IMPORTANT]
> **Initial Setup:** Rename `settings.example.toml` to `settings.toml` before your first run. The example file contains valid structure but requires your specific credentials and coordinates.

### Detailed Configuration Guide

#### 1. Station Settings `[station]`
Defines the physical characteristics and location of your PV system. This is crucial for accurate solar position and irradiance calculations.
- `latitude` / `longitude`: Decimal coordinates (e.g., Berlin: `52.52`, `13.40`).
- `tilt`: The angle of your panels relative to the horizontal (0° = flat, 90° = vertical).
- `azimuth`: Orientation of the panels. 
    - **Note**: FusionForecast uses Open-Meteo convention: `0` = South, `-90` = East, `90` = West, `180` = North.
- `altitude`: Station altitude in meters.

#### 2. InfluxDB Connection `[influxdb]`
Connection details for your InfluxDB v2 instance.
- `url`: The full HTTP(S) URL of your server.
- `token`: Your API access token with read/write permissions for the relevant buckets.
- `org`: Your organization name.

#### 3. Data Mapping `[influxdb.buckets]` & `[influxdb.measurements]`
Centralized mapping for where data is stored and retrieved.
- **Buckets**:
    - `history_produced`: Where your actual PV meters store past production.
    - `regressor_history` / `regressor_future`: Storage for weather data (training vs. prediction).
    - `target_forecast`: Where prediction results and nowcasts are written.
    - `live`: Real-time power data used for damping factor calculation.
- **Measurements**:
    - Defines the table names within InfluxDB for each category (e.g., `pv`, `weather`, `nowcast`).

#### 4. Field Mapping `[influxdb.fields]`
Individual field names within the measurements. **Every field is cross-referenced in the config comments to its bucket and measurement.**
- `produced`: The field name for actual power/energy (e.g., `generatedWh`).
- `forecast`: The target field for prediction output.
- `regressor_history` / `regressor_future`: The primary irradiance field (GHI).


#### 5. Weather Source `[weather.open_meteo]`
Configures how weather data is fetched from the Open-Meteo API.
- `historic` / `forecast`: Separate endpoints for training history and future predictions.
- `models`: Selection of the weather model (e.g., `best_match`, `icon_d2`).
- `minutely_15`: The specific variable to fetch (default: `diffuse_radiation`, `direct_normal_irradiance`).

#### 6. Model Parameters `[model]`
- `path`: File path for the trained Prophet model (`.pkl`).
- `training_days`: Period of history to use (default: 731 days for 2 full years).
- `forecast_days`: How many days to predict into the future (default: 14).

#### 7. Preprocessing `[model.preprocessing]`
Fine-tuning of data before it entering the ML model.
- `max_power_clip`: Hard limit in Watts to remove outliers or non-physical spikes.
- `produced_scale` / `regressor_scale`: Multipliers to normalize data (e.g., if your meter stores kW but you want Watts).
- `produced_offset` / `regressor_offset`: Crucial for aligning timestamps (e.g., if one source is 1h ahead).

#### 8. NeuralProphet ML Engine `[model.neuralprophet]`
Specific settings for the NeuralProphet model (PyTorch-based deep learning framework).

**Seasonality Settings:**
- `yearly_seasonality`: Enable annual cycles (default: `false` for PV, as daily patterns dominate).
- `weekly_seasonality`: Enable weekly patterns (default: `false`, less relevant for solar).
- `daily_seasonality`: Enable daily patterns (default: `true`, **critical for PV systems**).
- `seasonality_mode`: How seasonal effects combine - `"additive"` or `"multiplicative"` (default: `"additive"`).
- `growth`: Trend component - `"linear"`, `"off"`, or `"discontinuous"` (default: `"off"` for stationary PV data).

**Training Parameters:**
- `learning_rate`: Step size for gradient descent optimizer (default: `0.001`).
- `epochs`: Number of training iterations through the dataset (default: `40`).
- `batch_size`: Number of samples per training batch (default: `128`).

**Regularization (Overfitting Prevention):**
- `trend_reg`: L2 penalty for trend component (default: `0.01`).
- `seasonality_reg`: L2 penalty for seasonality (default: `0.01`).
- `ar_reg`: L2 penalty for AutoRegressive lags (default: `0.0`).
- `future_regressor_regularization`: L2 penalty for weather regressors (default: `0.01`).
- `regressor_mode`: How regressors combine with base forecast - `"additive"` or `"multiplicative"` (default: `"additive"`).

**Hardware Acceleration:**
- `accelerator`: Compute device - `"cpu"`, `"gpu"`, or `"auto"` (default: `"gpu"` for CUDA-enabled systems).

**AutoRegressive (AR) Configuration:**
These parameters enable multi-step forecasting with historical context:
- `n_lags`: Number of historical time steps to use as AR input (default: `8` = 2 hours at 15-min intervals).
  - **How it works**: The AR-Net looks back at the last `n_lags` actual production values to inform predictions.
  - **Example**: With `n_lags=8`, the model uses the past 2 hours of production history for intraday patterns.
- `n_forecasts`: Number of future steps to predict simultaneously (default: `96` = 24 hours).
  - **Multi-step prediction**: Instead of predicting one step at a time, the model generates all 96 future values in one forward pass.
- `ar_layers`: Hidden layer configuration for AR network (default: `[32, 16]` = two hidden layers for deep AR-Net).
  - **Non-empty list** enables deeper learning of complex autoregressive relationships.
- **Lagged Regressors**: Additional configuration under `[model.neuralprophet.lagged_regressors]` allows using other variables (like actual production) as lagged inputs alongside AR-Net.



#### 9. Hyperparameter Tuning `[model.tuning]`
- `trials`: Number of optimization trials (default: 100).
- `process_count`: Parallel CPU cores for Optuna optimization.
- `night_threshold`: Power level (Watts) below which data is ignored during evaluation to prevent "easy night wins" from skewing metrics.

#### 9. Lagged Regressors `[model.neuralprophet.lagged_regressors]`
Lagged regressors enable real-time intraday corrections by incorporating recent actual production data into future predictions.
- `Production_W`: Number of lags (time steps) of actual production to use (default: `8` = 2 hours at 15-min intervals).
  - **How it works**: The model uses the last 2 hours of actual production as an additional input feature alongside weather data.
  - **Effect**: Enables dynamic forecast corrections for immediate weather changes (e.g., fog, clouds) without a separate nowcast script.
  - **Combined with AR-Net**: Works alongside `n_lags` for comprehensive intraday prediction.

#### 10. Hyperparameter Tuning `[model.tuning]`

## Usage Workflow

### 1. Verify Connection

Before proceeding, you can verify the connection to your InfluxDB instance.

**Start:**
- Windows: `test_connection.bat`
- Linux: `./test_connection.sh`

### 2. Fetch Historical Weather Data

Before training the model, you must fetch historical weather data for your location to train the regressor.

**Start:**
- Windows: `fetch_historic_weather.bat` or `python -m src.fetch_historic_weather`
- Linux: `./fetch_historic_weather.sh` or `python3 -m src.fetch_historic_weather`

### 3. Train Model

The training script loads historical data, trains the NeuralProphet model, and saves it locally as a `.pkl` file.

**Start:**
- Windows: `train.bat` or `python -m src.train`
- Linux: `./train.sh` or `python3 -m src.train`

### 4. Fetch Future Weather Data

Before creating a forecast, you must fetch the latest weather forecast from DWD/Open-Meteo. This data serves as the regressor for the prediction.

**Start:**
- Windows: `fetch_future_weather.bat` or `python -m src.fetch_future_weather`
- Linux: `./fetch_future_weather.sh` or `python3 -m src.fetch_future_weather`

### 5. Create Forecast

The forecast script loads the saved model and future regressor data (e.g., weather forecast), calculates the forecast, and writes it to InfluxDB.

**Start:**
- Windows: `forecast.bat` or `python -m src.forecast`
- Linux: `./forecast.sh` or `python3 -m src.forecast`

### 6. Intraday Monitoring (Optional)

The forecast script automatically performs intraday corrections using the **lagged regressor** approach:
- **AR-Net** (`n_lags=8`): Uses last 2 hours of target values for autoregressive predictions.
- **Lagged Regressor** (`Production_W`): Uses last 2 hours of actual production to dynamically adjust forecasts.

No separate nowcast script needed - corrections happen automatically during forecast generation when live production data is available.

### 7. Run Pipeline (Full Automation)

Executes the complete workflow in order:
1. **Connection Test**: Checks InfluxDB health.
2. **Fetch Historic Weather**: Updates historic data.
3. **Fetch Future Weather**: Updates forecast data.
4. **Train Model**: Retrains the model.
5. **Create Forecast**: Generates the forecast with intraday correction.

**Start:**
- Windows: `run_pipeline.bat`
- Linux: `./run_pipeline.sh`

### 8. Hyperparameter Tuning

To optimize the model's accuracy, you can tune the hyperparameters (e.g., `changepoint_prior_scale`, `seasonality_mode`) using **Optuna** (Bayesian Optimization). The settings (trials, cpu cores) are defined in `settings.toml` under `[model.tuning]`.

**Start:**
- Windows: `tune.bat` or `python -m src.tune`
- Linux: `./tune.sh` or `python3 -m src.tune`

#### Evaluation Metrics

The tuning script evaluates model performance using Cross-Validation and calculates the following metrics. **Note**: To ensure relevance for PV systems, values below the `night_threshold` (defined in settings, e.g., 50 W) are **excluded** from these calculations. This prevents the metrics from being artificially improved by easy "0 Watt" predictions during the night.

*   **RMSE (Root Mean Squared Error)**:
    *   **What it is**: The square root of the average squared differences between incorrect forecasts and actual values.
    *   **PV Context**: RMSE penalizes **large errors** more heavily than small ones. This is critical for battery management: a single large prediction error (e.g., predicting 5kW when it's 2kW) can disrupt your charging strategy more than many small errors. A lower RMSE means fewer "big surprises".

*   **MAE (Mean Absolute Error)**:
    *   **What it is**: The average absolute difference between the forecast and the actual value.
    *   **PV Context**: This tells you, on average, how many Watts your forecast is off. It gives a linear representation of error and is easy to interpret. If MAE is 500W, your forecast is on average 500W deviation from reality.

*   **MAPE (Mean Absolute Percentage Error)**:
    *   **What it is**: The average error expressed as a percentage of the actual value.
    *   **PV Context**: Useful for understanding relative performance. A 10% MAPE means the model is usually within 10% of the actual generation. However, be cautious: MAPE can be erratic when production is very low (near the threshold), which is why filtering night values is crucial.

## Folder Structure

- `src/`: Source code (Python scripts).
- `models/`: Storage location for trained models (`prophet_model.pkl`).
- `node_red/`: Node-RED flows and documentation for consumer control.
- `settings.toml`: Configuration file.
- `requirements.txt`: Python dependencies.


## Script Overview

Here is a detailed description of the Python scripts located in `src/`:

### Core Pipeline
- **`src/train.py`**: Trains the **Production** (PV) model. Fetches historical production and regressor data, configures AR-Net (`n_lags=8`) and lagged regressors (`Production_W`), trains NeuralProphet, and saves `prophet_model.pkl`.
- **`src/forecast.py`**: Generates **Production** forecasts using multi-step prediction with **intraday correction**. Uses AR-Net for autoregressive patterns and lagged regressors for real-time corrections based on actual production. Predicts generation in 24-hour chunks and writes to InfluxDB.

### Data Fetching & Calculations
- **`src/fetch_future_weather.py`**: Fetches **current** weather forecasts from Open-Meteo. Uses `weather_utils.py` to calculate effective irradiance (GTI) and clearsky GHI.
- **`src/fetch_historic_weather.py`**: Fetches **historical** weather data from Open-Meteo. Uses `weather_utils.py` to calculate effective irradiance (GTI) and clearsky GHI.
- **`src/weather_utils.py`**: **Consolidated Physics Model**. Shared utility that handles `pvlib` calculations for solar position, plane of array (POA) irradiance, and clearsky GHI.

### Utilities & Maintenance

- **`src/tune.py`**: Performs **Hyperparameter Tuning** using **Optuna** to find the optimal NeuralProphet parameters (e.g., `learning_rate`, `epochs`) for your specific data.

- **`src/plot_model.py`**: Generates interactive Plotly charts of the model components (trend, seasonality) for visual inspection.


## InfluxDB Data Flow

This section describes which data is read from and written to InfluxDB, and why this is necessary.

### 1. Training (Model Creation)
*Scripts: `src/train.py`*

- **Reads**:
    - **Production History** (`buckets.history_produced`): Actual historical PV generation data.
    - **Regressor History** (`buckets.regressor_history`): Historical weather data (e.g., solar irradiance) corresponding to the production history.
- **Why**: The NeuralProphet model needs to learn the relationship between the target variable (Production) and time/weather. The model uses:
  - **AR-Net** (`n_lags=8`): Uses last 2 hours of target values to capture short-term autoregressive patterns.
  - **Lagged Regressor** (`Production_W`): Uses last 2 hours of actual production to enable real-time intraday corrections.

### 2. Forecasting (Prediction)
*Scripts: `src/forecast.py`*

- **Reads**:
    - **Future Regressor** (`buckets.regressor_future`): The current weather forecast for the next few days.
    - **Historical Context** (`buckets.live` or `buckets.history_produced`): Recent production data (last 24 hours) when AR mode is enabled.
- **Writes**:
    - **Target Forecast** (`buckets.target_forecast`): The predicted values for production.
- **Why**: To make a prediction for tomorrow, the model needs to know the expected weather (Regressor). The result is then stored so it can be visualized in Grafana or used by an energy management system (e.g., to charge a battery).

### 3. Data Fetching (External Sources)
*Scripts: `src/fetch_future_weather.py`, `src/fetch_historic_weather.py`*

- **Writes**:
    - **Historic Weather** (`buckets.regressor_history`): Stores historical weather data fetched from Open-Meteo.
    - **Future Weather** (`buckets.regressor_future`): Stores the latest weather forecast from Open-Meteo.
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
# Fetch Future Weather Data (e.g. every 15 minutes at minute 1, 16, 31, 46)
1,16,31,46 * * * * /path/to/fusionForecast/fetch_future_weather.sh >> /path/to/fusionForecast/logs/fetch_future_weather.log 2>&1

# Create a forecast every 15 minutes (e.g. at minute 2, 17, 32, 47)
2,17,32,47 * * * * /path/to/fusionForecast/forecast.sh >> /path/to/fusionForecast/logs/forecast.log 2>&1

# Fetch historic weather data once a month (e.g., 1st of the month at 01:00 AM)
0 1 1 * * /path/to/fusionForecast/fetch_historic_weather.sh >> /path/to/fusionForecast/logs/fetch_historic.log 2>&1

# Train the model once a month (e.g., 1st of the month at 02:00 AM)
0 2 1 * * /path/to/fusionForecast/train.sh >> /path/to/fusionForecast/logs/train.log 2>&1
```

Ensure the scripts are executable (`chmod +x *.sh`) and the path is absolute.
Logs will be stored in the `logs/` directory (created automatically by `install_deps.sh`).
