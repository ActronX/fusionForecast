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
- `n_lags`: Number of historical time steps to use as input (default: `96` = 24 hours at 15-min intervals).
  - **How it works**: The model looks back at the last `n_lags` actual production values to inform predictions.
  - **Example**: With `n_lags=96`, the model uses the past 24 hours of production history.
- `n_forecasts`: Number of future steps to predict simultaneously (default: `96` = 24 hours).
  - **Multi-step prediction**: Instead of predicting one step at a time, the model generates all 96 future values in one forward pass.
- `ar_layers`: Hidden layer configuration for AR network (default: `[]` = linear AR, no hidden layers).
  - **Empty list** means simpler, faster linear relationships.
- `num_hidden_layers`: Global model depth (default: `0` = linear model).
- `d_hidden`: Hidden layer dimension (default: `16`, unused when `num_hidden_layers=0`).



#### 9. Hyperparameter Tuning `[model.tuning]`
- `trials`: Number of optimization trials (default: 100).
- `process_count`: Parallel CPU cores for Optuna optimization.
- `night_threshold`: Power level (Watts) below which data is ignored during evaluation to prevent "easy night wins" from skewing metrics.

#### 10. Nowcast Settings `[nowcast]`
Real-time damping factor correction.
- `use_damping_factor`: Toggle the live correction.
- `min_damping_factor` / `max_damping_factor`: Safety limits to prevent extreme forecast scaling based on temporary anomalies.

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

### 6. Nowcast (Real-Time Correction)

The standard forecast (Step 5) is based on global weather models, which have a **spatial resolution of several kilometers** (e.g., ~2 km to 10 km). Therefore, they cannot perfectly predict small-scale local events (e.g., a single cloud field or fog patches directly over your roof).

**Furthermore, a forecast is always a simulation.** It can never match the accuracy of a **real-time measurement**. The actual production data reflects the *ground truth* of the current conditions (reflecting exact local shadowing, dirt, snow, or hardware limits), which a general weather model cannot fully capture.

The **Nowcast** script runs frequently (e.g., every 15 minutes) to correct the forecast for the immediate future.

**How it works (Damping Factor):**

1.  **Weighted History:** It compares **Actual Production** vs. **Forecast** for the last 3 hours using a time-based decay.
    *   **Why?** To react faster to changing weather (e.g., fog clearing). Data from 2 hours ago is faded out to prioritize the current trend.
2.  **Damping Factor:** It calculates a performance ratio (e.g., if Production is only 50% of Forecast, Factor = 0.5).
3.  **Apply (Decaying Influence):** The factor is applied to the next 24 hours of the forecast with a **time-based decay** (Half-Life: 1 hour).
    *   **Why?** Weather anomalies (like a passing cloud or morning fog) are often temporary.
    *   **Concept:**
        *   **Short-Term (0-1h):** We trust our *local* 'Live-Correction' fully. (If it's foggy *now*, it will likely be foggy in 30 mins).
        *   **Long-Term (2h+):** We trust the *global* 'Weather Forecast' again. (An individual cloud now doesn't mean the whole day is ruined).

**Effect:**
*   **Now (0h):** 100% Correction.
*   **+1h:** 50% Correction.
*   **+2h:** 25% Correction.
*   **+4h:** ~6% Correction (Back to original Forecast).

**Start:**
- Windows: `nowcast.bat` or `python -m src.nowcast`
- Linux: `./nowcast.sh` or `python3 -m src.nowcast`

### 7. Run Pipeline (Full Automation)

Executes the complete workflow in order:
1. **Connection Test**: Checks InfluxDB health.
2. **Fetch Historic Weather**: Updates historic data.
3. **Fetch Future Weather**: Updates forecast data.
4. **Train Model**: Retrains the model.
5. **Create Forecast**: Generates the forecast.
6. **Nowcast**: Real-Time Correction.

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
- **`src/train.py`**: Trains the **Production** (PV) model. Fetches historical production and regressor data, trains NeuralProphet with AR mode, and saves `prophet_model.pkl`.
- **`src/forecast.py`**: Generates **Production** forecasts using multi-step prediction. Loads the model, fetches future weather data and historical context (for AR mode), predicts generation in 24-hour chunks, and writes to InfluxDB.
- **`src/nowcast.py`**: **Real-Time Correction**. Adjusts the forecast based on the last 3 hours of actual production to react to immediate weather changes (e.g., fog).

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
- **Why**: The NeuralProphet model needs to learn the relationship between the target variable (Production) and time/weather. With AR mode enabled (`n_lags > 0`), it also learns to use recent production patterns to improve short-term accuracy.

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

# Run Nowcast every 15 minutes (e.g., shortly after forecast, at minute 4, 19, 34, 49)
4,19,34,49 * * * * /path/to/fusionForecast/nowcast.sh >> /path/to/fusionForecast/logs/nowcast.log 2>&1

# Fetch historic weather data once a month (e.g., 1st of the month at 01:00 AM)
0 1 1 * * /path/to/fusionForecast/fetch_historic_weather.sh >> /path/to/fusionForecast/logs/fetch_historic.log 2>&1

# Train the model once a month (e.g., 1st of the month at 02:00 AM)
0 2 1 * * /path/to/fusionForecast/train.sh >> /path/to/fusionForecast/logs/train.log 2>&1
```

Ensure the scripts are executable (`chmod +x *.sh`) and the path is absolute.
Logs will be stored in the `logs/` directory (created automatically by `install_deps.sh`).
