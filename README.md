# FusionForecast

FusionForecast is an ML-based tool for forecasting time series data (e.g., PV generation) using [**NeuralProphet**](https://neuralprophet.com/), [**InfluxDB**](https://www.influxdata.com/) and [**Open-Meteo**](https://open-meteo.com/). It trains a model based on historical data and external regressors (e.g., weather forecasts) and writes the forecasts back into an InfluxDB.

## Data Flow

![Dataflow Diagram](doc/dataflow_diagram.png)


## Features

- **Data Source**: Reads training data (target value and regressor) from InfluxDB.
- **Modeling**: Uses NeuralProphet (PyTorch) for time series forecasting.
- **Intraday Correction**: Dual-mechanism approach for real-time forecast adjustments:
  - **AR-Net** (`n_lags=96`): Learns autoregressive patterns from the last 24 hours of production to capture short-term trends.
  - **Intraday Correction**: The AR mechanism automatically uses recent live production data to adjust the forecast start point.
- **Server-Side Aggregation**: Performs downsampling (e.g., to 1h means) directly in the database.
- **Configurable**: All settings (buckets, measurements, offsets) are defined in `settings.toml`.
- **Offset Support**: Supports time offsets for regressors (e.g., to adjust time zones or lead times).

## Prerequisites

- Python 3.9+
- InfluxDB v2
- Access to relevant InfluxDB buckets
- **Historical PV Data:** At least 30 days of historical generation data are required for training. Ideally, 1 to 2 years of data should be available for better accuracy.
    - **Frequency:** Data should preferably be stored every 15 to 60 minutes. It is not critical if individual data points are missing; the system is designed to handle gaps and inexact timestamps.
    - **Timezone:** The correct timezone must be observed. The system operates in UTC.

## Docker Deployment (Recommended)

For production use, Docker provides the easiest and most reliable deployment method. The setup includes:
- **Zero-Manual-Config**: `settings.toml` is automatically generated and synchronized from your `.env`.
- **Auto-Initialization**: InfluxDB is automatically set up with all required buckets.
- **Persistent Data**: Models and logs are stored on the host.

### Quick Start

1. **Clone Repository & Prepare Configuration**:
   ```bash
   git clone <repository-url>
   cd fusionForecast/docker
   
   # Copy and edit environment variables
   cp .env.example .env
   nano .env  # Set your coordinates and passwords
   ```

2. **Prepare Data (Recommended)**:
   - Place your historical data file as `measurements.csv` in the project root (one level up).
   - Edit `docker-compose.yml` (already in `docker/` folder) and uncomment the volume line:
     ```yaml
     volumes:
       - ../measurements.csv:/app/measurements.csv
     ```

3. **Start Containers**:
   ```bash
   cd docker
   docker-compose up -d
   ```
   
   This will:
   - Start InfluxDB and create all required buckets automatically.
   - Dynamically generate `settings.toml` inside the container using your `.env` values.
   - Set up automated forecasts every 15 minutes via Cron.

4. **Monitor Setup**:
   ```bash
   # Watch logs during initial setup
   docker-compose logs -f fusionforecast
   
   # Check container status
   docker-compose ps
   ```

5. **Access InfluxDB Dashboard**:
   - URL: [http://localhost:8086](http://localhost:8086)
   - Username: `admin` (or as defined in your setup)
   - Password: `password` (or as defined in your setup)
   - Organization: `fusionforecast`


### Configuration via `.env`

The Docker setup uses environment variables to configure both InfluxDB and the application.

| Variable | Description | Default |
| :--- | :--- | :--- |
| `INFLUXDB_TOKEN` | API Token for InfluxDB access. | (Predefined) |
| `INFLUXDB_ORG` | Organization name in InfluxDB. | `fusionforecast` |
| `STATION_LATITUDE` | Decimal latitude of your location. | `52.5200` |
| `STATION_LONGITUDE` | Decimal longitude of your location. | `13.4050` |
| `STATION_TILT` | Tilt (0=flat, 90=vertical). | `30` |
| `STATION_AZIMUTH` | Azimuth (0=South, -90=East, 90=West). | `0` |
| `STATION_ALTITUDE` | Station altitude in meters. | `0` |
| `MODEL_TRAINING_DAYS`| Days of history to use for training. | `30` |
| `MAX_POWER_CLIP` | Max system output in Watts (outlier clipping).| `6000` |


### Importing Historical PV Data

To train the model, you must push historical data into InfluxDB.

👉 **[Read the full Data Import Guide](doc/import_data.md)** for details on:
- CSV Import (Volume or Manual)
- Manual Injection via Curl
- Pushing Live Data for Intraday Correction

### Container Management & Manual Execution

👉 **[Read the full Docker Management Guide](doc/docker_management.md)** for details on:
- Starting/Stopping/Restarting containers
- Viewing Logs
- **Manually triggering scripts** (Train, Forecast, Fetch Weather) inside the container

### Data Persistence

Docker volumes ensure your data survives container restarts:
- **influxdb-data**: All InfluxDB measurements and forecasts.
- **./models**: Trained NeuralProphet models (mounted from host).
- **./logs**: Application logs (mounted from host).

### Automated Updates

The container automatically:
- ✅ Fetches weather forecasts every 15 minutes.
- ✅ Generates PV forecasts with intraday correction every 15 minutes.
- ✅ Retrains model monthly (1st of month at 02:00).

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

### InfluxDB Login Guide for FusionForecast
👉 **[Read the InfluxDB Documentation](doc/influxdb-login-guide.md)**
![influx](doc/influx.jpg)

## Manual Installation
For manual installation details, please refer to [manual_installation.md](doc/manual_installation.md).

# Smart Consumer Control (Node-RED)

For users who want to use the forecast data to control physical devices (e.g., heating, EV charging), I provide a ready-to-use **Node-RED** flow.

![Wiring](node_red/Wiring.jpg)

`[Inject] --> [Template] --> [InfluxDB] --> [Function] --> [Output]`

Instead of simple threshold switching, it calculates the **projected solar energy surplus** for the next 24 hours. It switches the consumer **ON** only if the surplus is sufficient to cover the runtime costs without draining the home battery below a reserved level.

It includes advanced protection features:
* **Hysteresis:** Prevents rapid toggling ("flip-flopping") by requiring a specific charge level recovery.
* **Dynamic Reserve:** Maintains a high safety buffer when battery is low, but reduces it when battery is full to maximize capacity usage.
* **Safety Guard:** Prevents operation if forecast data is incomplete or outdated.
* **Real-Time Forecast Correction:** Dynamically adjusts the forecast curve ("Damping Factor") based on the actual solar performance since sunrise. If the day is cloudier/sunnier than predicted, the future forecast is scaled accordingly.
* **Battery Protection:** Hard cutoff when SoC is critically low.

👉 **[Read the full Node-RED Documentation](node_red/README.md)**
