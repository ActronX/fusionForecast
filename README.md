# FusionForecast

FusionForecast is an ML-based tool for forecasting time series data (e.g., PV generation) using [**Prophet**](https://facebook.github.io/prophet/), [**InfluxDB**](https://www.influxdata.com/) and [**Open-Meteo**](https://open-meteo.com/). It trains a model based on historical data and external regressors (e.g., weather forecasts) and writes the forecasts back into an InfluxDB.

![Logo](Logo.jpg)

## Features

- **Data Source**: Reads training data (target value and regressor) from InfluxDB.
- **Modeling**: Uses Facebook Prophet for time series forecasting.
- **Server-Side Aggregation**: Performs downsampling (e.g., to 1h means) directly in the database.
- **Configurable**: All settings (buckets, measurements, offsets) are defined in `settings.toml`.
- **Advanced Irradiance Modeling (Optional)**: Support for Perez POA, IAM losses, and SAPM cell temperature modeling via `pvlib`.
- **Offset Support**: Supports time offsets for regressors (e.g., to adjust time zones or lead times).

## Prerequisites

- Python 3.9+
- InfluxDB v2
- Access to relevant InfluxDB buckets
- **Historical PV Data:** At least 30 days of historical generation data are required for training. Ideally, 1 to 2 years of data should be available for better accuracy.
    - **Frequency:** Data should preferably be stored every 15 to 60 minutes. It is not critical if individual data points are missing; the system is designed to handle gaps and inexact timestamps.
    - **Timezone:** The correct timezone must be observed. The system operates in UTC. An offset can be defined in the settings.

## Docker Deployment (Recommended)

For production use, Docker provides the easiest and most reliable deployment method. The setup includes:
- **Zero-Manual-Config**: `settings.toml` is automatically generated and synchronized from your `.env`.
- **Auto-Initialization**: InfluxDB is automatically set up with all required buckets.
- **Persistent Data**: Models and logs are stored on the host.

### Quick Start

1. **Clone Repository & Prepare Configuration**:
   ```bash
   git clone <repository-url>
   cd fusionForecast
   
   # Copy and edit environment variables
   cp .env.example .env
   nano .env  # Set your coordinates and passwords
   ```

2. **Start Containers**:
   ```bash
   docker-compose up -d
   ```
   
   This will:
   - Start InfluxDB and create all required buckets automatically.
   - Dynamically generate `settings.toml` inside the container using your `.env` values.
   - Set up automated forecasts every 15 minutes via Cron.

3. **Monitor Setup**:
   ```bash
   # Watch logs during initial setup
   docker-compose logs -f fusionforecast
   
   # Check container status
   docker-compose ps
   ```

### Configuration via `.env`

The Docker setup uses environment variables to configure both InfluxDB and the application.

| Variable | Description | Default |
| :--- | :--- | :--- |
| `INFLUXDB_TOKEN` | API Token for InfluxDB access. | (Predefined in `.env.example`) |
| `INFLUXDB_ORG` | Organization name in InfluxDB. | `fusionforecast` |
| `STATION_LATITUDE` | Decimal latitude of your location. | `52.5200` |
| `STATION_LONGITUDE` | Decimal longitude of your location. | `13.4050` |
| `STATION_TILT` | Tilt of your PV panels (0-90°). | `30` |
| `STATION_AZIMUTH` | Azimuth (0=South, -90=East, 90=West). | `0` |
| `MODEL_TRAINING_DAYS`| Number of days to use for model training. | `30` |
| `MAX_POWER_CLIP` | Max system output in Watts (physical limit).| `6000` |


### Importing Historical PV Data (at least 30 days)

If you have existing PV generation history (e.g., exported as CSV from your inverter or an online portal), you can import it into InfluxDB to train the model. **At least 30 days of historical data are required** to achieve accurate forecasts from the start. This is best done **before** or during initial setup.

**CSV Format:** Simple CSV **without a header line**:
- **Column 1**: Timestamp in **UTC** (e.g., `YYYY-MM-DD HH:MM:SS`)
- **Column 2**: Power/Energy value in Watts

**Example (`my_data.csv`):**
```csv
2024-01-01 12:00:00, 1500.5
2024-01-01 12:15:00, 1480.0
2024-01-01 12:30:00, 1620.2
2024-01-01 12:45:00, 1590.8
2024-01-01 13:00:00, 1710.0
2024-01-01 13:15:00, 1820.5
```

**How to Import:**
1.  **Copy file** into container: `docker cp my_data.csv fusionforecast-app:/app/my_data.csv`
2.  **Run import**: `docker exec fusionforecast-app python3 -m src.import_pv_history my_data.csv`
#### Option 2: Manual Injection via Curl
To train the model, you must push historical data into InfluxDB using the following mapping:
- **Bucket**: `energy_data` (History)
- **Measurement**: `energy_meter` (Produced)
- **Field**: `power_produced` (Produced)

If you have a script or sensor that can send HTTP requests, you can push data directly (Note: for historical data you **must** include a Unix timestamp):

```bash
# Format: <measurement> <field>=<value> <timestamp_in_seconds>
# Example: Two data points for Jan 1st 2024
curl -X POST "http://localhost:8086/api/v2/write?org=fusionforecast&bucket=energy_data&precision=s" \
  -H "Authorization: Token YOUR_TOKEN" \
  --data-raw "energy_meter power_produced=1500.0 1704110400"

curl -X POST "http://localhost:8086/api/v2/write?org=fusionforecast&bucket=energy_data&precision=s" \
  -H "Authorization: Token YOUR_TOKEN" \
  --data-raw "energy_meter power_produced=1550.0 1704111300"
```
3.  **Retrain model**: `docker exec fusionforecast-app python3 -m src.train`

### Pushing Live Data via Curl (Nowcast)

For real-time forecasting and correction, you must regularly push your current PV production into the `live` bucket using the following mapping:
- **Bucket**: `energy_meter` (Live)
- **Measurement**: `energy_meter` (Live)
- **Field**: `production` (Live)

```bash
curl -X POST "http://localhost:8086/api/v2/write?org=fusionforecast&bucket=energy_meter&precision=s" \
  -H "Authorization: Token YOUR_TOKEN" \
  --data-raw "energy_meter production=1250.5"
```

**Parameters for Curl:**
- `bucket`: Target bucket (e.g., `energy_meter` for live, `energy_data` for history).
- `--data-raw`: Format is `measurement field=value` (e.g., `energy_meter production=1200`).
- `precision=s`: Uses the current server time for the data point.

> [!TIP]
> You can find all bucket, measurement, and field names in your `settings.toml` under the `[influxdb]` sections.

### Container Management

```bash
# View logs
docker-compose logs -f [fusionforecast|influxdb]

# Restart services
docker-compose restart

# Stop all containers
docker-compose down

# Stop and remove volumes (⚠️ deletes all InfluxDB data!)
docker-compose down -v

# Rebuild and restart after code changes
docker-compose up -d --build

# Open shell in container
docker exec -it fusionforecast-app bash
```

### Data Persistence

Docker volumes ensure your data survives container restarts:
- **influxdb-data**: All InfluxDB measurements and forecasts.
- **./models**: Trained Prophet models (mounted from host).
- **./logs**: Application logs (mounted from host).

### Automated Updates

The container automatically:
- ✅ Fetches weather forecasts every 15 minutes.
- ✅ Generates PV forecasts every 15 minutes.
- ✅ Updates nowcast corrections every 15 minutes.
- ✅ Retrains model monthly (1st of month at 02:00).

### Troubleshooting

- **Coordinates not updating?** If you change coordinates in `.env`, run `docker-compose up -d` to regenerate the `settings.toml` inside the container.
- **InfluxDB connection failed?** Ensure `INFLUXDB_TOKEN` in `.env` matches the one used during the very first initialization.
- **Bucket conflicts?** The system is idempotent. If a bucket already exists, it will skip creation and continue.

**InfluxDB buckets missing:**
```bash
# Verify bucket creation
docker-compose exec influxdb influx bucket list

# If buckets are missing, restart InfluxDB
docker-compose restart influxdb
```

**Reset everything:**
```bash
docker-compose down -v
rm -rf models/ logs/
docker-compose up -d
```

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
- **Physical Model Fields** (Required if `use_pvlib = true`):
    - `diffuse` / `direct`: IR components for Perez POA.
    - `poa_perez`: Resulting raw plane-of-array irradiance.
    - `effective_irradiance`: Irradiance after IAM reflection losses.
    - `temp_amb` / `wind_speed` / `temp_cell`: Ambient and calculated cell temperatures.

#### 5. Weather Source `[weather.open_meteo]`
Configures how weather data is fetched from the Open-Meteo API.
- `historic` / `forecast`: Separate endpoints for training history and future predictions.
- `models`: Selection of the weather model (e.g., `best_match`, `icon_d2`).
- `minutely_15`: The specific variable to fetch (default: `global_tilted_irradiance_instant`).

#### 6. Model Parameters `[model]`
- `path`: File path for the trained Prophet model (`.pkl`).
- `training_days`: Period of history to use (default: 731 days for 2 full years).
- `forecast_days`: How many days to predict into the future (default: 14).

#### 7. Preprocessing `[model.preprocessing]`
Fine-tuning of data before it entering the ML model.
- `max_power_clip`: Hard limit in Watts to remove outliers or non-physical spikes.
- `produced_scale` / `regressor_scale`: Multipliers to normalize data (e.g., if your meter stores kW but you want Watts).
- `produced_offset` / `regressor_offset`: Crucial for aligning timestamps (e.g., if one source is 1h ahead).

#### 8. Prophet ML Engine `[model.prophet]`
Specific settings for the Facebook Prophet model.
- `use_pvlib`: **The Physics Toggle.** Enables physical solar modelling.
    - While Prophet learns seasonally, enabling `use_pvlib` allows the model to "understand" the physical geometry of your panels, potentially leading to better accuracy in complex weather.
- `changepoint_prior_scale`: Controls trend flexibility. Lower = smoother, Higher = more reactive to changes.
- `seasonality_prior_scale`: Intensity of yearly/daily cycles.
- `regressor_prior_scale`: How much the model trusts the weather forecast vs. its internal patterns.

#### 9. Physical Model coefficients `[model.pvlib]`
Parameters for `pvlib` when `use_pvlib` is active.
- `iam_b`: Incidence Angle Modifier coefficient (typical value 0.05).
- `sapm_a` / `sapm_b` / `sapm_deltaT`: Sandia Array Performance Model coefficients for cell temperature.

#### 10. Hyperparameter Tuning `[model.tuning]`
- `process_count`: Parallel CPU cores for grid search.
- `night_threshold`: Power level (Watts) below which data is ignored during evaluation to prevent "easy night wins" from skewing metrics.

#### 11. Nowcast Settings `[nowcast]`
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

The training script loads historical data, trains the Prophet model, and saves it locally as a `.pkl` file.

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

To optimize the model's accuracy, you can tune the hyperparameters (e.g., `changepoint_prior_scale`, `seasonality_mode`) using a grid search. The parameters to be tested are defined in `settings.toml` under `[prophet.tuning]`.

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
- **`src/train.py`**: Trains the **Production** (PV) model. Fetches historical production and regressor data, trains Prophet, and saves `prophet_model.pkl`.
- **`src/forecast.py`**: Generates **Production** forecasts. Loads the model, fetches future weather data, predicts generation, and writes to InfluxDB.
- **`src/nowcast.py`**: **Real-Time Correction**. Adjusts the forecast based on the last 3 hours of actual production to react to immediate weather changes (e.g., fog).

### Data Fetching & Calculations
- **`src/fetch_future_weather.py`**: Fetches **current** weather forecasts from Open-Meteo. If `use_pvlib` is enabled, it automatically triggers the consolidated calculation.
- **`src/fetch_historic_weather.py`**: Fetches **historical** weather data. If `use_pvlib` is enabled, it automatically triggers the consolidated calculation.
- **`src/calc_effective_irradiance.py`**: **Consolidated Physics Model**. Calculates Plane of Array (POA) irradiance, applies Incidence Angle Modifier (IAM) losses, and computes the SAPM cell temperature.

### Utilities & Maintenance

- **`src/tune.py`**: Performs **Hyperparameter Tuning** using a grid search to find the optimal Prophet parameters (e.g., `changepoint_prior_scale`) for your specific data.

- **`src/plot_model.py`**: Generates interactive Plotly charts of the model components (trend, seasonality) for visual inspection.


## InfluxDB Data Flow

This section describes which data is read from and written to InfluxDB, and why this is necessary.

### 1. Training (Model Creation)
*Scripts: `src/train.py`*

- **Reads**:
    - **Production History** (`buckets.history_produced`): Actual historical PV generation data.
    - **Regressor History** (`buckets.regressor_history`): Historical weather data (e.g., solar irradiance) corresponding to the production history.
- **Why**: The Prophet model needs to learn the relationship between the target variable (Production) and time/weather. For example, it learns that "high irradiance = high production".

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
