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
   cd fusionForecast
   
   # Copy and edit environment variables
   cp .env.example .env
   nano .env  # Set your coordinates and passwords
   ```

2. **Prepare Data (Recommended)**:
   - Place your historical data file as `measurements.csv` in the folder.
   - Edit `docker-compose.yml` and uncomment the volume line:
     ```yaml
     volumes:
       - ./measurements.csv:/app/measurements.csv
     ```

3. **Start Containers**:
   ```bash
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
| `NIGHT_THRESHOLD` | Ignore values <= this during evaluation (Watts). | `50` |

### Importing Historical PV Data (at least 30 days)

If you have existing PV generation history (e.g., exported as CSV from your inverter or an online portal), you can import it into InfluxDB to train the model. **At least 30 days of historical data are required** to achieve accurate forecasts from the start. This is best done **before** or during initial setup.

**CSV Format:** Simple CSV **without a header line**:

> [!IMPORTANT]
> **Important:** The CSV file must **NOT** contain a header row (e.g., "Timestamp,Value"). Column 1 must be the timestamp, and Column 2 the value.
- **Column 1**: Timestamp in **UTC**. Supported formats include:
    - ISO 8601: `2024-12-31T12:00:00`
    - Standard: `2024-12-31 12:00:00`
    - Date with points/slashes: `2024.12.31 12:00` or `12/31/2024 12:00`
    - With Timezone: `2024-12-31 12:00:00+02:00` (will be converted to UTC)
    - *Note: For unambiguous parsing, `YYYY-MM-DD` format is recommended.*
- **Column 2**: Power/Energy value in Watts

**Example (`measurements.csv`):**
```csv
2024-12-31 12:00:00, 1500.5
2024-12-31 12:15:00, 1480.0
2024-12-31 12:30:00, 1620.2
2024-12-31 12:45:00, 1590.8
2024-12-31 13:00:00, 1710.0
2024-12-31 13:15:00, 1820.5
```

**How to Import:**

#### Option 1: Volume Import (Recommended)
The easiest way is to mount your CSV file into the container. It will be automatically imported on startup.

**Docker Compose method:**
1.  Save your CSV file as `measurements.csv` in the same folder as `docker-compose.yml`.
2.  Add the volume to `docker-compose.yml`:
    ```yaml
    volumes:
      - ./measurements.csv:/app/measurements.csv
    ```
3.  Start the container: `docker-compose up -d`

#### Option 2: Manual Copy & Import (Interactive)
If you prefer to import data into a running container without restarting:

1.  **Copy file** into container: `docker cp measurements.csv fusionforecast-app:/app/measurements.csv`
2.  **Run import**: `docker exec fusionforecast-app python3 -m src.import_pv_history measurements.csv`
3.  **Run Pipeline**: `docker exec fusionforecast-app python3 run_pipeline.py`
    *(Fetches weather, retrains model, and calculates forecast)*

#### Option 3: Manual Injection via Curl
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

# Apply changes from .env (recreates container with new env vars, no rebuild needed)
docker-compose up -d

# Force full rebuild (e.g. after code changes)
docker-compose build --no-cache fusionforecast
docker-compose up -d
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

### InfluxDB Login Guide for FusionForecast
👉 **[Read the InfluxDB Documentation](influxdb-login-guide.md)**
![influx](influx.jpg)

## Manual Installation (Alternative)
👉 **[Read the Manual Installation Documentation](manual_installation.md)**

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
