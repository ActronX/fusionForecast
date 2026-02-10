# Importing PV Data

This guide explains how to import historical PV generation data for training and how to push live data for intraday corrections.

## 1. Importing Historical Data (Required for Training)

If you have existing PV generation history (e.g., exported as CSV from your inverter or an online portal), you can import it into InfluxDB to train the model. **At least 30 days of historical data are required** to achieve accurate forecasts from the start. This is best done **before** or during initial setup.

### CSV Format

The CSV file must be simple and **without a header line**:

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

### Method 1: Volume Import (Recommended)
The easiest way is to mount your CSV file into the container. It will be automatically imported on startup.

**Docker Compose method:**
1.  Save your CSV file as `measurements.csv` in the same folder as `docker-compose.yml`.
2.  Add the volume to `docker-compose.yml`:
    ```yaml
    volumes:
      - ./measurements.csv:/app/measurements.csv
    ```
3.  Start the container: `cd docker && docker-compose up -d`

### Method 2: Manual Copy & Import (Interactive)
If you prefer to import data into a running container without restarting:

1.  **Copy file** into container: `docker cp measurements.csv fusionforecast-app:/app/measurements.csv`
2.  **Run import**: `docker exec fusionforecast-app python3 -m src.import_pv_history measurements.csv`
3.  **Run Pipeline**: `docker exec fusionforecast-app python3 run_pipeline.py`
    *(Fetches weather, retrains model, and calculates forecast)*

### Method 3: Manual Injection via Curl
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

## 2. Pushing Live Data (Intraday Correction)

For real-time intraday correction, you can optionally push your current PV production into the `live` bucket. The model uses this data as a **lagged regressor** (last 2 hours of actual production) to dynamically adjust forecasts. 

**Mapping:**
- **Bucket**: `energy_meter` (Live)
- **Measurement**: `energy_meter` (Live)
- **Field**: `production` (Live)

```bash
curl -X POST "http://localhost:8086/api/v2/write?org=fusionforecast&bucket=energy_meter&precision=s" \
  -H "Authorization: Token YOUR_TOKEN" \
  --data-raw "energy_meter production=1250.5"
```



## 3. Pushing Battery SoC Data (Node-RED Consumer Control)

If you are using the **Smart Consumer Control** feature with Node-RED, the system needs access to your battery's **State of Charge (SoC)** to make intelligent decisions about when to activate high-power consumers.

### Why is this necessary?

The Node-RED flow calculates the **projected solar energy surplus** for the next 24 hours and decides whether to switch on consumers (e.g., heating, EV charging) based on:
- Current battery charge level
- Forecasted solar production
- Expected consumption

Without real-time SoC data, the system cannot:
- Prevent battery discharge below safe levels
- Apply dynamic reserve strategies
- Optimize energy usage based on available battery capacity

### Battery Data Mapping

**Mapping:**
- **Bucket**: `batterie`
- **Measurement**: `pv`
- **Field**: `USOC` (User State of Charge, typically 0-100%)

### Push Command

If your battery management system or inverter can send HTTP requests, use the following curl command to push the current SoC:

```bash
curl -X POST "http://localhost:8086/api/v2/write?org=fusionforecast&bucket=batterie&precision=s" \
  -H "Authorization: Token YOUR_TOKEN" \
  --data-raw "pv USOC=75.5"
```

**Example with different SoC values:**

```bash
# Battery at 85% charge
curl -X POST "http://localhost:8086/api/v2/write?org=fusionforecast&bucket=batterie" \
  -H "Authorization: Token YOUR_TOKEN" \
  --data-raw "pv USOC=85.0"

# Battery at 45% charge
curl -X POST "http://localhost:8086/api/v2/write?org=fusionforecast&bucket=batterie" \
  -H "Authorization: Token YOUR_TOKEN" \
  --data-raw "pv USOC=45.0"
```

> [!NOTE]
> The Node-RED flow queries the last SoC value within the past hour. If no data is found, it falls back to 0% (safe mode). For optimal operation, push SoC data every 1-5 minutes.

---

### General Parameters for Curl Commands

- `bucket`: Target bucket (e.g., `energy_meter` for live production, `energy_data` for history, `batterie` for battery data).
- `--data-raw`: Format is `measurement field=value` (e.g., `energy_meter production=1200` or `pv USOC=75.5`).
- `precision=s`: Uses the current server time for the data point.

> [!TIP]
> You can find all bucket, measurement, and field names in your `settings.toml` under the `[influxdb]` sections.
