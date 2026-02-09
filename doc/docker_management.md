# Docker Management & Manual Script Execution

This guide provides commands for managing the FusionForecast Docker container and manually triggering specific pipeline steps.

## Container Management

All commands should be run from the project root (or `docker/` directory depending on your `docker-compose` context, usually root if using the provided makefile or scripts, but here we assume standard docker-compose usage).

```bash
# View logs (all services)
docker-compose logs -f

# View logs (specific service)
docker-compose logs -f fusionforecast
docker-compose logs -f influxdb

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

## Manual Script Execution

You can manually trigger specific parts of the pipeline inside the running container without waiting for the cron schedule.

### 1. Fetch Weather Data

**Current Forecast (Future):**
Updates the `regressor_future` bucket with the latest weather forecast.
```bash
docker exec fusionforecast-app python3 -m src.fetch_future_weather
```

**Historical Data (Past):**
Updates the `regressor_history` bucket. Useful if you missed some days or re-initialized the database.
```bash
docker exec fusionforecast-app python3 -m src.fetch_historic_weather
```

### 2. Train Model

Retrains the NeuralProphet model using the latest data in `energy_data` (history) and `regressor_history`.
```bash
docker exec fusionforecast-app python3 -m src.train
```

### 3. Generate Forecast

Generates a new production forecast based on the trained model and current weather forecast.
```bash
docker exec fusionforecast-app python3 -m src.forecast
```

### 4. Run Full Pipeline

Executes the entire sequence: Check Connection -> Fetch Weather -> Train -> Forecast.
```bash
docker exec fusionforecast-app python3 run_pipeline.py
```
