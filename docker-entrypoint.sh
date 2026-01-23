#!/bin/bash
set -e

echo "=== FusionForecast Container Starting ==="

# Wait for InfluxDB to be fully ready
echo "Waiting for InfluxDB to be ready..."
until curl -sf http://influxdb:8086/health > /dev/null 2>&1; do
  echo "  InfluxDB is unavailable - sleeping"
  sleep 2
done
echo "✓ InfluxDB is ready"

# Check if InfluxDB token is provided
echo "Checking InfluxDB token..."
INFLUXDB_ORG=${INFLUXDB_ORG:-fusionforecast}

if [ -z "$INFLUXDB_TOKEN" ]; then
  echo "✗ INFLUXDB_TOKEN environment variable is not set"
  echo "Please set INFLUXDB_TOKEN in your .env file"
  exit 1
fi

echo "✓ InfluxDB token found"
echo "  Token: ${INFLUXDB_TOKEN:0:10}...${INFLUXDB_TOKEN: -10}"


# Check if settings.toml template exists
if [ ! -f "/app/settings.toml.template" ]; then
  echo "⚠ settings.toml.template not found!"
  echo "Please provide a settings.toml file via volume mount to /app/settings.toml.template"
  exit 1
fi

echo "✓ Configuration template found"

# Update settings.toml from template with the retrieved token
echo "Generating settings.toml from template..."

# Create a base settings.toml
cp /app/settings.toml.template /app/settings.toml

# Replace values in settings.toml specifically in the [influxdb] section
sed -i "/\[influxdb\]/,/\[/ s|token = .*|token = \"$INFLUXDB_TOKEN\"|" /app/settings.toml
sed -i "/\[influxdb\]/,/\[/ s|url = .*|url = \"http://influxdb:8086\"|" /app/settings.toml
sed -i "/\[influxdb\]/,/\[/ s|org = .*|org = \"$INFLUXDB_ORG\"|" /app/settings.toml

# Replace values in the [station] section if environment variables are set
if [ -n "$STATION_LATITUDE" ]; then sed -i "/\[station\]/,/\[/ s|latitude = .*|latitude = $STATION_LATITUDE|" /app/settings.toml; fi
if [ -n "$STATION_LONGITUDE" ]; then sed -i "/\[station\]/,/\[/ s|longitude = .*|longitude = $STATION_LONGITUDE|" /app/settings.toml; fi
if [ -n "$STATION_TILT" ]; then sed -i "/\[station\]/,/\[/ s|tilt = .*|tilt = $STATION_TILT|" /app/settings.toml; fi
if [ -n "$STATION_AZIMUTH" ]; then sed -i "/\[station\]/,/\[/ s|azimuth = .*|azimuth = $STATION_AZIMUTH|" /app/settings.toml; fi

# Replace values in the [model] section if environment variables are set
if [ -n "$MODEL_TRAINING_DAYS" ]; then sed -i "/\[model\]/,/\[/ s|training_days = .*|training_days = $MODEL_TRAINING_DAYS|" /app/settings.toml; fi

# Replace values in the [model.preprocessing] section if environment variables are set
if [ -n "$MAX_POWER_CLIP" ]; then sed -i "/\[model.preprocessing\]/,/\[/ s|max_power_clip = .*|max_power_clip = $MAX_POWER_CLIP|" /app/settings.toml; fi

echo "✓ Settings generated and localized for Docker"

# Create required InfluxDB buckets if they don't exist
echo "Creating required InfluxDB buckets..."
INFLUX_URL="http://influxdb:8086"

# Extract bucket names dynamically from settings.toml
# This AWK script looks for the [influxdb.buckets] section and extracts values after the '=' sign
BUCKETS=$(awk '/\[influxdb\.buckets\]/{flag=1;next} /^\[/{flag=0} flag && /=/{gsub(/^[ \t]+|[ \t]+$/, "", $0); split($0, a, "="); gsub(/^[ \t"]+|[ \t"]+|#.*$/, "", a[2]); if(a[2] != "") print a[2]}' /app/settings.toml | sort -u)

if [ -z "$BUCKETS" ]; then
  echo "  ⚠ Could not parse buckets from settings.toml, using defaults"
  BUCKETS="energy_data weather_data forecast_results energy_meter"
fi

for BUCKET in $BUCKETS; do
  echo "  Checking bucket: $BUCKET"
  
  # Check if bucket exists
  BUCKET_CHECK=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: Token $INFLUXDB_TOKEN" \
    "${INFLUX_URL}/api/v2/buckets?org=${INFLUXDB_ORG}&name=${BUCKET}")
  
  if [ "$(echo "${BUCKET_CHECK:-0}" | head -n1)" -eq 200 ]; then
    # Verify bucket actually exists in response
    BUCKET_EXISTS=$(curl -s \
      -H "Authorization: Token $INFLUXDB_TOKEN" \
      "${INFLUX_URL}/api/v2/buckets?org=${INFLUXDB_ORG}&name=${BUCKET}" | grep -c "\"name\":\"${BUCKET}\"" || echo "0")
    
    if [ "$(echo "${BUCKET_EXISTS:-0}" | head -n1)" -gt 0 ]; then
      echo "    ✓ Bucket '$BUCKET' already exists"
      continue
    fi
  fi
  
  # Get org ID
  echo "    Fetching ID for organization '${INFLUXDB_ORG}'..."
  ORG_RESPONSE=$(curl -s -H "Authorization: Token $INFLUXDB_TOKEN" "${INFLUX_URL}/api/v2/orgs?org=${INFLUXDB_ORG}")
  
  # Use Python to reliably parse the ID from the possibly multi-line JSON response
  ORG_ID=$(echo "$ORG_RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['orgs'][0]['id'])" 2>/dev/null || echo "")
  
  if [ -z "$ORG_ID" ]; then
    echo "    ⚠ Could not find ID for organization '${INFLUXDB_ORG}'"
    echo "    Response: $ORG_RESPONSE"
    continue
  fi
  
  echo "    Org ID: $ORG_ID"

  # Create bucket
  echo "    → Creating bucket '$BUCKET'..."
  CREATE_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "${INFLUX_URL}/api/v2/buckets" \
    -H "Authorization: Token $INFLUXDB_TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"orgID\": \"${ORG_ID}\", \"name\": \"${BUCKET}\", \"retentionRules\": [{\"type\": \"expire\", \"everySeconds\": 0}]}")
  
  HTTP_CODE=$(echo "$CREATE_RESPONSE" | tail -n1)
  
  if [ "$HTTP_CODE" -eq 201 ]; then
    echo "    ✓ Bucket '$BUCKET' created successfully"
  else
    echo "    ⚠ Could not create bucket '$BUCKET' (HTTP $HTTP_CODE)"
    echo "    Response: $(echo "$CREATE_RESPONSE" | head -n-1)"
  fi
done

echo "✓ Bucket creation complete"

# Validate configuration
echo "Validating configuration..."
python3 -c "from src.config import settings; print('✓ Configuration loaded successfully')" || {
  echo "✗ Configuration validation failed"
  exit 1
}

# Test InfluxDB connection
echo "Testing InfluxDB connection..."
python3 test_connection.py || {
  echo "✗ InfluxDB connection test failed"
  echo "Please check your settings.toml configuration"
  exit 1
}

# Check if model exists, if not run initial training
if [ ! -f "/app/models/prophet_model.pkl" ]; then
  echo "No trained model found. Running initial training..."
  echo "This may take several minutes..."
  
  # Fetch historic weather data first
  echo "[1/3] Fetching historic weather data..."
  python3 -m src.fetch_historic_weather || {
    echo "⚠ Warning: Failed to fetch historic weather data"
  }
  
  # Train model
  echo "[2/3] Training Prophet model..."
  python3 -m src.train || {
    echo "✗ Training failed"
    exit 1
  }
  
  # Create initial forecast
  echo "[3/3] Creating initial forecast..."
  python3 -m src.fetch_future_weather && \
  python3 -m src.forecast || {
    echo "⚠ Warning: Initial forecast failed"
  }
  
  echo "✓ Initial setup complete"
else
  echo "✓ Trained model found, skipping initial training"
fi

# Do NOT start cron here, it's started by CMD in Dockerfile
echo "=== FusionForecast is ready ==="
echo "Logs are written to /app/logs/"
echo ""
echo "Scheduled tasks:"
echo "  - Fetch weather data: every 15 minutes"
echo "  - Create forecast: every 15 minutes"
echo "  - Update nowcast: every 15 minutes"
echo ""

# Execute the main command (keeping container alive)
exec "$@"
