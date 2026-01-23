#!/bin/bash
set -e

echo "=== InfluxDB Bucket Initialization ==="

# Wait for InfluxDB to be fully ready
sleep 5

# Read environment variables
INFLUX_URL="http://localhost:8086"
INFLUX_TOKEN="${DOCKER_INFLUXDB_INIT_ADMIN_TOKEN}"
INFLUX_ORG="${DOCKER_INFLUXDB_INIT_ORG}"

# List of buckets needed by FusionForecast
# These should match the buckets defined in settings.toml
BUCKETS=(
  "energy_data"           # history_produced
  "weather_data"          # regressor_history & regressor_future
  "forecast_results"      # target_forecast
  "energy_meter"          # live
)

echo "Creating required buckets for FusionForecast..."

for BUCKET in "${BUCKETS[@]}"; do
  echo "Checking bucket: $BUCKET"
  
  # Check if bucket exists
  BUCKET_EXISTS=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: Token $INFLUX_TOKEN" \
    "$INFLUX_URL/api/v2/buckets?org=$INFLUX_ORG&name=$BUCKET")
  
  if [ "$BUCKET_EXISTS" -eq 200 ]; then
    # Check if any bucket with this name exists in response
    BUCKET_COUNT=$(curl -s \
      -H "Authorization: Token $INFLUX_TOKEN" \
      "$INFLUX_URL/api/v2/buckets?org=$INFLUX_ORG&name=$BUCKET" | grep -c "\"name\":\"$BUCKET\"" || echo "0")
    
    if [ "$BUCKET_COUNT" -gt 0 ]; then
      echo "  ✓ Bucket '$BUCKET' already exists"
      continue
    fi
  fi
  
  # Create bucket
  echo "  → Creating bucket '$BUCKET'..."
  CREATE_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$INFLUX_URL/api/v2/buckets" \
    -H "Authorization: Token $INFLUX_TOKEN" \
    -H "Content-Type: application/json" \
    -d "{
      \"orgID\": \"$(curl -s -H "Authorization: Token $INFLUX_TOKEN" "$INFLUX_URL/api/v2/orgs?org=$INFLUX_ORG" | grep -o '\"id\":\"[^\"]*\"' | head -1 | cut -d'"' -f4)\",
      \"name\": \"$BUCKET\",
      \"retentionRules\": [{\"type\": \"expire\", \"everySeconds\": 0}]
    }")
  
  HTTP_CODE=$(echo "$CREATE_RESPONSE" | tail -n1)
  
  if [ "$HTTP_CODE" -eq 201 ]; then
    echo "  ✓ Bucket '$BUCKET' created successfully"
  else
    echo "  ✗ Failed to create bucket '$BUCKET' (HTTP $HTTP_CODE)"
    echo "    Response: $(echo "$CREATE_RESPONSE" | head -n-1)"
  fi
done

echo "=== Bucket initialization complete ==="
