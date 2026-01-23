# InfluxDB Login Guide for FusionForecast

## Overview

InfluxDB in your FusionForecast Docker deployment provides a web-based user interface (UI) running on **port 8086**. This guide explains how to access and authenticate to InfluxDB for managing your PV energy forecasting data.

---

## 1. Accessing the InfluxDB Web Interface

### Step 1: Open Your Browser

Once your Docker containers are running, open your web browser and navigate to:

```
http://localhost:8086
```

**Note:** If you're accessing InfluxDB from a different machine on your network, replace `localhost` with your server's IP address:

```
http://<your-server-ip>:8086
```

Example: `http://192.168.1.100:8086`

### Step 2: Initial Setup (First Time Only)

If InfluxDB has never been initialized before, you'll see a setup screen. This typically occurs before your docker-compose auto-initialization completes. In the FusionForecast Docker deployment, this is **automatically handled** by the initialization environment variables, so you may skip to the login section.

---

## 2. Logging In to InfluxDB

### Authentication Credentials

Your FusionForecast setup uses the following credentials (configured in your `.env` file):

| Item | Environment Variable | Default Value |
|------|---------------------|----------------|
| **Organization** | `DOCKER_INFLUXDB_INIT_ORG` | `fusionforecast` |
| **Bucket (for setup)** | `DOCKER_INFLUXDB_INIT_BUCKET` | `energy_meter` |
| **Admin User** | `DOCKER_INFLUXDB_INIT_USERNAME` | (Defined in `.env.example`) |
| **Admin Password** | `DOCKER_INFLUXDB_INIT_PASSWORD` | (Defined in `.env.example`) |
| **Admin Token** | `DOCKER_INFLUXDB_INIT_ADMIN_TOKEN` | (Predefined in `.env.example`) |

### Login Process

1. **Enter Your Credentials:**
   - **Username:** Use the admin username from your `.env` file (typically `admin`)
   - **Password:** Use the admin password from your `.env` file

2. **Click "Sign In"**

3. **You're Now Authenticated**

You'll be redirected to the InfluxDB dashboard where you can manage buckets, view data, and create API tokens.

---

## 3. Finding Your API Token

Your FusionForecast application communicates with InfluxDB using an **API Token** for authentication. This token is configured via the `INFLUXDB_TOKEN` environment variable.

### Option A: Using the Web UI

1. Log into InfluxDB at `http://localhost:8086`
2. Click **Load Data** in the left sidebar
3. Select **API Tokens**
4. Your tokens will be listed here, including the initial **Operator Token** created during setup

### Option B: Retrieving the Token from `.env`

The token used by FusionForecast is stored in your `.env` file:

```bash
cat .env | grep INFLUXDB_TOKEN
```

Output example:
```
INFLUXDB_TOKEN=s3cr3tT0k3n1234567890abcdefghijklmnop
```

### Option C: Accessing the Token from the Docker Container

If you need to retrieve the admin token from within the running container:

```bash
docker exec influxdb influx auth list --org fusionforecast
```

This command lists all authentication tokens in your organization.

---

## 4. Verifying InfluxDB Connection

### Using Curl to Test the Connection

Test if your InfluxDB instance is accessible and your token is valid:

```bash
# Set environment variables
export INFLUX_HOST="http://localhost:8086"
export INFLUX_TOKEN="YOUR_TOKEN_FROM_.env"
export INFLUX_ORG="fusionforecast"

# Test the connection
curl -X GET "$INFLUX_HOST/api/v2/buckets" \
  -H "Authorization: Token $INFLUX_TOKEN" \
  -H "Content-Type: application/json"
```

**Expected Response:** A JSON list of your buckets (energy_meter, energy_data, etc.)

### Using the InfluxDB CLI Inside the Container

```bash
# Access the InfluxDB CLI within the container
docker exec -it influxdb influx auth list

# List all buckets
docker exec -it influxdb influx bucket list
```

---

## 5. Understanding the Data Structure

Once logged in, you'll see the following key buckets in FusionForecast:

| Bucket | Purpose | Data Type |
|--------|---------|-----------|
| `energy_meter` | **Live PV production data** | Real-time sensor readings (Watts) |
| `energy_data` | **Historical PV data** | Training data for the Prophet model |
| `forecasts` | **Generated PV forecasts** | Predicted power output (24-48 hours ahead) |
| `nowcasts` | **Corrected short-term forecasts** | 15-minute updated predictions with live data |

### Exploring Data in the UI

1. Click **Explore** in the sidebar
2. Select a **bucket** (e.g., `energy_meter`)
3. Choose a **measurement** (e.g., `energy_meter`)
4. Select a **field** (e.g., `power_produced` or `production`)
5. Click **Submit** to visualize your data

---

## 6. Common Issues & Troubleshooting

### Issue 1: "Connection Refused" Error

**Symptom:** `curl: (7) Failed to connect to localhost port 8086`

**Solution:**
- Ensure Docker containers are running: `docker-compose ps`
- Check if InfluxDB container is active: `docker-compose logs influxdb`
- Restart InfluxDB: `docker-compose restart influxdb`

### Issue 2: "Unauthorized" (401) Error

**Symptom:** Curl request returns `401 Unauthorized`

**Solution:**
- Verify your token is correct in the `Authorization` header
- Use the format: `Authorization: Token YOUR_TOKEN` (not `Bearer`)
- Check that the token hasn't expired or been revoked

### Issue 3: Cannot Log In to Web UI

**Symptom:** Invalid username or password error

**Solution:**
- Verify credentials in your `.env` file
- Ensure the environment variables match: `DOCKER_INFLUXDB_INIT_USERNAME` and `DOCKER_INFLUXDB_INIT_PASSWORD`
- If the container was already initialized, you must use the original credentials (changing `.env` won't affect an existing database)
- To reset: `docker-compose down -v` (⚠️ this deletes all data) and `docker-compose up -d`

### Issue 4: "Bucket Does Not Exist"

**Symptom:** Error when trying to write data to a bucket

**Solution:**
- Verify bucket names: `docker exec influxdb influx bucket list`
- Ensure your token has write permissions to that bucket
- Check the exact bucket name in your settings.toml or web UI

---

## 7. Quick Reference: Token-Based Requests

### Write Data to InfluxDB

```bash
curl -X POST "http://localhost:8086/api/v2/write?org=fusionforecast&bucket=energy_meter&precision=s" \
  -H "Authorization: Token YOUR_TOKEN" \
  --data-raw "energy_meter production=1250.5"
```

### Query Data from InfluxDB

```bash
curl -X POST "http://localhost:8086/api/v2/query?org=fusionforecast" \
  -H "Authorization: Token YOUR_TOKEN" \
  -H "Content-Type: application/vnd.flux" \
  --data 'from(bucket:"energy_meter") |> range(start: -24h)'
```

### Create a New API Token (Via CLI)

```bash
docker exec influxdb influx auth create \
  --org fusionforecast \
  --description "FusionForecast Read Token" \
  --read-bucket energy_meter,energy_data,forecasts,nowcasts
```

### Option C: Change InfluxDB Bind Address (Not Recommended)

In your `docker-compose.yml`, expose InfluxDB to all interfaces:

```yaml
ports:
  - "0.0.0.0:8086:8086"  # Accessible from any IP on port 8086
```

⚠️ **Warning:** This exposes InfluxDB to your entire network. Always use authentication and a firewall.

---

## 8. Additional Resources

- **InfluxDB Official Docs:** https://docs.influxdata.com/influxdb/v2/
- **API Token Reference:** https://docs.influxdata.com/influxdb/v2/admin/tokens/
- **FusionForecast GitHub:** https://github.com/ActronX/fusionForecast
- **Docker Compose Setup:** https://docs.influxdata.com/influxdb/v2/install/use-docker-compose/

---

## Summary

| Task | Command / URL |
|------|---------------|
| **Access Web UI** | `http://localhost:8086` |
| **Get API Token** | Check `.env` or UI: Load Data > API Tokens |
| **Test Connection** | `curl -H "Authorization: Token $TOKEN" http://localhost:8086/api/v2/buckets` |
| **View Logs** | `docker-compose logs -f influxdb` |
| **Access CLI** | `docker exec -it influxdb influx` |
| **List Buckets** | `docker exec influxdb influx bucket list` |
| **List Tokens** | `docker exec influxdb influx auth list` |

This guide should help you successfully authenticate to InfluxDB in your FusionForecast environment. For specific issues related to your setup, refer to the troubleshooting section or check the Docker container logs.
