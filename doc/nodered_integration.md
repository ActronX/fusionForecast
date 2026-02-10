# Node-RED Integration Guide

This guide explains how Node-RED is integrated into the FusionForecast Docker environment, how to manage credentials securely, and how to handle version control.

## 1. Docker Setup

Node-RED runs as a custom container built from `Dockerfile.nodered` within the `docker-compose.yml`.

*   **Base Image:** `nodered/node-red:latest`
*   **Port:** `1880` (Mapped to host)
*   **URL:** [http://localhost:1880](http://localhost:1880)

### Volume Mapping
The entire `node_red/` directory is mounted from the host to persist flows and configuration:

```yaml
volumes:
  - ../node_red:/data
```

This mount strategy prevents `EBUSY` errors on Windows when Node-RED tries to save changes, and ensures all flows, credentials, and installed nodes persist across container restarts.

### Environment Variables
The container receives InfluxDB connection details from your `.env` file:

```yaml
environment:
  - INFLUXDB_URL=http://influxdb:8086
  - INFLUXDB_TOKEN=${INFLUXDB_TOKEN}
  - INFLUXDB_ORG=${INFLUXDB_ORG:-fusionforecast}
  - CREDENTIAL_SECRET=${NODERED_CREDENTIAL_SECRET:-fusionForecastSecret}
```

## 2. Credential Management

Node-RED stores encrypted credentials separately from flow logic to enable sharing flows without exposing secrets.

### Initial Setup Process

**When you first start Node-RED**, the credentials are not automatically configured. You must manually enter them in the UI:

1.  Open Node-RED at `http://localhost:1880`
2.  Double-click any InfluxDB node in the flow
3.  Click the pencil icon next to the Server configuration
4.  Enter the following values:
    *   **URL:** `http://influxdb:8086`
    *   **Token:** Your `INFLUXDB_TOKEN` from the `.env` file
    *   **Organization:** Your `INFLUXDB_ORG` from the `.env` file (default: `fusionforecast`)
5.  Click **Update**, then **Done**
6.  Click **Deploy** to save the configuration

<img src="node_red_credentials.jpg" width="50%">

After this initial setup, Node-RED will encrypt and save the credentials to `flows_cred.json`. The `CREDENTIAL_SECRET` environment variable ensures consistent encryption/decryption across container restarts.

### How It Works
*   The Dockerfile patches the `settings.js` to use `process.env.CREDENTIAL_SECRET` for credential encryption
*   On startup, the entrypoint script attempts to substitute `${INFLUXDB_TOKEN}` placeholders in `flows.json` (if present)
*   However, InfluxDB node credentials are stored in the separate `flows_cred.json` file and **must be configured manually** via the UI