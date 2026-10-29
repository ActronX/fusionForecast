# Node-RED Integration Guide

Node-RED runs in the FusionForecast Docker environment and loads the local custom node `node-red-contrib-solar-consumer` from this repository.

## 1. Docker setup

- **Base image:** `nodered/node-red:latest`
- **Port:** `1880`
- **URL:** [http://localhost:1880](http://localhost:1880)

`docker/Dockerfile.nodered` installs:

- `node-red-contrib-influxdb`
- the local `node_red/node-red-contrib-solar-consumer` package (packed and installed into the Node-RED image)

The image seeds [`node_red/example-flow.json`](../node_red/example-flow.json) as `/data/flows.json`. Docker uses the named volume `nodered-data` for `/data`, so flows and credentials persist and no runtime files are written into the repository.

On the first start of a fresh volume, Docker initializes `/data` with the image contents, including the seeded example flow. An existing `nodered-data` volume is never overwritten: import `example-flow.json` manually in that case, or recreate the volume deliberately.

Rebuild after changing the custom node:

```bash
docker compose -f docker/docker-compose.yml build nodered
docker compose -f docker/docker-compose.yml up -d nodered
```

## 2. Flow

The example flow in `node_red/example-flow.json` is:

```text
Inject -> InfluxDB in (read-only) -> Solar Consumer Cascade
                                          |-> output 1: C1 relay (Boolean payload)
                                          |-> output 2: C2 relay (Boolean payload)
                                          |-> output 3: C3 relay (Boolean payload)
                                          |-> output 4: debug_calc
```

Connect each consumer output directly to its relay. `msg.payload` is a Boolean (`true = ON`, `false = OFF`) and `msg.topic` identifies the consumer (`solar-consumer/consumer1..3`). The diagnostic output carries `msg.payload = debug_calc`. On a configuration error all consumer outputs emit `false` and the debug output reports `CONFIG_ERROR`.

The read-only Flux query and the InfluxDB node configuration are documented in the [node README](../node_red/node-red-contrib-solar-consumer/README.md#influxdb-query-and-node-configuration). It reads:

- `sonnen_live` / `pv` / `USOC` (last 30 minutes)
- `fusionForecastData` / `dwd` / `global_tilted_irradianceN` (raw 15-minute bins)

It does not query `type_production` and does not write to InfluxDB.

## 3. InfluxDB credentials

Node-RED stores credentials encrypted in `/data/flows_cred.json` inside the `nodered-data` volume. Set them once in the UI:

1. Open `http://localhost:1880`.
2. Double-click the InfluxDB node.
3. Edit the server configuration.
4. Enter:
   - **URL:** `http://influxdb:8086`
   - **Token:** your `INFLUXDB_TOKEN`
   - **Organization:** `fusionforecast` (or your `INFLUXDB_ORG`)
5. Update, Done, Deploy.

The container uses `CREDENTIAL_SECRET` to decrypt the credentials consistently across restarts. The Docker entrypoint attempts a `${INFLUXDB_TOKEN}` substitution in `flows.json` and ignores a failed write (for example on a read-only test mount) so Node-RED still starts.

## 4. Tests

```bash
# Unit, integration and static read-only checks
docker compose -f docker/docker-compose.test.yml build tests
docker compose -f docker/docker-compose.test.yml up --abort-on-container-exit --exit-code-from tests tests

# E2E with isolated test InfluxDB + Node-RED
docker compose -f docker/docker-compose.test.yml up --build --abort-on-container-exit --exit-code-from e2e e2e
docker compose -f docker/docker-compose.test.yml down -v
```

The test stack uses a separate InfluxDB and a test-only `flows.json`; it does not touch production data or credentials.
