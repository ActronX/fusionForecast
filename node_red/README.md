# Solar Consumer Cascade (Node-RED)

The custom Node-RED node `solar-consumer` controls a three-stage consumer cascade (C1-C3) from battery SoC and PV forecast data. The evaluation logic is implemented in [`lib/evaluate.js`](node-red-contrib-solar-consumer/lib/evaluate.js) and configured in the node editor, with defaults from [`lib/defaults.js`](node-red-contrib-solar-consumer/lib/defaults.js).

## What it does

- Reads only from InfluxDB (`type_soc` + `type_forecast`). It never writes to InfluxDB.
- Derives the usable PV window from forecast bins where PV exceeds the household base load.
- Calculates the projected energy surplus, dynamic reserve and the 95 % charge target.
- Evaluates C1 -> C2 -> C3 in order. A failed enabled stage blocks all later stages; disabled slots are pass-through.
- Allows battery support only within `max_battery_support_w`; otherwise rejects with `GRID_IMPORT_REQUIRED`.
- Keeps a crash/hysteresis lock in Node-RED context and fails safe (all relays OFF) on unexpected errors.

## Installation

The node is installed into the Docker image by `docker/Dockerfile.nodered`:

```bash
docker compose -f docker/docker-compose.yml build nodered
docker compose -f docker/docker-compose.yml up -d nodered
```

For a standalone Node-RED installation:

```bash
cd ~/.node-red
npm install /path/to/node-red-contrib-solar-consumer
```

Restart Node-RED afterwards. The node appears in the **solar** category as **Solar Consumer Cascade**.

## Flow wiring

```text
[Inject] -> [InfluxDB in (read-only)] -> [Solar Consumer Cascade]
                                             |-> output 1: C1 relay (payload Boolean)
                                             |-> output 2: C2 relay (payload Boolean)
                                             |-> output 3: C3 relay (payload Boolean)
                                             |-> output 4: debug_calc

output 1 -> Relay C1
output 2 -> Relay C2
output 3 -> Relay C3
output 4 -> Debug (calculation details)
```

The minimal example is [`example-flow.json`](example-flow.json). Import it in Node-RED via **Menu > Import** (or **Import > Clipboard**) to get the wiring above.

Docker seeds this example as `/data/flows.json` when the `nodered-data` named volume is created on first start. An existing volume is not overwritten; in that case import `example-flow.json` manually or recreate the volume deliberately.

## Configuration

All values are configured in the node editor. The 24-hour profiles are indexed by **UTC hour** (00-23). Defaults are defined in [`lib/defaults.js`](node-red-contrib-solar-consumer/lib/defaults.js).

| Field | Unit | Default | Description |
|---|---:|---:|---|
| `min_soc` | % | 20 | Low-battery cutoff; `min_soc + soc_hysteresis` must be <= 100 |
| `soc_hysteresis` | % | 15 | Recovery buffer above `min_soc` |
| `min_runtime_minutes` | min | 30 | Runtime used for the cycle-cost calculation |
| `battery_capacity_kwh` | kWh | 4.5 | Battery capacity |
| `reserve_kwh` | kWh | 2 | Dynamic reserve base |
| `battery_efficiency` | 0-1 | 0.85 | Charging efficiency |
| `allow_battery_support` | bool | true | Allow battery discharge for consumers above base load |
| `max_battery_support_w` | W | 2000 | Maximum battery support power |
| `base_load_w` | W | 24-value array | Household base load per UTC hour |
| `consumers.consumer1` | W | enabled, 24-value array | Cascade stage 1 |
| `consumers.consumer2` | W | enabled, 24 x 1800 | Cascade stage 2 |
| `consumers.consumer3` | W | enabled, 24 x 800 | Cascade stage 3 |

`base_load_w` may also be a single number in a `msg.config` override; the editor stores a 24-value array.

### Optional overrides

- `msg.config.consumer` deep-merges over the editor configuration for one message.
- `msg.reset = true` clears `lowBatLock`, `consumerStates` and `lastState` before evaluating.

## Input contract

`msg.payload` must be an array of InfluxDB rows:

| `_field` | Meaning |
|---|---|
| `type_soc` | Battery SoC in %, newest row |
| `type_forecast` | PV power in W per 15-minute UTC bin |

`type_production` is not part of the contract and is ignored. If the SoC is missing or outside 0-100 %, the node outputs `ERROR_NO_DATA` / `SOC_MISSING` or `SOC_INVALID` and keeps all relays OFF.

## Output contract

- Output 1 — C1 relay: `msg.payload` is a Boolean (`true = ON`, `false = OFF`), `msg.topic = "solar-consumer/consumer1"`.
- Output 2 — C2 relay: same contract with `msg.topic = "solar-consumer/consumer2"`.
- Output 3 — C3 relay: same contract with `msg.topic = "solar-consumer/consumer3"`.
- Output 4 — diagnostic message with `msg.payload = debug_calc` (full calculation details, including `debug_calc.consumers[id]`), `msg.topic = "solar-consumer/debug"`.

The consumer outputs carry only `payload` and `topic`; they do not include `consumer_states`, `debug_calc` or `influx_raw`. On a configuration error all three consumer outputs emit `false` and the debug output emits `{state: "CONFIG_ERROR", error}`.

**Breaking change:** version 3.0.0 emits one Boolean per consumer output instead of the former combined message (`msg.payload` = cascade level `0..3`, relay bits in `msg.consumer_states`). Existing flows must connect each consumer output directly to its relay; `msg.consumer_states` is no longer available on consumer outputs.

## Status

| Color | Meaning |
|---|---|
| Red | Low SoC or current PV at/below base load |
| Grey | Hysteresis lock, PV window unavailable/before/after, no current forecast point, no consumers enabled |
| Yellow | PV window too short, all OFF with a rejection |
| Blue | At least one relay ON and battery support active |
| Green | At least one relay ON without battery support |

Shape is `dot` when at least one relay is ON, otherwise `ring`.

## Read-only InfluxDB query

The read-only Flux query and the InfluxDB server configuration are documented in the [node README](node-red-contrib-solar-consumer/README.md#influxdb-query-and-node-configuration).

It reads SoC from `sonnen_live` / `pv` / `USOC` (last 30 minutes) and the forecast from `fusionForecastData` / `dwd` / `global_tilted_irradianceN` (raw 15-minute bins). There is no `type_production` stream and no InfluxDB write.

## Tests

```bash
# Unit + integration
docker compose -f docker/docker-compose.test.yml build tests
docker compose -f docker/docker-compose.test.yml up --abort-on-container-exit --exit-code-from tests tests

# E2E with isolated test InfluxDB
docker compose -f docker/docker-compose.test.yml up --build --abort-on-container-exit --exit-code-from e2e e2e
docker compose -f docker/docker-compose.test.yml down -v

# Production image
docker compose -f docker/docker-compose.yml build nodered
```

Local tests (Node.js only):

```bash
cd node_red/node-red-contrib-solar-consumer
npm install
npm test
```
