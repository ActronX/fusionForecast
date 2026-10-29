# node-red-contrib-solar-consumer

A Node-RED custom node that evaluates a three-stage solar consumer cascade from battery SoC and PV forecast rows.

## Installation

```bash
cd ~/.node-red
npm install /path/to/node-red-contrib-solar-consumer
```

Restart Node-RED. The node is available as **Solar Consumer Cascade** in the **solar** category.

## Configuration

Configure everything in the node editor. The 24-hour profiles use **UTC hours 00-23**. Defaults are defined in `lib/defaults.js`.

- **Battery:** min SoC 20 %, hysteresis 15 %, minimum runtime 30 min, capacity 4.5 kWh, reserve 2 kWh, efficiency 0.85.
- **Battery support:** enabled, `max_battery_support_w` 2000 W.
- **Base load:** `base_load_w` = `[300, 301, 302, 303, 304, 305, 306, 407, 508, 509, 1110, 1111, 612, 613, 614, 515, 616, 617, 518, 519, 420, 421, 322, 323]` (24 values, UTC hours 00-23).
- **Consumer 1:** enabled, `power_w` = `[600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 10, 11, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623]`.
- **Consumer 2:** enabled, `power_w` = 24 x 1800 W.
- **Consumer 3:** enabled, `power_w` = 24 x 800 W.

Optional per-message overrides:

- `msg.config.consumer` deep-merges over the editor config.
- `msg.reset = true` clears the lock and all relay states.

## InfluxDB query and node configuration

The `influxdb in` node is configured as follows:

- URL `http://influxdb:8086`, Org `fusionforecast`, Bucket `default` (configurable via the Compose environment).
- InfluxDB version `2.0`, no TLS, `rejectUnauthorized: false`, timeout `10 s`.

The query is strictly read-only: it contains no `to()` write call, no `type_production` stream and no SoC fallback. If the SoC is missing or invalid, the node emits `ERROR_NO_DATA` (`SOC_MISSING` / `SOC_INVALID`) and keeps all relays OFF.

```flux
// Flux query for the Node-RED "InfluxDB Query" node.
//
// Changes against the previous version (2026-10-21):
// 1. The production stream (bucket "sonnen_live", field "Production_W",
//    mapped to "type_production") was removed. The live PV meter was never
//    evaluated by the logic node, it only added rows that could invalidate the
//    input. Consequence: the actual PV output is no longer available for a
//    forecast/actual comparison - it would have to be queried separately.
//
// 2. The SoC stream is limited to the last 30 minutes (range(start: -30m)).
//    This window is the freshness limit of the whole system: the logic node
//    only checks that a usable SoC value is present, the query decides which
//    record is current. Because range() ends at now() by default, a record
//    with a timestamp in the future is excluded as well.
//
// 3. group() is placed before last() in the SoC stream, so that a source with
//    tags still delivers exactly one SoC record instead of one per tag
//    combination.
//
// 4. The forecast stream is NOT resampled any more. The previous version used
//    aggregateWindow(every: 15m, fn: mean, timeSrc: "_stop"). The source
//    (Open-Meteo Minutely15) already delivers exactly one value per 15-minute
//    interval, so the mean over the window was a no-op for the values and only
//    shifted every timestamp by 15 minutes (interval start -> interval end).
//    That shift made the forecast 15 minutes "older" than it is and moved the
//    PV window 15 minutes too far. The raw values are now passed through 1:1.
//    The source writes one series without tags, so there is exactly one value
//    per timestamp (verified in the raw dump of 2026-10-21).
//
// 5. The forecast stream is grouped by _time and reduced with last(), so that
//    exactly one value per timestamp reaches the logic node. If the source
//    ever delivers several series (e.g. by adding tags), the newest value wins
//    instead of the energy balance counting the bin twice. Without duplicates
//    this is a no-op.
//
// 6. Both streams are mapped explicitly to the same four columns
//    (_time, _measurement, _field, _value) and are grouped without a group key
//    before the union. The old "map(fn: (r) => ({r with _field: ...}))" kept all
//    tags of the source measurement, so the two streams had different schemas
//    and union() failed with a label error.
//
// 7. The result is sorted by _time, so the payload order is deterministic
//    (forecast ascending, SoC last).
//
// Result contract for the "Logik" function node:
// - _field == "type_soc":      state of charge in %, newest record of the last
//   30 minutes. If the source writes nothing in that window, no type_soc row
//   is delivered and the logic node switches all consumers OFF.
// - _field == "type_forecast": PV power in W per 15-minute bin, UTC, exactly
//   one value per timestamp. The bin timestamp is the start of the interval as
//   written by the source, so the bin labelled 10:00Z covers 10:00Z-10:15Z.
//   The logic node adds the 15 minutes itself when it derives the end of the
//   PV window.
//
// Note: the forecast is taken from the fusion forecast bucket without the
// integral field. The unit/magnitude of that series (global_tilted_irradianceN
// vs. real PV power) is still unverified and is not part of this change.
import "date"

// As a precaution, we query until tomorrow morning, but later cut off in the script
heute_start = date.truncate(t: now(), unit: 1d)
morgen_frueh = date.add(d: 24h, to: heute_start)

// 1) State of charge: newest record of the last 30 minutes.
soc_stream = from(bucket: "sonnen_live")
  |> range(start: -30m)
  |> filter(fn: (r) => r["_measurement"] == "pv")
  |> filter(fn: (r) => r["_field"] == "USOC")
  |> group()
  |> last()
  |> map(fn: (r) => ({_time: r._time, _measurement: "pv", _field: "type_soc", _value: float(v: r._value)}))

// 2) Forecast curve (without integral); raw 15-minute values, no resampling
forecast_stream = from(bucket: "fusionForecastData")
  |> range(start: heute_start, stop: morgen_frueh)
  |> filter(fn: (r) => r["_measurement"] == "dwd")
  |> filter(fn: (r) => r["_field"] == "global_tilted_irradianceN")
  // One value per timestamp. If the source ever delivers several series (e.g.
  // by adding tags), the newest one wins instead of being counted twice in
  // the energy balance of the logic node. Without duplicates this is a no-op.
  |> group(columns: ["_time"])
  |> last()
  |> group()
  |> map(fn: (r) => ({_time: r._time, _measurement: "dwd", _field: "type_forecast", _value: float(v: r._value)}))

// 3) Union all streams and keep only the relevant columns
union(tables: [soc_stream, forecast_stream])
  |> keep(columns: ["_time", "_field", "_value"])
  |> sort(columns: ["_time"])
```

The example flow [`example-flow.json`](../../example-flow.json) contains this query verbatim. Keep the README and the flow in sync when changing it.

Credentials (the InfluxDB token) are not stored in the flow or this repository. Set them in the Node-RED UI; Node-RED stores them encrypted in its `/data` volume. `CREDENTIAL_SECRET` must stay stable so the credentials can be decrypted after a restart.

## Input

`msg.payload` is an array of InfluxDB rows with `_time`, `_field` and `_value` (see the read-only query above):

- `type_soc` — newest SoC in %
- `type_forecast` — PV power in W per 15-minute UTC bin

Missing/invalid SoC produces `ERROR_NO_DATA` (`SOC_MISSING`/`SOC_INVALID`) and keeps all relays OFF. `type_production` is ignored.

## Output

- Output 1: consumer 1 relay. `msg.payload` is a Boolean (`true = ON`, `false = OFF`), `msg.topic = "solar-consumer/consumer1"`.
- Output 2: consumer 2 relay, same contract with `msg.topic = "solar-consumer/consumer2"`.
- Output 3: consumer 3 relay, same contract with `msg.topic = "solar-consumer/consumer3"`.
- Output 4: diagnostic message with `msg.payload = debug_calc` (calculation details, including `debug_calc.consumers[id]`) and `msg.topic = "solar-consumer/debug"`.

The consumer outputs contain only `payload` and `topic`. On a configuration error they emit `false` and the debug output emits `{state: "CONFIG_ERROR", error}`.

**Breaking change:** as of 3.0.0 each consumer output carries its own Boolean state. `msg.consumer_states`, `msg.debug_calc` and `msg.influx_raw` are no longer set on consumer outputs; connect each output directly to its relay.

## Status

| Color | Meaning |
|---|---|
| Red | Low SoC or PV at/below base load |
| Grey | Hysteresis lock, PV window unavailable/before/after, no current forecast point, no consumers enabled |
| Yellow | PV window too short, all OFF with rejection |
| Blue | Relay ON with battery support |
| Green | Relay ON without battery support |

`shape = dot` if at least one relay is ON, otherwise `ring`.

## Tests

```bash
npm install
npm test
npm run test:readonly
```

The E2E scripts require an InfluxDB and a running Node-RED; use the Docker test stack from the repository root:

```bash
docker compose -f docker/docker-compose.test.yml up --build --abort-on-container-exit --exit-code-from e2e e2e
```

## License

MIT
