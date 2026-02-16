# node-red-contrib-solar-consumer

Intelligent load switching based on **battery SoC**, **solar forecast**, and **energy surplus calculation** with optional real-time forecast damping.

## Installation

```bash
cd ~/.node-red
npm install /path/to/node-red-contrib-solar-consumer
```

Or directly in your Node-RED project directory:

```bash
npm install ./node-red-contrib-solar-consumer
```

Restart Node-RED afterwards.

## Configuration

All parameters are configured directly in the node editor:

| Parameter | Unit | Default | Description |
|---|---|---|---|
| Min SoC | % | 20 | Minimum battery state of charge |
| Hysteresis | % | 20 | SoC threshold above Min SoC to release lock |
| Capacity | kWh | 5.0 | Battery capacity |
| Efficiency | 0–1 | 0.85 | Charge/discharge efficiency |
| Reserve | kWh | 2.0 | Energy reserve for overnight |
| PV Peak | W | 5200 | Maximum PV system power |
| Consumer Power | W | 1000 | Power consumption of the load |
| Min Runtime | min | 30 | Minimum runtime per cycle |
| Base Load | W | 300 | Household base load |
| Damping Factor | bool | false | Real-time forecast correction |
| Battery Support | bool | true | Allow battery discharge for the consumer |

## Input

`msg.payload` — Array of InfluxDB data points containing:
- `_field: "type_soc"` — current battery SoC
- `_field: "type_forecast"` — PV forecast (15-min intervals)
- `_field: "type_production"` — actual production (for damping)

## Output

- `msg.payload` — `0` (ON) or `1` (OFF)
- `msg.debug_calc` — detailed debug object with all calculation values

## Node Status

| Color | Meaning |
|---|---|
| 🟢 Green | Consumer ON |
| 🟡 Yellow | Insufficient surplus / battery charging |
| 🔴 Red | Battery critical / error |
| 🔵 Blue | Hysteresis lock active |
| ⚫ Black | Safety guard (not enough sun hours left) |
| ⚪ Grey | No data / config error |
