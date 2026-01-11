# Smart Solar Consumer Control (Node-RED)

## Overview

This Node-RED flow implements an intelligent control logic for high-power electrical consumers (e.g., heaters, pumps, EV chargers).

Instead of simple threshold switching, it calculates the **projected solar energy surplus** for the next 24 hours. It switches the consumer **ON** only if the surplus is sufficient to cover the runtime costs without draining the home battery below a reserved level.

It includes advanced protection features:
* **Hysteresis:** Prevents rapid toggling ("flip-flopping") by requiring a specific charge level recovery.
* **Safety Guard:** Prevents operation if forecast data is incomplete or outdated.
* **Battery Protection:** Hard cutoff when SoC is critically low.

---

## Wiring / Node Connections

To make this logic work, connect your Node-RED nodes in the following linear sequence:

![Wiring](Wiring.jpg)

`[Inject] --> [Template] --> [InfluxDB] --> [Function] --> [Output]`



### 1. Inject Node (Trigger)
* **Setting:** Set an interval (e.g., every 15 minutes).
* **Purpose:** Starts the process regularly.

### 2. Template Node (Configuration)
* **Setting:** Set "Property" to `msg.config`.
* **Format:** Select "JSON".
* **Content:** Paste your configuration JSON here (e.g., `{"consumer": {"min_soc": 10, ...}}`).
* **Purpose:** Loads the static settings into the message **before** the query runs.

### 3. InfluxDB Node (Data Fetch)
* **Setting:** Use the specific Flux query provided in the documentation.
* **Purpose:** Fetches the SoC and Forecast data.
* **Note:** The node writes the result to `msg.payload`, but it preserves the `msg.config` attached in the previous step.

### 4. Function Node (The Logic)
* **Setting:** Paste the JavaScript code provided for this project.
* **Purpose:** Performs the calculation, checks safety rules, and decides the state.

### 5. Output Nodes (Action)
* **Debug Node:** Connect `msg.debug_calc` to see the calculation details in the sidebar.
* **MQTT / HTTP / Switch:** Connect `msg.payload` to the node that actually controls your physical device (0=ON, 1=OFF).

---

## Architecture

### 1. Data Source (InfluxDB)
The flow expects an input message containing an array of objects from an InfluxDB Flux query.
* **Aggregation:** Data should be aggregated into 15-minute windows to optimize performance.
* **Required Fields:**
    * "type_soc": Current Battery State of Charge (%).
    * "type_forecast": Solar Irradiance Forecast (W).

### 2. Configuration (Template Node)
Configuration is passed via `msg.config.consumer`.

| Parameter | Description | Unit |
| :--- | :--- | :--- |
| min_soc | Critical battery level for emergency cutoff. | % |
| soc_hysteresis | Recovery buffer above min_soc to prevents toggling. | % |
| battery_capacity_kwh | Total capacity of the home battery. | kWh |
| reserve_kwh | Safety buffer to ensure battery reaches 100% even if the forecast was too optimistic. | kWh |
| base_load_w | Average house consumption to subtract from solar gain. | W |
| consumer_power_w | Power consumption of the controlled device. | W |
| min_runtime_minutes | Minimum runtime used to calculate "Cycle Cost". | min |
| forecast_conversion_factor | Factor to convert Irradiance to AC Watts. | - |

---

## Logic & State Machine

The logic node processes the data in the following order:

### A. Data Validation (Strict)
1.  **Configuration Check:** Validates that all configuration parameters are numbers >= 0. If any value is invalid, the script **aborts immediately** with a "Config Error".
2.  **Payload Check:** Checks if the InfluxDB payload contains valid SoC and Forecast data.

### B. Prediction Calculation
1.  Iterates through the forecast list (up to 24h).
2.  Calculates **Solar Yield** - **Base Load** = **Net Surplus**.
3.  Determines the **Surplus kWh** available above the target (Battery Capacity + Reserve).
4.  Compares this against the **Cycle Cost** (Energy needed to run the device for `min_runtime_minutes`).

### C. Safety & Decision Rules
The final switch state is determined by this priority list:

1.  **Low Battery Cutoff (Color: RED):**
    * If SoC <= min_soc, the device is forced **OFF**.
2.  **Hysteresis Recovery (Color: BLUE):**
    * If the device was previously **OFF** and the battery is charging, it remains **OFF** until SoC >= (min_soc + soc_hysteresis).
3.  **Safety Guard (Color: BLACK):**
    * If **remaining usable sun hours** (where generation > base load) are less than **0.5 hours**, the device is forced **OFF**.
4.  **Surplus Decision (Color: GREEN):**
    *   **Primary Check:** Is the predicted *Total Daily Surplus* >= Cycle Cost?
    *   **Secondary Check (Cloud Buffer Logic):** Even if daily surplus is high, we check if we can run *NOW* without crashing the battery.
        *   **Risk:** If we turn ON now, but the sun is weak and the battery is low, we might hit `min_soc` before the cycle finishes.
        *   **Rule:** We calculate a **SAFE_BUFFER_SOC** (Min SOC + Hysteresis OR Cycle Cost, whichever is higher).
        *   If **Battery < SAFE_BUFFER_SOC** AND **Current Forecast < Base Load**, we force **OFF** (Waiting for Sun/Battery).
        *   Otherwise (Battery > SAFE_BUFFER_SOC OR Current Forecast > Base Load), we switch **ON**.
5.  **No Surplus (Color: YELLOW):**
    *   If the Total Daily Surplus is insufficient, the device is switched **OFF**.

---


### Examples: How `SAFE_BUFFER_SOC` is calculated
The buffer's main goal is to ensure the battery can **survive at least one full operation cycle** (min_runtime) without falling below the critical limit. We take the larger of "Cycle Cost" or "Hysteresis".

**Formula:** `SAFE_BUFFER_SOC = min_soc + MAX(soc_hysteresis, required_cycle_soc)`

#### Case A: Hysteresis is dominant (Small Load)
*   `min_soc` = 30%
*   `soc_hysteresis` = 20%
*   `cycle_cost` = 1.5 kWh (approx 15% of a 10kWh battery)
*   **Result:** 30% + MAX(20%, 15%) = 30% + 20% = **50%**

#### Case B: Cycle Cost is dominant (Heavy Consumer)
*   `min_soc` = 30%
*   `soc_hysteresis` = 10%
*   `cycle_cost` = 3.0 kWh (30% of a 10kWh battery)
*   **Result:** 30% + MAX(10%, 30%) = 30% + 30% = **60%**

---

## Status Codes (Visual Indicators)

The Node-RED status dot provides immediate visual feedback:

| Color | Status | Meaning |
| :--- | :--- | :--- |
| **RED** | CRITICAL | **Hard Cutoff.** Battery is below min_soc. |
| **BLUE** | WAITING | **Charging.** Battery is above minimum but has not reached the recovery target (+20%) yet. |
| **BLACK**| SAFETY | **Data Issue.** Remaining usable sun too short (< 0.5h). |
| **GREEN**| ON | **Active.** Sufficient solar surplus calculated. |
| **YELLOW**| OFF | **Low Surplus.** System is healthy, but not enough sun to run the device. |
| **GREY (Dot)** | ERROR | **No Data.** InfluxDB query returned no valid payload. |
| **GREY (Ring)** | CONFIG | **Config Error.** Invalid configuration values detected (not a number or negative). Script execution aborted. |

---

## Outputs

### msg.payload (Switch Signal)
The node uses **Active Low** logic (or specific logic defined in your setup):
* 0 = **ON** (Enable Consumer)
* 1 = **OFF** (Disable Consumer)

### msg.debug_calc (Debugging)
A JSON object containing the full calculation details for the sidebar:
```json
{
  "state": "CALCULATED",
  "soc": 45.5,
  "solar_gain_kwh": 5.2,
  "surplus_kwh": 1.5,
  "required_cycle_kwh": 1.2,
  "required_cycle_soc_pct": 12.0,
  "remaining_sun_hours": 12.5
}