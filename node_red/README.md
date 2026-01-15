# Smart Solar Consumer Control (Node-RED)

## Overview

This Node-RED flow implements an intelligent control logic for high-power electrical consumers (e.g., heaters, pumps, EV chargers).

Instead of simple threshold switching, it calculates the **projected solar energy surplus** for the next 24 hours. It switches the consumer **ON** only if the surplus is sufficient to cover the runtime costs without draining the home battery below a reserved level.

It includes advanced protection features:
* **Hysteresis:** Prevents rapid toggling ("flip-flopping") by requiring a specific charge level recovery.
* **Dynamic Reserve:** Maintains a high safety buffer when battery is low, but reduces it when battery is full to maximize capacity usage.
* **Safety Guard:** Prevents operation if forecast data is incomplete or outdated.
* **Real-Time Forecast Correction:** Dynamically adjusts the forecast curve ("Damping Factor") based on the actual solar performance since sunrise. If the day is cloudier/sunnier than predicted, the future forecast is scaled accordingly.
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
* **Content:** Paste your configuration JSON here (e.g., `{"consumer": {"min_soc": 10, "battery_efficiency": 0.90, ...}}`).
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
* **Required Fields & Time Ranges:**
    
    | Field Name | Description | Needed Time Range |
    |Str |Str |Str |
    | :--- | :--- | :--- |
    | `type_soc` | Battery State of Charge (%) | Current Value (last 1h) |
    | `type_forecast` | Solar Power Forecast (W) | **History** (since sunrise) AND **Future** (next 24-48h) |
    | `type_production` | Actual Solar Production (W) | **History** (last 2h sufficient) |

    > **Important:** The logic matches `type_production` and `type_forecast` timestamps to calculate the Damping Factor. Ensure both have the same time grid (e.g. 15-min aligned) and availability for the *past* hours of the current day.

### 2. Configuration (Template Node)
Configuration is passed via `msg.config.consumer`.

| Parameter | Description | Unit |
| :--- | :--- | :--- |
| min_soc | Critical battery level for emergency cutoff. | % |
| soc_hysteresis | Recovery buffer above min_soc to prevents toggling. | % |
| battery_capacity_kwh | Total capacity of the home battery. | kWh |
| pv_peak_power_w | Maximum physical power output of the PV system. Used for sanity checks and limits. | W |
| reserve_kwh | Safety buffer to ensure battery reaches 100%. Scaled dynamically (100% Reserve at empty battery, down to **10% (Floor)** at full battery). | kWh |
| base_load_w | Average house consumption to subtract from solar gain. | W |
| consumer_power_w | Power consumption of the controlled device. | W |
| min_runtime_minutes | Minimum runtime used to calculate "Cycle Cost". | min |
| forecast_conversion_factor | Factor to convert Irradiance to AC Watts. | - |
| battery_efficiency | Efficiency of the battery charging process (e.g. 0.90 for 90%). | - |
| use_damping_factor | Enable/Disable real-time forecast correction (true/false). | - |

---

## Logic & State Machine

The logic node processes the data in the following order:

### A. Data Validation (Strict)
1.  **Configuration Check:** Validates that all configuration parameters are numbers >= 0. If any value is invalid, the script **aborts immediately** with a "Config Error".
2.  **Payload Check:** Checks if the InfluxDB payload contains valid SoC and Forecast data.

### B. Prediction Calculation
1.  Iterates through the forecast list (up to 24h).
2.  Calculates **Solar Yield** - **Base Load** = **Net Surplus**.
3.  Determines the **Surplus kWh** available above the target (Battery Capacity + Reserve, adjusted for charging efficiency).
4.  Compares this against the **Cycle Cost** (Energy needed to run the device for `min_runtime_minutes`).

### C. Dynamic Reserve Logic (Safety Curve)
To balance safety (don't run out of battery) with efficiency (maximize self-consumption), the `reserve_kwh` is **not static**.

*   **Concept:**
    *   **Low SoC:** We keep 100% of the configured reserve (Safety First).
    *   **High SoC:** We reduce the reserve significantly (Trusting the full battery).
    *   **Curve:** "Quadratic Safe" -> `Factor = 0.1 + 0.9 * (1 - (SoC_decimal ^ 2))`

*   **Examples:**
    *   **0% SoC:** 100% of Configured Reserve.
    *   **50% SoC:** ~78% of Configured Reserve.
    *   **80% SoC:** ~42% of Configured Reserve.
    *   **100% SoC:** 10% of Configured Reserve (Minimum Floor).

### D. Real-Time Correction (Damping Factor)
Before making decisions, the script compares the **Forecast** vs. **Actual Production** from recent history (e.g. last 2 hours).

*   **Goal:** Detect if the day is persistently better or worse than predicted (e.g., unexpected fog or clear sky).
*   **Logic:**
    1.  Sum up all historical forecast values (last 2h).
    2.  Sum up all actual production values for the same timepoints.
    3.  Calculate `Factor = Sum(Actual) / Sum(Forecast)`.
    4.  **Clamp:** The factor is limited to a safety range (e.g., **0.75x** to **1.50x**) to prevent extreme distortion.
    5.  **Apply (Decaying Influence):** The damping factor is applied to the forecast curve with a **time-based decay** (Half-Life: 1h).
        *   **Why?** Weather anomalies (like a passing cloud or morning fog) are often temporary.
        *   **Concept:**
            *   **Short-Term (0-1h):** We trust our *local* 'Live-Correction' fully. If it's foggy now, it will likely be foggy in 30 mins.
            *   **Long-Term (2h+):** We trust the *global* 'Weather Forecast' again. The fog will likely clear up, and the original prediction becomes valid again.
        *   **Effect:**
            *   **Now (0h):** 100% Influence.
            *   **+1h:** 50% Influence.
            *   **+2h:** 25% Influence.
            *   **+4h:** ~6% Influence (Back to original Forecast).
*   **Threshold:** This correction is only applied if the accumulated forecast energy exceeds **3% of the PV Peak Power** (to avoid mathematical noise at dawn/dusk).
*   **Physical Limit:** The adjusted forecast value is capped at `pv_peak_power_w` to ensure values remain within realistic system limits.

### E. Safety & Decision Rules
The final switch state is determined by this priority list:

1.  **Low Battery Cutoff (Color: RED):**
    * If SoC <= min_soc, the device is forced **OFF**.
2.  **Hysteresis Recovery (Color: BLUE):
    * If the device was previously **OFF** and the battery is charging, it remains **OFF** until SoC >= (min_soc + soc_hysteresis).
3.  **Safety Guard (Color: BLACK):
    * If **remaining usable sun hours** (where generation > base load) are less than **2x min_runtime**, the device is forced **OFF**.
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
| **BLACK**| SAFETY | **Data Issue.** Remaining usable sun too short (< 2x min_runtime). |
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
  "config_used": { ... },
  "low_bat_lock_active": false,
  "hysteresis_recovery_target": 50,
  "current_forecast_w": 1500,
  "damping_factor": "1.02",
  "past_analysis": "Matches:4 | Prod:1200 vs Fcst:1176 (OK)",
  "solar_gain_kwh": 5.2,
  "base_loss_kwh": 1.1,
  "surplus_kwh": 1.5,
  "required_cycle_kwh": 2.1,
  "required_cycle_soc_pct": 12.0,
  "cutoff_reason": "",
  "remaining_sun_hours": 6.5,
  "dynamic_reserve_kwh": 0.5
}