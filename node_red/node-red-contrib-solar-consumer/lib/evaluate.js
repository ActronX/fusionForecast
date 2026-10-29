"use strict";

const CONSUMER_IDS = ["consumer1", "consumer2", "consumer3"];
const STATE_CONVENTION = 2;
const FORECAST_STEP_MS = 15 * 60 * 1000;
const TARGET_SOC_LIMIT = 0.95;

function evaluate(input) {
    const { config: conf, rows } = input;
    const contextState = input.contextState || {};
    const now = input.nowMs === undefined ? Date.now() : input.nowMs;

    const RECOVERY_SOC = Math.min(conf.min_soc + conf.soc_hysteresis, 100);

    const legacyStates = contextState.stateConvention !== STATE_CONVENTION;
    const toCurrentConvention = value =>
        (value === 0 || value === 1) ? (legacyStates ? (value === 0 ? 1 : 0) : value) : 0;

    let lastState = contextState.lastState;
    if (lastState === undefined) lastState = 0;
    else lastState = toCurrentConvention(lastState);

    const storedConsumerStates = contextState.consumerStates;
    const previousConsumerStates = {};
    CONSUMER_IDS.forEach((id, index) => {
        if (storedConsumerStates && (storedConsumerStates[id] === 0 || storedConsumerStates[id] === 1)) {
            previousConsumerStates[id] = toCurrentConvention(storedConsumerStates[id]);
        } else {
            previousConsumerStates[id] = index === 0 ? lastState : 0;
        }
    });

    let lowBatLock = contextState.lowBatLock;
    if (lowBatLock === undefined) lowBatLock = false;
    const lowBatLockAtStart = lowBatLock;

    let consumerStates = { consumer1: 0, consumer2: 0, consumer3: 0 };
    let statusFill = "grey";
    let statusText = "Idle";
    let debugInfo = {
        state: "INIT",
        consumers: {},
        available_cascade_level: 0,
        cumulative_active_power_w: 0
    };
    let error = null;

    function pointTimestamp(point) {
        return point.timestampMs;
    }

    function getBaseLoadAt(timestampMs) {
        return conf.base_load_w[new Date(timestampMs).getUTCHours()];
    }

    function isUsablePvPoint(point) {
        return point.watts > getBaseLoadAt(pointTimestamp(point));
    }

    function selectCurrentForecastPoint(points, nowMs) {
        let best = null;
        for (const point of points) {
            const ts = pointTimestamp(point);
            if (ts <= nowMs && (best === null || ts > pointTimestamp(best))) best = point;
        }
        return best;
    }

    function derivePvWindow(points, nowMs, minimumRuntimeMs, stepMs) {
        const firstPairIndex = points.findIndex((point, index) =>
            index < points.length - 1 && isUsablePvPoint(point) && isUsablePvPoint(points[index + 1])
        );
        let lastPairEndIndex = -1;
        for (let index = points.length - 1; index > 0; index--) {
            if (isUsablePvPoint(points[index]) && isUsablePvPoint(points[index - 1])) {
                lastPairEndIndex = index;
                break;
            }
        }

        if (firstPairIndex < 0 || lastPairEndIndex < firstPairIndex + 1) {
            return { startMs: null, endMs: null, state: "UNAVAILABLE" };
        }

        const startMs = pointTimestamp(points[firstPairIndex]);
        const endMs = pointTimestamp(points[lastPairEndIndex]) + stepMs;
        const state = nowMs < startMs
            ? "BEFORE_WINDOW"
            : nowMs >= endMs
                ? "AFTER_WINDOW"
                : nowMs + minimumRuntimeMs > endMs
                    ? "WINDOW_TOO_SHORT"
                    : "WITHIN_WINDOW";
        return { startMs, endMs, state };
    }

    function pvWindowReason(state) {
        const reasons = {
            UNAVAILABLE: "No usable PV window in forecast",
            BEFORE_WINDOW: "Before usable PV window",
            WITHIN_WINDOW: "Within usable PV window",
            WINDOW_TOO_SHORT: "Not enough usable PV time for minimum runtime",
            AFTER_WINDOW: "After usable PV window"
        };
        return reasons[state];
    }

    try {
        let currentSoc = -1;
        let socTimestampMs = null;
        let socAgeMs = null;
        let fullForecastList = [];
        let dataError = null;
        let dataValid = false;

        const rowsArray = Array.isArray(rows) ? rows : [];
        const forecastPoints = [];
        let newestSoc = null;

        for (const row of rowsArray) {
            if (!row || typeof row !== "object") continue;
            const timestampMs = Date.parse(row._time);
            const value = Number(row._value);
            if (!Number.isFinite(timestampMs) || !Number.isFinite(value)) continue;
            if (row._field === "type_soc") {
                if (newestSoc === null || timestampMs > newestSoc.timestampMs) {
                    newestSoc = { timestampMs, soc: value };
                }
            } else if (row._field === "type_forecast") {
                forecastPoints.push({ timestampMs, watts: value });
            }
        }

        forecastPoints.sort((a, b) => a.timestampMs - b.timestampMs);
        fullForecastList = forecastPoints;

        if (newestSoc === null) {
            dataError = "SOC_MISSING";
        } else if (newestSoc.soc < 0 || newestSoc.soc > 100) {
            dataError = "SOC_INVALID";
        } else {
            currentSoc = newestSoc.soc;
            socTimestampMs = newestSoc.timestampMs;
            socAgeMs = now - socTimestampMs;
            dataValid = true;
        }

        if (dataValid && currentSoc >= RECOVERY_SOC) {
            lowBatLock = false;
        }

        let currentForecastWatts = 0;
        const activeWindowStartMs = now - FORECAST_STEP_MS;
        const pvWindow = derivePvWindow(
            fullForecastList,
            now,
            conf.min_runtime_minutes * 60 * 1000,
            FORECAST_STEP_MS
        );
        const pvWindowStartMs = pvWindow.startMs;
        const pvWindowEndMs = pvWindow.endMs;
        const pvWindowState = pvWindow.state;
        const pvWindowStateReason = pvWindowReason(pvWindowState);
        const currentForecastPoint = selectCurrentForecastPoint(fullForecastList, now);
        if (currentForecastPoint) currentForecastWatts = currentForecastPoint.watts;

        const energyForecastPoints = pvWindowEndMs === null
            ? []
            : fullForecastList.filter(point =>
                pointTimestamp(point) >= activeWindowStartMs && pointTimestamp(point) < pvWindowEndMs
            );

        const currentUtcHour = new Date(now).getUTCHours();
        const currentBaseLoad = conf.base_load_w[currentUtcHour];
        const currentConsumerPowerW = Object.fromEntries(CONSUMER_IDS.map(id => [
            id,
            conf.consumers[id].power_w[currentUtcHour]
        ]));
        const enabledConsumers = CONSUMER_IDS.filter(id => conf.consumers[id].enabled);

        debugInfo = {
            state: "INIT",
            soc: currentSoc,
            config_used: conf,
            low_bat_lock_active: lowBatLock,
            hysteresis_recovery_target: RECOVERY_SOC,
            data_error: dataError,
            soc_timestamp: socTimestampMs === null ? null : new Date(socTimestampMs).toISOString(),
            soc_age_minutes: socAgeMs === null ? null : Math.round(socAgeMs / 60000),
            current_forecast_w: currentForecastWatts,
            current_base_load_w: currentBaseLoad,
            solar_gain_kwh: 0,
            base_loss_kwh: 0,
            surplus_kwh: 0,
            battery_energy_to_full_kwh: 0,
            cumulative_cycle_energy_kwh: 0,
            cumulative_cycle_soc_pct: 0,
            pv_window_start: null,
            pv_window_end: null,
            pv_window_state: "UNAVAILABLE",
            available_cascade_level: 0,
            cumulative_active_power_w: 0,
            consumers: {}
        };

        CONSUMER_IDS.forEach(id => {
            debugInfo.consumers[id] = {
                enabled: conf.consumers[id].enabled,
                state: 0,
                eligible: false,
                power_w: currentConsumerPowerW[id],
                cumulative_power_w: 0,
                cumulative_cycle_energy_kwh: 0,
                cumulative_cycle_soc_pct: 0,
                rejection_reason: "NOT_EVALUATED",
                battery_support_needed_w: null
            };
        });

        if (!dataValid) {
            statusFill = "grey";
            statusText = "Invalid Influx data";
            debugInfo.state = "ERROR_NO_DATA";
            CONSUMER_IDS.forEach(id => {
                debugInfo.consumers[id].rejection_reason = "INVALID_INFLUX_DATA";
            });
        } else {
            let solarKwhSum = 0;
            let baseLoadKwhSum = 0;
            const timeStepHours = 0.25;

            for (const point of energyForecastPoints) {
                const pointTime = pointTimestamp(point);
                const generatedWatts = point.watts;
                const pointBaseLoad = getBaseLoadAt(pointTime);

                const stepEnergy = (generatedWatts * timeStepHours) / 1000;
                const stepBaseLoad = (pointBaseLoad * timeStepHours) / 1000;

                solarKwhSum += stepEnergy;
                baseLoadKwhSum += stepBaseLoad;
            }

            const BATTERY_EFFICIENCY = conf.battery_efficiency;
            const currentBatteryEnergyKwh = (currentSoc / 100) * conf.battery_capacity_kwh;
            const targetCapacityKwh = conf.battery_capacity_kwh * TARGET_SOC_LIMIT;
            const currentChargeEnergyKwhNeeded = Math.max(
                0,
                (targetCapacityKwh - currentBatteryEnergyKwh) / BATTERY_EFFICIENCY
            );
            const fullCapacityEnergyKwhNeeded = Math.max(
                0,
                (conf.battery_capacity_kwh - currentBatteryEnergyKwh) / BATTERY_EFFICIENCY
            );
            debugInfo.battery_energy_to_full_kwh = fullCapacityEnergyKwhNeeded;

            const projectedEnergyEndOfDay = currentBatteryEnergyKwh + (solarKwhSum - baseLoadKwhSum);
            const effectiveSoc = Math.min(1.0, Math.max(0, currentSoc / 95.0));
            const socFactor = 0.1 + (0.9 * (1 - Math.pow(effectiveSoc, 2)));
            const dynamicReserve = conf.reserve_kwh * socFactor;
            const targetEnergyEndOfDay = currentBatteryEnergyKwh + currentChargeEnergyKwhNeeded + dynamicReserve;
            const surplusKwh = projectedEnergyEndOfDay - targetEnergyEndOfDay;

            debugInfo.state = "CALCULATED";
            debugInfo.current_forecast_w = currentForecastWatts;
            debugInfo.solar_gain_kwh = solarKwhSum;
            debugInfo.base_loss_kwh = baseLoadKwhSum;
            debugInfo.surplus_kwh = surplusKwh;
            debugInfo.pv_window_start = pvWindowStartMs === null ? null : new Date(pvWindowStartMs).toISOString();
            debugInfo.pv_window_end = pvWindowEndMs === null ? null : new Date(pvWindowEndMs).toISOString();
            debugInfo.pv_window_state = pvWindowState;
            debugInfo.dynamic_reserve_kwh = dynamicReserve;

            let commonRejection = "";
            let commonState = "";
            let availableCascadeLevel = 0;
            let cumulativeConfiguredPowerW = 0;
            let cumulativeCycleEnergyKwh = 0;
            let previousStageFailed = false;

            if (currentSoc <= conf.min_soc) {
                if (CONSUMER_IDS.some(id => previousConsumerStates[id] === 1)) {
                    lowBatLock = true;
                    commonState = "LOW_BATTERY_CRASH";
                } else {
                    commonState = "LOW_BATTERY_IDLE";
                }
                commonRejection = "LOW_BATTERY";
            } else if (lowBatLock === true) {
                commonRejection = "HYSTERESIS_LOCKED";
                commonState = "HYSTERESIS_LOCKED";
            } else if (pvWindowState !== "WITHIN_WINDOW") {
                commonRejection = "PV_WINDOW_GUARD";
                commonState = "SAFETY_GUARD_ACTIVE";
            }

            for (let index = 0; index < CONSUMER_IDS.length; index++) {
                const id = CONSUMER_IDS[index];
                const consumer = conf.consumers[id];
                const decision = debugInfo.consumers[id];
                if (consumer.enabled) cumulativeConfiguredPowerW += currentConsumerPowerW[id];

                const candidateCycleKwh = (cumulativeConfiguredPowerW / 1000) * (conf.min_runtime_minutes / 60);
                const candidateCycleSoc = (candidateCycleKwh / conf.battery_capacity_kwh) * 100 / BATTERY_EFFICIENCY;
                const neededBuffer = Math.max(conf.soc_hysteresis, candidateCycleSoc);
                const requiredSafeBufferSoc = conf.min_soc + neededBuffer;

                decision.cumulative_power_w = cumulativeConfiguredPowerW;
                decision.cumulative_cycle_energy_kwh = candidateCycleKwh;
                decision.cumulative_cycle_soc_pct = candidateCycleSoc;

                if (commonRejection) {
                    decision.rejection_reason = commonRejection;
                    continue;
                }
                if (previousStageFailed) {
                    decision.rejection_reason = "BLOCKED_BY_PREVIOUS_STAGE";
                    continue;
                }
                if (!consumer.enabled) {
                    decision.eligible = true;
                    decision.rejection_reason = "";
                    continue;
                }

                let rejection = "";
                const totalLoadW = currentBaseLoad + cumulativeConfiguredPowerW;
                const batterySupportNeededW = totalLoadW - currentForecastWatts;
                decision.battery_support_needed_w = batterySupportNeededW;

                if (currentForecastWatts <= currentBaseLoad) {
                    rejection = "FORECAST_BELOW_BASE_LOAD";
                } else if (requiredSafeBufferSoc > 100) {
                    rejection = "CUMULATIVE_SAFETY_BUFFER_OVER_100";
                } else if (surplusKwh < candidateCycleKwh) {
                    rejection = surplusKwh > 0 ? "CUMULATIVE_SURPLUS_TOO_LOW" : "CUMULATIVE_SURPLUS_NEGATIVE";
                } else if (conf.allow_battery_support && batterySupportNeededW > conf.max_battery_support_w) {
                    rejection = "GRID_IMPORT_REQUIRED";
                } else if (conf.allow_battery_support && currentForecastWatts < totalLoadW && currentSoc < requiredSafeBufferSoc) {
                    rejection = "FILLING_BATTERY";
                } else if (totalLoadW >= currentForecastWatts && currentBaseLoad < currentForecastWatts && !conf.allow_battery_support) {
                    rejection = "BATTERY_SUPPORT_DISABLED";
                }

                if (rejection) {
                    decision.rejection_reason = rejection;
                    previousStageFailed = true;
                    continue;
                }

                decision.eligible = true;
                decision.rejection_reason = "";
                availableCascadeLevel = index + 1;
                cumulativeCycleEnergyKwh = candidateCycleKwh;
            }

            CONSUMER_IDS.forEach((id, index) => {
                const canRun = index < availableCascadeLevel && conf.consumers[id].enabled;
                consumerStates[id] = canRun ? 1 : 0;
                debugInfo.consumers[id].state = canRun ? 1 : 0;
            });

            const activeConsumerIds = CONSUMER_IDS.filter(id => consumerStates[id] === 1);
            const activeConsumerText = activeConsumerIds.map(id => `C${CONSUMER_IDS.indexOf(id) + 1}`).join(",");
            const hasActiveRelay = activeConsumerIds.length > 0;
            const activePowerW = activeConsumerIds.reduce((total, id) => total + currentConsumerPowerW[id], 0);
            const minRuntimeMs = conf.min_runtime_minutes * 60 * 1000;

            const fmtWatts = value => `${Math.round(value)} W`;
            const fmtMinutes = ms => `${Math.max(0, Math.round(ms / 60000))} min`;
            const fmtClock = value => {
                const stamp = new Date(value);
                return `${String(stamp.getUTCHours()).padStart(2, "0")}:${String(stamp.getUTCMinutes()).padStart(2, "0")} UTC`;
            };

            const firstFailedIndex = CONSUMER_IDS.findIndex(id => {
                const stageRejection = debugInfo.consumers[id].rejection_reason;
                return stageRejection && stageRejection !== "BLOCKED_BY_PREVIOUS_STAGE" && stageRejection !== "NOT_EVALUATED";
            });
            const downstreamIndex = firstFailedIndex < 0 ? -1 : CONSUMER_IDS.findIndex((id, index) =>
                index >= firstFailedIndex && conf.consumers[id].enabled
            );
            const failedId = firstFailedIndex < 0 ? null : CONSUMER_IDS[firstFailedIndex];
            const blockedId = downstreamIndex < 0 ? null : CONSUMER_IDS[downstreamIndex];
            const rejection = failedId ? debugInfo.consumers[failedId].rejection_reason : "";
            const failedCycleEnergyKwh = failedId
                ? debugInfo.consumers[failedId].cumulative_cycle_energy_kwh
                : 0;
            const failedCycleSocPct = failedId
                ? debugInfo.consumers[failedId].cumulative_cycle_soc_pct
                : 0;
            const failedStagePowerW = failedId
                ? debugInfo.consumers[failedId].cumulative_power_w
                : 0;
            const failedBatterySupportW = failedId
                ? debugInfo.consumers[failedId].battery_support_needed_w
                : 0;

            function readableRejection(stageRejection) {
                if (stageRejection === "CUMULATIVE_SURPLUS_NEGATIVE") {
                    return `energy deficit ${Math.abs(surplusKwh).toFixed(2)} kWh`;
                }
                if (stageRejection === "CUMULATIVE_SURPLUS_TOO_LOW") {
                    const deficit = Math.max(0, failedCycleEnergyKwh - surplusKwh);
                    return `energy deficit ${deficit.toFixed(2)} kWh`;
                }
                if (stageRejection === "FORECAST_BELOW_BASE_LOAD") {
                    return `PV below base load (${fmtWatts(currentForecastWatts)} <= ${fmtWatts(currentBaseLoad)})`;
                }
                if (stageRejection === "FILLING_BATTERY") {
                    const neededSoc = conf.min_soc + Math.max(conf.soc_hysteresis, failedCycleSocPct);
                    return `battery charging (SoC ${currentSoc.toFixed(1)}% < ${neededSoc.toFixed(1)}%)`;
                }
                if (stageRejection === "BATTERY_SUPPORT_DISABLED") {
                    return `battery support disabled (PV ${fmtWatts(currentForecastWatts)}, load ${fmtWatts(currentBaseLoad + failedStagePowerW)})`;
                }
                if (stageRejection === "GRID_IMPORT_REQUIRED") {
                    return `grid import required (battery ${fmtWatts(failedBatterySupportW)} > ${fmtWatts(conf.max_battery_support_w)})`;
                }
                if (stageRejection === "CUMULATIVE_SAFETY_BUFFER_OVER_100") return "safety buffer exceeds 100%";
                if (stageRejection === "LOW_BATTERY") return `low battery (${currentSoc.toFixed(1)}%)`;
                if (stageRejection === "HYSTERESIS_LOCKED") return `hysteresis lock (${currentSoc.toFixed(1)}%)`;
                if (stageRejection === "PV_WINDOW_GUARD") return pvWindowStateReason;
                return stageRejection || "not available";
            }

            if (commonRejection) {
                debugInfo.state = commonState;
            } else if (enabledConsumers.length === 0) {
                debugInfo.state = "NO_CONSUMERS_ENABLED";
            } else if (currentForecastPoint === null) {
                debugInfo.state = "NO_CURRENT_FORECAST_POINT";
            } else if (hasActiveRelay) {
                debugInfo.state = "STAGED_CONSUMERS_ON";
            } else if (rejection === "CUMULATIVE_SURPLUS_NEGATIVE") {
                debugInfo.state = "SURPLUS_IS_NEGATIVE";
            } else if (rejection === "CUMULATIVE_SURPLUS_TOO_LOW") {
                debugInfo.state = "SURPLUS_TOO_LOW";
            } else if (rejection === "FILLING_BATTERY") {
                debugInfo.state = "CURRENT_POWER_AND_SOC_LOW";
            } else if (rejection === "BATTERY_SUPPORT_DISABLED") {
                debugInfo.state = "BATTERY_SUPPORT_DISABLED";
            } else if (rejection === "GRID_IMPORT_REQUIRED") {
                debugInfo.state = "GRID_IMPORT_REQUIRED";
            } else if (rejection === "FORECAST_BELOW_BASE_LOAD") {
                debugInfo.state = "SURPLUS_TOO_LOW";
            } else {
                debugInfo.state = "CONSUMER_STAGE_BLOCKED";
            }

            if (commonRejection) {
                if (commonState === "LOW_BATTERY_IDLE" || commonState === "LOW_BATTERY_CRASH") {
                    statusFill = "red";
                    statusText = `Low SoC ${currentSoc.toFixed(1)}% <= ${conf.min_soc}%`;
                } else if (commonState === "HYSTERESIS_LOCKED") {
                    statusFill = "grey";
                    statusText = `Hysteresis lock: SoC ${currentSoc.toFixed(1)}% < ${RECOVERY_SOC}%`;
                } else if (pvWindowState === "UNAVAILABLE") {
                    statusFill = "grey";
                    statusText = "No usable PV window in forecast";
                } else if (pvWindowState === "BEFORE_WINDOW") {
                    statusFill = "grey";
                    statusText = pvWindowStartMs === null
                        ? "Before PV window"
                        : `Before PV window (from ${fmtClock(pvWindowStartMs)})`;
                } else if (pvWindowState === "AFTER_WINDOW") {
                    statusFill = "grey";
                    statusText = pvWindowEndMs === null
                        ? "After PV window"
                        : `After PV window (ended ${fmtClock(pvWindowEndMs)})`;
                } else {
                    const remainingMs = pvWindowEndMs === null ? 0 : Math.max(0, pvWindowEndMs - now);
                    statusFill = "yellow";
                    statusText = `PV window too short: ${fmtMinutes(remainingMs)} left, ${fmtMinutes(minRuntimeMs)} needed`;
                }
            } else if (enabledConsumers.length === 0) {
                statusFill = "grey";
                statusText = "No consumers enabled";
            } else if (currentForecastPoint === null) {
                statusFill = "grey";
                statusText = "No current PV forecast point";
            } else if (currentForecastWatts <= currentBaseLoad) {
                statusFill = "red";
                statusText = `PV below base load: ${fmtWatts(currentForecastWatts)} <= ${fmtWatts(currentBaseLoad)}`;
            } else if (hasActiveRelay) {
                const requiredPowerW = currentBaseLoad + activePowerW;
                if (conf.allow_battery_support && currentForecastWatts < requiredPowerW) {
                    const batterySupportW = requiredPowerW - currentForecastWatts;
                    statusFill = "blue";
                    statusText = `${activeConsumerText} ON | battery support ${fmtWatts(batterySupportW)} (PV ${fmtWatts(currentForecastWatts)}, load ${fmtWatts(requiredPowerW)})`;
                } else {
                    statusFill = "green";
                    statusText = blockedId && rejection
                        ? `${activeConsumerText} ON | C${downstreamIndex + 1} blocked: ${readableRejection(rejection)}`
                        : `${activeConsumerText} ON`;
                }
            } else if (rejection) {
                statusFill = "yellow";
                statusText = `All OFF | ${readableRejection(rejection)}`;
            } else {
                statusFill = "yellow";
                statusText = "All OFF | no consumer available";
            }

            debugInfo.available_cascade_level = availableCascadeLevel;
            debugInfo.cumulative_active_power_w = activePowerW;
            debugInfo.cumulative_cycle_energy_kwh = cumulativeCycleEnergyKwh;
            debugInfo.cumulative_cycle_soc_pct =
                cumulativeCycleEnergyKwh / conf.battery_capacity_kwh * 100 / BATTERY_EFFICIENCY;
        }
    } catch (err) {
        lowBatLock = lowBatLockAtStart;
        consumerStates = { consumer1: 0, consumer2: 0, consumer3: 0 };
        statusFill = "red";
        statusText = `Error: ${err.message}`;
        debugInfo = {
            state: "ERROR_EXCEPTION",
            error: err.message,
            consumers: {},
            available_cascade_level: 0,
            cumulative_active_power_w: 0
        };
        error = err;
    }

    return {
        payload: debugInfo.available_cascade_level,
        consumerStates,
        debug: debugInfo,
        status: {
            fill: statusFill,
            shape: Object.values(consumerStates).some(state => state === 1) ? "dot" : "ring",
            text: statusText
        },
        nextContextState: {
            lowBatLock,
            lastState: consumerStates.consumer1,
            consumerStates,
            stateConvention: STATE_CONVENTION
        },
        error
    };
}

module.exports = { evaluate, CONSUMER_IDS, STATE_CONVENTION };
