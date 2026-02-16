module.exports = function (RED) {
    function SolarConsumerNode(config) {
        RED.nodes.createNode(this, config);
        var node = this;

        // --- READ CONFIGURATION FROM EDITOR ---
        var nodeConf = {
            min_soc: parseFloat(config.min_soc) || 20,
            soc_hysteresis: parseFloat(config.soc_hysteresis) || 20,
            min_runtime_minutes: parseFloat(config.min_runtime_minutes) || 30,
            battery_capacity_kwh: parseFloat(config.battery_capacity_kwh) || 5.0,
            pv_peak_power_w: parseFloat(config.pv_peak_power_w) || 5200,
            reserve_kwh: parseFloat(config.reserve_kwh) || 2.0,
            consumer_power_w: parseFloat(config.consumer_power_w) || 1000,
            base_load_w: parseFloat(config.base_load_w) || 300,
            battery_efficiency: parseFloat(config.battery_efficiency) || 0.85,
            use_damping_factor: config.use_damping_factor,
            allow_battery_support: config.allow_battery_support
        };

        node.on('input', function (msg, send, done) {
            send = send || function () { node.send.apply(node, arguments); };

            // --- CONFIGURATION VALIDATION ---
            function requireVal(val, name) {
                var num = Number(val);
                if (isNaN(num) || num < 0) {
                    throw new Error("Invalid value for '" + name + "': " + val);
                }
                return num;
            }

            var conf;
            try {
                conf = {
                    min_soc: requireVal(nodeConf.min_soc, "min_soc"),
                    soc_hysteresis: Number(nodeConf.soc_hysteresis),
                    min_runtime_minutes: requireVal(nodeConf.min_runtime_minutes, "min_runtime_minutes"),
                    battery_capacity_kwh: requireVal(nodeConf.battery_capacity_kwh, "battery_capacity_kwh"),
                    pv_peak_power_w: requireVal(nodeConf.pv_peak_power_w, "pv_peak_power_w"),
                    reserve_kwh: requireVal(nodeConf.reserve_kwh, "reserve_kwh"),
                    consumer_power_w: requireVal(nodeConf.consumer_power_w, "consumer_power_w"),
                    base_load_w: requireVal(nodeConf.base_load_w, "base_load_w"),
                    battery_efficiency: requireVal(nodeConf.battery_efficiency, "battery_efficiency"),
                    use_damping_factor: Boolean(nodeConf.use_damping_factor),
                    allow_battery_support: Boolean(nodeConf.allow_battery_support)
                };
            } catch (error) {
                node.error(error.message, msg);
                node.status({ fill: "grey", shape: "ring", text: "Config Error" });
                if (done) done();
                return;
            }

            // --- HYSTERESIS CONSTANTS ---
            var RECOVERY_SOC = Math.min(conf.min_soc + conf.soc_hysteresis, 100);

            // --- LOAD CONTEXT (MEMORY) ---
            var context = node.context();
            var lastState = context.get('lastState');
            if (lastState === undefined) lastState = 1; // Default: OFF (1)

            var lowBatLock = context.get('lowBatLock');
            if (lowBatLock === undefined) lowBatLock = false;

            // --- EXTRACT DATA FROM INFLUX ---
            var currentSoc = -1;
            var fullForecastList = [];
            var productionList = [];
            var dataValid = false;

            if (Array.isArray(msg.payload)) {
                var socObj = msg.payload.find(function (r) { return r._field === "type_soc"; });
                if (socObj && socObj._value !== undefined) {
                    currentSoc = Number(socObj._value);
                }

                fullForecastList = msg.payload.filter(function (r) {
                    return r._field === "type_forecast" && r._time && r._value !== null;
                });

                productionList = msg.payload.filter(function (r) {
                    return r._field === "type_production" && r._time && r._value !== null;
                });

                try {
                    var timeSort = function (a, b) {
                        return new Date(a._time).getTime() - new Date(b._time).getTime();
                    };
                    fullForecastList.sort(timeSort);
                    productionList.sort(timeSort);
                } catch (e) {
                    node.warn("Error sorting: " + e.message);
                }

                if (currentSoc !== -1) dataValid = true;
            }

            // --- UPDATE LOCK STATE (RESET) ---
            if (dataValid && currentSoc >= RECOVERY_SOC) {
                lowBatLock = false;
            }

            // --- DAMPING FACTOR CALCULATION ---
            var nowMs = Date.now();
            var dampingFactor = 1.0;
            var sumForecastPast = 0;
            var sumProductionPast = 0;
            var dampingReason = "OK";
            var matchCount = 0;

            if (conf.use_damping_factor) {
                if (productionList.length > 0 && fullForecastList.length > 0) {
                    for (var i = 0; i < productionList.length; i++) {
                        var prod = productionList[i];
                        var pTime = new Date(prod._time).getTime();
                        if (pTime > nowMs) continue;

                        var match = fullForecastList.find(function (f) {
                            return Math.abs(new Date(f._time).getTime() - pTime) < 7.5 * 60 * 1000;
                        });

                        if (match) {
                            var fVal = Number(match._value);
                            var pVal = Number(prod._value);

                            if (fVal > conf.base_load_w / 2.0) {
                                var ageHours = Math.max(0, (nowMs - pTime) / (3600 * 1000));
                                var weight = Math.pow(0.5, ageHours / 1.0);
                                sumForecastPast += fVal * weight;
                                sumProductionPast += pVal * weight;
                                matchCount++;
                            }
                        }
                    }

                    if (sumForecastPast > conf.base_load_w / 2.0 * 3.0) {
                        dampingFactor = sumProductionPast / sumForecastPast;
                        dampingFactor = Math.max(0.75, Math.min(dampingFactor, 1.5));
                    } else {
                        dampingReason = "Insufficient Sun-Hours (Sums too low)";
                    }
                } else {
                    if (productionList.length === 0 && fullForecastList.length === 0) dampingReason = "No Data (Prod & Fcst)";
                    else if (productionList.length === 0) dampingReason = "No Production Data";
                    else if (fullForecastList.length === 0) dampingReason = "No Forecast Data";
                }
            } else {
                dampingReason = "Disabled by Config";
            }

            // --- FILTER & ADJUST FORECAST ---
            var forecastList = [];
            var currentForecastWatts = 0;

            if (fullForecastList.length > 0) {
                var activeWindowStart = nowMs - (15 * 60 * 1000);

                forecastList = fullForecastList
                    .filter(function (f) { return new Date(f._time).getTime() >= activeWindowStart; })
                    .map(function (item) {
                        var adjustedForecastValue = Number(item._value);

                        if (conf.use_damping_factor) {
                            var itemTime = new Date(item._time).getTime();
                            var diffHours = Math.max(0, (itemTime - nowMs) / (1000 * 60 * 60));
                            var decayWeight = Math.pow(0.5, diffHours / 1.0);
                            var effectiveFactor = 1.0 + ((dampingFactor - 1.0) * decayWeight);
                            adjustedForecastValue = Math.min(adjustedForecastValue * effectiveFactor, conf.pv_peak_power_w);
                        }

                        return { _time: item._time, _field: item._field, _value: adjustedForecastValue };
                    });

                if (forecastList.length > 0) {
                    currentForecastWatts = Number(forecastList[0]._value);
                }
            }

            // --- DEBUG INFO ---
            var debugInfo = {
                state: "INIT",
                soc: currentSoc,
                config_used: conf,
                low_bat_lock_active: lowBatLock,
                hysteresis_recovery_target: RECOVERY_SOC,
                current_forecast_w: currentForecastWatts,
                damping_factor: dampingFactor.toFixed(2),
                past_analysis: "Matches:" + matchCount + " | Prod:" + sumProductionPast.toFixed(0) + " vs Fcst:" + sumForecastPast.toFixed(0) + " (" + dampingReason + ")",
                solar_gain_kwh: 0,
                base_loss_kwh: 0,
                surplus_kwh: 0,
                required_cycle_kwh: 0,
                required_cycle_soc_pct: 0,
                cutoff_reason: "",
                remaining_sun_hours: 0
            };

            // --- LOGIC ---
            var switchState = 1; // Default OFF
            var reason = "";
            var color = "red";

            if (!dataValid) {
                reason = "Error: No Influx Data";
                color = "grey";
                debugInfo.state = "ERROR_NO_DATA";
            }
            else {
                // --- CALCULATION ---
                var solarKwhSum = 0;
                var baseLoadKwhSum = 0;
                var remainingSunHours = 0;
                var cutoffReason = "End of DB Forecast data";
                var timeStepHours = 0.25; // 15 min

                for (var j = 0; j < forecastList.length; j++) {
                    var point = forecastList[j];
                    var generatedWatts = Number(point._value);

                    if (generatedWatts < conf.base_load_w) {
                        cutoffReason = "Night: PV Power < House BaseLoad";
                        break;
                    }

                    var stepEnergy = (generatedWatts * timeStepHours) / 1000;
                    var stepBaseLoad = (conf.base_load_w * timeStepHours) / 1000;

                    solarKwhSum += stepEnergy;
                    baseLoadKwhSum += stepBaseLoad;
                    remainingSunHours += timeStepHours;
                }

                var BATTERY_EFFICIENCY = conf.battery_efficiency;
                var currentBatteryEnergyKwh = (currentSoc / 100) * conf.battery_capacity_kwh;

                var TARGET_SOC_LIMIT = 0.95;
                var targetCapacityKwh = conf.battery_capacity_kwh * TARGET_SOC_LIMIT;
                var currentChargeEnergyKwhNeeded = Math.max(0, (targetCapacityKwh - currentBatteryEnergyKwh) / BATTERY_EFFICIENCY);

                var projectedEnergyEndOfDay = currentBatteryEnergyKwh + (solarKwhSum - baseLoadKwhSum);

                // Dynamic Reserve Logic (Quadratic)
                var effectiveSoc = Math.min(1.0, Math.max(0, currentSoc / 95.0));
                var socFactor = 0.1 + (0.9 * (1 - Math.pow(effectiveSoc, 2)));
                var dynamicReserve = conf.reserve_kwh * socFactor;

                node.warn("[Reserve] currentSoc=" + currentSoc.toFixed(1) + "% -> effectiveSoc=" + effectiveSoc.toFixed(3) + " -> socFactor=" + socFactor.toFixed(3) + " -> dynamicReserve=" + dynamicReserve.toFixed(2) + "kWh (base=" + conf.reserve_kwh.toFixed(2) + "kWh)");

                var targetEnergyEndOfDay = currentBatteryEnergyKwh + currentChargeEnergyKwhNeeded + dynamicReserve;
                var surplusKwh = projectedEnergyEndOfDay - targetEnergyEndOfDay;

                var consumerKw = conf.consumer_power_w / 1000;
                var cycleCostKwh = consumerKw * (conf.min_runtime_minutes / 60);
                var cycleCostSoc = (cycleCostKwh / conf.battery_capacity_kwh) * 100 / BATTERY_EFFICIENCY;

                var neededBuffer = conf.allow_battery_support ? Math.max(conf.soc_hysteresis, cycleCostSoc) : conf.soc_hysteresis;
                var requiredSafeBufferSoc = conf.min_soc + neededBuffer;

                if (requiredSafeBufferSoc > 100) {
                    var err = "Safety Buffer " + requiredSafeBufferSoc.toFixed(1) + "% > 100%. Battery too small for consumer_power/min_runtime!";
                    node.error(err, msg);
                    node.status({ fill: "red", shape: "ring", text: "Config: Battery too small" });
                    if (done) done();
                    return;
                }

                var SAFE_BUFFER_SOC = requiredSafeBufferSoc;

                debugInfo.state = "CALCULATED";
                debugInfo.current_forecast_w = currentForecastWatts;
                debugInfo.solar_gain_kwh = solarKwhSum;
                debugInfo.base_loss_kwh = baseLoadKwhSum;
                debugInfo.surplus_kwh = surplusKwh;
                debugInfo.required_cycle_kwh = cycleCostKwh;
                debugInfo.required_cycle_soc_pct = cycleCostSoc;
                debugInfo.cutoff_reason = cutoffReason;
                debugInfo.remaining_sun_hours = remainingSunHours;
                debugInfo.dynamic_reserve_kwh = dynamicReserve;

                var MIN_REMAINING_SUN_HOURS = 2 * (conf.min_runtime_minutes / 60);

                // --- DECISION LOGIC ---

                // 1. Battery CRITICAL
                if (currentSoc <= conf.min_soc) {
                    if (lastState === 0) {
                        lowBatLock = true;
                        debugInfo.state = "LOW_BATTERY_CRASH";
                    } else {
                        debugInfo.state = "LOW_BATTERY_IDLE";
                    }
                    reason = "Low Battery (" + currentSoc.toFixed(1) + "% <= " + conf.min_soc + "%)";
                    switchState = 1;
                    color = "red";
                }
                // 2. Battery RECOVERING (Lock Active)
                else if (lowBatLock === true) {
                    reason = "Hysteresis Lock... Wait for " + RECOVERY_SOC + "% (curr: " + currentSoc.toFixed(1) + "%)";
                    debugInfo.state = "HYSTERESIS_LOCKED";
                    switchState = 1;
                    color = "blue";
                }
                // 3. Safety Guard
                else if (remainingSunHours <= MIN_REMAINING_SUN_HOURS) {
                    switchState = 1;
                    color = "black";
                    reason = "OFF. Remaining sun too short (" + remainingSunHours + "h <= " + MIN_REMAINING_SUN_HOURS + "h)";
                    debugInfo.state = "SAFETY_GUARD_ACTIVE";
                }
                // 4. Primary Check: Is Surplus Enough?
                else if (surplusKwh >= cycleCostKwh) {

                    if (conf.allow_battery_support && currentForecastWatts < (conf.consumer_power_w + conf.base_load_w) && currentSoc < SAFE_BUFFER_SOC) {
                        switchState = 1;
                        color = "yellow";
                        reason = "OFF. Filling Battery (" + currentForecastWatts.toFixed(0) + "W, " + currentSoc.toFixed(1) + "% < " + SAFE_BUFFER_SOC.toFixed(1) + "%)";
                        debugInfo.state = "CURRENT_POWER_AND_SOC_LOW";
                    }
                    else if ((conf.base_load_w + conf.consumer_power_w) < currentForecastWatts) {
                        switchState = 0;
                        color = "green";
                        reason = "ON. Full Power: " + currentForecastWatts.toFixed(0) + "W >= " + (conf.base_load_w + conf.consumer_power_w).toFixed(0) + "W";
                        debugInfo.state = "SURPLUS_ENOUGH";
                    }
                    else if (conf.base_load_w < currentForecastWatts) {
                        if (conf.allow_battery_support) {
                            switchState = 0;
                            color = "green";
                            reason = "ON. Battery Support: " + currentForecastWatts.toFixed(0) + "W < " + (conf.base_load_w + conf.consumer_power_w).toFixed(0) + "W";
                            debugInfo.state = "SURPLUS_DRAINING";
                        } else {
                            switchState = 1;
                            color = "yellow";
                            reason = "OFF. Battery Support Disabled (" + currentForecastWatts.toFixed(0) + "W < " + (conf.base_load_w + conf.consumer_power_w).toFixed(0) + "W)";
                            debugInfo.state = "BATTERY_SUPPORT_DISABLED";
                        }
                    }
                    else {
                        switchState = 1;
                        color = "yellow";
                        reason = "OFF. Forecast too low (" + currentForecastWatts.toFixed(0) + "W < " + conf.base_load_w.toFixed(0) + "W)";
                        debugInfo.state = "SURPLUS_TOO_LOW";
                    }
                }
                // 5. No Surplus
                else {
                    switchState = 1;
                    color = "yellow";
                    if (surplusKwh > 0) {
                        reason = "OFF. Surplus too low (" + surplusKwh.toFixed(2) + " < " + cycleCostKwh.toFixed(2) + ")";
                        debugInfo.state = "SURPLUS_TOO_LOW";
                    } else {
                        reason = "OFF. Deficit: " + Math.abs(surplusKwh).toFixed(2) + " kWh";
                        debugInfo.state = "SURPLUS_IS_NEGATIVE";
                    }
                }
            }

            // --- SAVE STATE ---
            context.set('lowBatLock', lowBatLock);
            context.set('lastState', switchState);

            msg.payload = switchState;
            msg.debug_calc = debugInfo;

            node.status({ fill: color, shape: "dot", text: reason });

            send(msg);
            if (done) done();
        });
    }

    RED.nodes.registerType("solar-consumer", SolarConsumerNode);
};
