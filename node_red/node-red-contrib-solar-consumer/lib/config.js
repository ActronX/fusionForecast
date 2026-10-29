"use strict";

const DEFAULTS = require("./defaults");

function isPlainObject(value) {
    return value !== null && typeof value === "object" && !Array.isArray(value);
}

function clone(value) {
    if (Array.isArray(value)) return value.map(clone);
    if (isPlainObject(value)) {
        const copy = {};
        Object.keys(value).forEach(key => {
            copy[key] = clone(value[key]);
        });
        return copy;
    }
    return value;
}

function mergeDeep(base, override) {
    if (!isPlainObject(override)) return clone(base);
    const result = isPlainObject(base) ? clone(base) : {};
    Object.keys(override).forEach(key => {
        const value = override[key];
        if (isPlainObject(value) && isPlainObject(result[key])) {
            result[key] = mergeDeep(result[key], value);
        } else {
            result[key] = clone(value);
        }
    });
    return result;
}

function requireVal(val, name) {
    const isNumericString = typeof val === "string" && val.trim() !== "";
    if (typeof val !== "number" && !isNumericString) {
        throw new Error(`Invalid value for '${name}': ${val}`);
    }
    const num = Number(val);
    if (!Number.isFinite(num) || num < 0) {
        throw new Error(`Invalid value for '${name}': ${val}`);
    }
    return num;
}

function requirePositiveVal(val, name) {
    const num = requireVal(val, name);
    if (num <= 0) {
        throw new Error(`Invalid value for '${name}': expected a positive number`);
    }
    return num;
}

function parseHourlyArray(rawValue, name) {
    if (Array.isArray(rawValue)) {
        if (rawValue.length !== 24) {
            throw new Error(`${name} array must have exactly 24 elements (0-23h)`);
        }
        return rawValue.map(value => requireVal(value, `${name} item`));
    }
    const singleVal = requireVal(rawValue, name);
    return Array(24).fill(singleVal);
}

function parseConsumer(rawValue, name) {
    if (!rawValue || typeof rawValue !== "object" || Array.isArray(rawValue)) {
        throw new Error(`Invalid value for '${name}': expected an object`);
    }
    if (typeof rawValue.enabled !== "boolean") {
        throw new Error(`Invalid value for '${name}.enabled': expected a Boolean`);
    }
    if (!Array.isArray(rawValue.power_w)) {
        throw new Error(`Invalid value for '${name}.power_w': expected a 24-element array`);
    }
    if (rawValue.power_w.length !== 24) {
        throw new Error(`Invalid value for '${name}.power_w': array must have exactly 24 elements (0-23h)`);
    }
    const powerW = Array.from(rawValue.power_w, (value, index) => {
        if (typeof value !== "number" || !Number.isFinite(value)) {
            throw new Error(`Invalid value for '${name}.power_w[${index}]': expected a finite number`);
        }
        if (value < 0) {
            throw new Error(`Invalid value for '${name}.power_w[${index}]': expected a non-negative number`);
        }
        return value;
    });
    return { enabled: rawValue.enabled, power_w: powerW };
}

function validateConfig(rawConf) {
    if (!isPlainObject(rawConf)) {
        throw new Error("Invalid consumer configuration: expected an object");
    }
    const parsedBaseLoad = parseHourlyArray(rawConf.base_load_w, "base_load_w");
    const consumers = {
        consumer1: parseConsumer(rawConf.consumers && rawConf.consumers.consumer1, "consumers.consumer1"),
        consumer2: parseConsumer(rawConf.consumers && rawConf.consumers.consumer2, "consumers.consumer2"),
        consumer3: parseConsumer(rawConf.consumers && rawConf.consumers.consumer3, "consumers.consumer3")
    };
    const minSoc = requireVal(rawConf.min_soc, "min_soc");
    const socHysteresis = requireVal(rawConf.soc_hysteresis, "soc_hysteresis");
    if (minSoc > 100 || socHysteresis > 100 || minSoc + socHysteresis > 100) {
        throw new Error("min_soc and soc_hysteresis must be within 0-100 and total at most 100");
    }
    if (typeof rawConf.allow_battery_support !== "boolean") {
        throw new Error("Invalid value for 'allow_battery_support': expected a Boolean");
    }
    const config = {
        min_soc: minSoc,
        soc_hysteresis: socHysteresis,
        min_runtime_minutes: requirePositiveVal(rawConf.min_runtime_minutes, "min_runtime_minutes"),
        battery_capacity_kwh: requirePositiveVal(rawConf.battery_capacity_kwh, "battery_capacity_kwh"),
        reserve_kwh: requireVal(rawConf.reserve_kwh, "reserve_kwh"),
        consumers,
        base_load_w: parsedBaseLoad,
        battery_efficiency: requirePositiveVal(rawConf.battery_efficiency, "battery_efficiency"),
        allow_battery_support: rawConf.allow_battery_support,
        max_battery_support_w: requireVal(rawConf.max_battery_support_w, "max_battery_support_w")
    };
    if (config.battery_efficiency > 1) {
        throw new Error("battery_efficiency must be at most 1 when expressed as a fraction");
    }
    return config;
}

function buildConfig(editorConfig, override) {
    const withDefaults = mergeDeep(DEFAULTS, editorConfig || {});
    const merged = mergeDeep(withDefaults, override || {});
    return validateConfig(merged);
}

module.exports = {
    DEFAULTS,
    clone,
    mergeDeep,
    requireVal,
    requirePositiveVal,
    parseHourlyArray,
    parseConsumer,
    validateConfig,
    buildConfig
};
