"use strict";

const BASE_LOAD_W = [
    300, 301, 302, 303, 304, 305,
    306, 407, 508, 509, 1110, 1111,
    612, 613, 614, 515, 616, 617,
    518, 519, 420, 421, 322, 323
];

const CONSUMER1_POWER_W = [
    600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 10, 11,
    612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623
];

module.exports = {
    min_soc: 20,
    soc_hysteresis: 15,
    min_runtime_minutes: 30,
    battery_capacity_kwh: 4.5,
    reserve_kwh: 2,
    battery_efficiency: 0.85,
    allow_battery_support: true,
    max_battery_support_w: 2000,
    base_load_w: BASE_LOAD_W,
    consumers: {
        consumer1: { enabled: true, power_w: CONSUMER1_POWER_W },
        consumer2: { enabled: true, power_w: Array(24).fill(1800) },
        consumer3: { enabled: true, power_w: Array(24).fill(800) }
    }
};
