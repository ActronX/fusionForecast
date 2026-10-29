"use strict";

const { buildConfig } = require("./lib/config");
const { evaluate, CONSUMER_IDS } = require("./lib/evaluate");

module.exports = function (RED) {
    function SolarConsumerNode(config) {
        RED.nodes.createNode(this, config);
        const node = this;

        function editorConfig() {
            let consumers = config.consumers;
            if (typeof consumers === "string") {
                try {
                    consumers = JSON.parse(consumers);
                } catch (err) {
                    consumers = null;
                }
            }
            return {
                min_soc: config.min_soc,
                soc_hysteresis: config.soc_hysteresis,
                min_runtime_minutes: config.min_runtime_minutes,
                battery_capacity_kwh: config.battery_capacity_kwh,
                reserve_kwh: config.reserve_kwh,
                battery_efficiency: config.battery_efficiency,
                allow_battery_support: config.allow_battery_support,
                max_battery_support_w: config.max_battery_support_w,
                base_load_w: config.base_load_w,
                consumers
            };
        }

        node.on("input", function (msg, send, done) {
            let conf;
            try {
                const override = msg.config && msg.config.consumer
                    ? msg.config.consumer
                    : undefined;
                conf = buildConfig(editorConfig(), override);
            } catch (err) {
                const text = `Config error: ${err.message}`;
                node.status({ fill: "grey", shape: "ring", text });
                node.error(err, msg);
                const consumerMessages = CONSUMER_IDS.map((id, index) => ({
                    payload: false,
                    topic: `solar-consumer/consumer${index + 1}`
                }));
                const debugMsg = {
                    payload: { state: "CONFIG_ERROR", error: err.message },
                    topic: "solar-consumer/debug"
                };
                send(consumerMessages.concat([debugMsg]));
                if (done) done(err);
                return;
            }

            const rawRows = msg.payload;
            const context = node.context();
            const contextState = {
                lastState: context.get("lastState"),
                consumerStates: context.get("consumerStates"),
                lowBatLock: context.get("lowBatLock"),
                stateConvention: context.get("stateConvention")
            };

            if (msg.reset === true) {
                contextState.lastState = 0;
                contextState.consumerStates = { consumer1: 0, consumer2: 0, consumer3: 0 };
                contextState.lowBatLock = false;
                contextState.stateConvention = 2;
            }

            const result = evaluate({
                config: conf,
                rows: rawRows,
                contextState,
                nowMs: Date.now()
            });

            context.set("lowBatLock", result.nextContextState.lowBatLock);
            context.set("lastState", result.nextContextState.lastState);
            context.set("consumerStates", result.nextContextState.consumerStates);
            context.set("stateConvention", result.nextContextState.stateConvention);

            node.status(result.status);

            const consumerMessages = CONSUMER_IDS.map((id, index) => ({
                payload: result.consumerStates[id] === 1,
                topic: `solar-consumer/consumer${index + 1}`
            }));
            const debugMsg = RED.util.cloneMessage(msg);
            debugMsg.payload = result.debug;
            debugMsg.topic = "solar-consumer/debug";

            if (result.error) {
                node.error(result.error, msg);
            }

            send(consumerMessages.concat([debugMsg]));
            if (done) done();
        });
    }

    RED.nodes.registerType("solar-consumer", SolarConsumerNode);
};
