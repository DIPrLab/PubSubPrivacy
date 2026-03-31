"""
Factory IoT data generators — realistic sensor publishers for a
manufacturing facility with multiple production lines, machines,
and sensor types.

Topic hierarchy:
  factory/raw/{line_id}/{machine_id}/{sensor_type}

Each publisher simulates a sensor with:
  - Realistic baseline + drift (machine warm-up, cooldown)
  - Operating regime shifts (idle, running, maintenance)
  - Correlated sensor readings (e.g., power draw affects temperature)
  - Gaussian measurement noise
"""

from __future__ import annotations

import json
import logging
import math
import random
import threading
import time
from dataclasses import dataclass
from enum import Enum

import numpy as np
import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)


class MachineState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    MAINTENANCE = "maintenance"


@dataclass
class SensorSpec:
    """Specification for a sensor type."""

    sensor_type: str
    unit: str
    min_val: float
    max_val: float
    publish_interval: float
    noise_std: float


# Realistic operating profiles per machine state
OPERATING_PROFILES = {
    "temperature": {
        MachineState.IDLE: {"base_frac": 0.15, "variation": 0.02},
        MachineState.RUNNING: {"base_frac": 0.65, "variation": 0.08},
        MachineState.MAINTENANCE: {"base_frac": 0.10, "variation": 0.01},
    },
    "vibration": {
        MachineState.IDLE: {"base_frac": 0.02, "variation": 0.01},
        MachineState.RUNNING: {"base_frac": 0.45, "variation": 0.15},
        MachineState.MAINTENANCE: {"base_frac": 0.05, "variation": 0.02},
    },
    "power_draw": {
        MachineState.IDLE: {"base_frac": 0.03, "variation": 0.01},
        MachineState.RUNNING: {"base_frac": 0.70, "variation": 0.10},
        MachineState.MAINTENANCE: {"base_frac": 0.01, "variation": 0.005},
    },
    "humidity": {
        MachineState.IDLE: {"base_frac": 0.40, "variation": 0.05},
        MachineState.RUNNING: {"base_frac": 0.55, "variation": 0.08},
        MachineState.MAINTENANCE: {"base_frac": 0.35, "variation": 0.03},
    },
}


class SensorPublisher:
    def __init__(
        self,
        client: mqtt.Client,
        raw_prefix: str,
        line_id: str,
        machine_id: str,
        spec: SensorSpec,
        machine_state_fn,
    ):
        self.client = client
        self.topic = f"{raw_prefix}/{line_id}/{machine_id}/{spec.sensor_type}"
        self.publisher_id = f"{line_id}/{machine_id}/{spec.sensor_type}"
        self.spec = spec
        self.machine_state_fn = machine_state_fn

        self._phase = random.uniform(0, 2 * math.pi)
        self._drift = 0.0
        self._rng = np.random.default_rng(hash(self.publisher_id) % (2**31))

    def generate_value(self, elapsed: float) -> float:
        state = self.machine_state_fn()
        profile = OPERATING_PROFILES.get(self.spec.sensor_type, {}).get(
            state, {"base_frac": 0.5, "variation": 0.1}
        )

        val_range = self.spec.max_val - self.spec.min_val
        base = self.spec.min_val + profile["base_frac"] * val_range

        # Slow sinusoidal drift (thermal cycling, load variation)
        drift = profile["variation"] * val_range * math.sin(
            2 * math.pi * elapsed / 120.0 + self._phase
        )

        # Gradual warm-up ramp over first 30 seconds
        warmup = min(1.0, elapsed / 30.0)

        # Measurement noise
        noise = self._rng.normal(0, self.spec.noise_std)

        value = base * warmup + drift + noise
        return float(np.clip(value, self.spec.min_val, self.spec.max_val))

    def publish(self, elapsed: float):
        value = self.generate_value(elapsed)
        payload = {
            "publisher_id": self.publisher_id,
            "value": round(value, 4),
            "unit": self.spec.unit,
            "ts": time.time(),
        }
        self.client.publish(self.topic, json.dumps(payload))


class FactorySimulator:
    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        raw_prefix: str = "factory/raw",
        num_lines: int = 2,
        machines_per_line: int = 4,
        sensor_specs: list[SensorSpec] | None = None,
        publish_interval: float = 5.0,
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.raw_prefix = raw_prefix
        self.num_lines = num_lines
        self.machines_per_line = machines_per_line
        self.publish_interval = publish_interval

        if sensor_specs is None:
            sensor_specs = [
                SensorSpec("temperature", "celsius", 15.0, 120.0, publish_interval, 0.5),
                SensorSpec("vibration", "mm/s", 0.0, 50.0, publish_interval, 0.3),
                SensorSpec("power_draw", "kW", 0.0, 500.0, publish_interval, 2.0),
                SensorSpec("humidity", "percent", 10.0, 95.0, publish_interval, 0.8),
            ]
        self.sensor_specs = sensor_specs

        # Machine states
        self._machine_states: dict[str, MachineState] = {}
        self._publishers: list[SensorPublisher] = []
        self._client: mqtt.Client | None = None
        self._running = False

    def _init_machines(self):
        for line in range(1, self.num_lines + 1):
            for mach in range(1, self.machines_per_line + 1):
                line_id = f"line{line:02d}"
                machine_id = f"machine{mach:02d}"
                key = f"{line_id}/{machine_id}"
                self._machine_states[key] = MachineState.RUNNING

                for spec in self.sensor_specs:
                    pub = SensorPublisher(
                        client=self._client,
                        raw_prefix=self.raw_prefix,
                        line_id=line_id,
                        machine_id=machine_id,
                        spec=spec,
                        machine_state_fn=lambda k=key: self._machine_states[k],
                    )
                    self._publishers.append(pub)

    def _transition_machines(self):
        for key, state in self._machine_states.items():
            r = random.random()
            if state == MachineState.RUNNING:
                if r < 0.02:  # 2% chance to go idle
                    self._machine_states[key] = MachineState.IDLE
                elif r < 0.025:  # 0.5% chance to enter maintenance
                    self._machine_states[key] = MachineState.MAINTENANCE
            elif state == MachineState.IDLE:
                if r < 0.15:  # 15% chance to resume
                    self._machine_states[key] = MachineState.RUNNING
            elif state == MachineState.MAINTENANCE:
                if r < 0.05:  # 5% chance to finish maintenance
                    self._machine_states[key] = MachineState.RUNNING

    def start(self, duration: float, blocking: bool = True):
        self._client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"factory-sim-{random.randint(0, 9999)}",
        )
        self._client.connect(self.broker_host, self.broker_port)
        self._client.loop_start()
        self._init_machines()
        self._running = True

        def _run():
            start_time = time.time()
            while self._running and (time.time() - start_time) < duration:
                elapsed = time.time() - start_time
                self._transition_machines()
                for pub in self._publishers:
                    pub.publish(elapsed)
                time.sleep(self.publish_interval)
            self._running = False
            self._client.loop_stop()
            self._client.disconnect()
            logger.info("Factory simulation completed")

        if blocking:
            _run()
        else:
            t = threading.Thread(target=_run, daemon=True)
            t.start()
            return t

    def stop(self):
        self._running = False
