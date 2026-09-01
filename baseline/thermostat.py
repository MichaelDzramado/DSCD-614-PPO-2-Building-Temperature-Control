"""Rule-based thermostat baseline for PPO-2."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThermostatConfig:
    """Configuration for the deadband thermostat."""

    setpoint_c: float = 22.0
    tolerance_c: float = 1.0


class ThermostatController:
    """Simple rule-based heating/cooling controller."""

    def __init__(
        self,
        config: ThermostatConfig | None = None,
    ):
        self.config = config or ThermostatConfig()

    def get_action(self, indoor_temp_c: float) -> float:
        """Return HVAC action in [-1, 1]."""

        cfg = self.config

        lower_bound = cfg.setpoint_c - cfg.tolerance_c
        upper_bound = cfg.setpoint_c + cfg.tolerance_c

        if indoor_temp_c < lower_bound:
            return 1.0

        if indoor_temp_c > upper_bound:
            return -1.0

        return 0.0
