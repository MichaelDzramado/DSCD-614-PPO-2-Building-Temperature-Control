"""Physics-inspired single-zone building thermal model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThermalConfig:
    """Parameters governing single-zone indoor temperature dynamics."""

    outdoor_coupling: float = 0.08
    hvac_effect_c_per_action: float = 3.0
    occupancy_heat_gain_c: float = 0.15
    disturbance_std_c: float = 0.0


class ThermalModel:
    """Simple physics-inspired building thermal model.

    Positive action = heating.
    Negative action = cooling.
    """

    def __init__(
        self,
        config: ThermalConfig | None = None,
        rng=None,
    ):
        self.config = config or ThermalConfig()
        self.rng = rng

    def next_temperature(
        self,
        indoor_temp_c: float,
        outdoor_temp_c: float,
        occupancy: float,
        action: float,
    ) -> float:
        """Calculate the next indoor temperature."""

        if not -1.0 <= action <= 1.0:
            raise ValueError("HVAC action must be in [-1, 1].")

        if not 0.0 <= occupancy <= 1.0:
            raise ValueError("Occupancy must be in [0, 1].")

        cfg = self.config

        outdoor_effect = (
            cfg.outdoor_coupling
            * (outdoor_temp_c - indoor_temp_c)
        )

        hvac_effect = (
            cfg.hvac_effect_c_per_action
            * action
        )

        occupancy_effect = (
            cfg.occupancy_heat_gain_c
            * occupancy
        )

        disturbance = 0.0

        if cfg.disturbance_std_c > 0:
            if self.rng is None:
                raise ValueError(
                    "An RNG is required when disturbance_std_c > 0."
                )

            disturbance = self.rng.normal(
                0.0,
                cfg.disturbance_std_c,
            )

        next_temp = (
            indoor_temp_c
            + outdoor_effect
            + hvac_effect
            + occupancy_effect
            + disturbance
        )

        return float(next_temp)
