"""Reward function for PPO-2 building temperature control."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    """Configuration for the energy-comfort reward."""

    energy_weight: float = 0.30
    cost_weight: float = 0.20
    comfort_weight: float = 0.40
    violation_weight: float = 1.00

    comfort_tolerance_c: float = 1.0

    # Broad safety limits used to identify severe temperature conditions.
    lower_temperature_limit_c: float = 18.0
    upper_temperature_limit_c: float = 26.0

    # Reference scales used for normalization.
    max_energy: float = 1.0
    max_price: float = 1.0


class RewardCalculator:
    """Calculate energy-efficiency and thermal-comfort rewards."""

    def __init__(self, config: RewardConfig | None = None):
        self.config = config or RewardConfig()

    def calculate(
        self,
        indoor_temp_c: float,
        setpoint_c: float,
        action: float,
        electricity_price: float,
    ) -> dict:
        """Calculate reward and its individual components."""

        cfg = self.config

        if not -1.0 <= action <= 1.0:
            raise ValueError("HVAC action must be in [-1, 1].")

        if electricity_price < 0:
            raise ValueError("Electricity price must be non-negative.")

        # --------------------------------------------------------
        # 1. HVAC energy
        # --------------------------------------------------------
        energy = abs(action) * cfg.max_energy

        normalized_energy = (
            energy / cfg.max_energy
            if cfg.max_energy > 0
            else 0.0
        )

        # --------------------------------------------------------
        # 2. Electricity cost
        # --------------------------------------------------------
        cost = electricity_price * energy

        max_reference_cost = cfg.max_price * cfg.max_energy

        normalized_cost = (
            cost / max_reference_cost
            if max_reference_cost > 0
            else 0.0
        )

        # --------------------------------------------------------
        # 3. Comfort deviation
        # --------------------------------------------------------
        deviation = abs(indoor_temp_c - setpoint_c)

        discomfort = max(
            0.0,
            deviation - cfg.comfort_tolerance_c,
        )

        # --------------------------------------------------------
        # 4. Severe temperature violation
        # --------------------------------------------------------
        violation = float(
            indoor_temp_c < cfg.lower_temperature_limit_c
            or indoor_temp_c > cfg.upper_temperature_limit_c
        )

        # --------------------------------------------------------
        # 5. Reward components
        # --------------------------------------------------------
        energy_penalty = (
            cfg.energy_weight * normalized_energy
        )

        cost_penalty = (
            cfg.cost_weight * normalized_cost
        )

        comfort_penalty = (
            cfg.comfort_weight * discomfort
        )

        violation_penalty = (
            cfg.violation_weight * violation
        )

        reward = -(
            energy_penalty
            + cost_penalty
            + comfort_penalty
            + violation_penalty
        )

        return {
            "reward": float(reward),
            "energy": float(energy),
            "electricity_cost": float(cost),
            "comfort_deviation": float(deviation),
            "discomfort": float(discomfort),
            "temperature_violation": bool(violation),
            "energy_penalty": float(energy_penalty),
            "cost_penalty": float(cost_penalty),
            "comfort_penalty": float(comfort_penalty),
            "violation_penalty": float(violation_penalty),
        }
