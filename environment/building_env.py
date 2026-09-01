"""Gymnasium environment for energy-efficient building temperature control."""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .thermal_model import ThermalConfig, ThermalModel
from .reward import RewardConfig, RewardCalculator


class BuildingTemperatureEnv(gym.Env):
    """Single-zone building temperature-control environment.

    Action:
        Continuous HVAC control in [-1, 1].
        Negative = cooling.
        Zero = HVAC off.
        Positive = heating.

    Observation:
        [indoor_temperature,
         outdoor_temperature,
         comfort_setpoint,
         electricity_price,
         occupancy,
         hour_sin,
         hour_cos]
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        profiles,
        thermal_config: ThermalConfig | None = None,
        reward_config: RewardConfig | None = None,
        initial_indoor_temp_c: float = 22.0,
    ):
        super().__init__()

        required_columns = {
            "profile_id",
            "hour",
            "outdoor_temp_c",
            "occupancy",
            "electricity_price",
            "comfort_setpoint_c",
        }

        missing = required_columns - set(profiles.columns)

        if missing:
            raise ValueError(
                f"Profiles are missing required columns: {sorted(missing)}"
            )

        self.profiles = profiles.copy()

        # Validate that every profile represents one 24-hour episode.
        profile_lengths = self.profiles.groupby("profile_id").size()

        if not (profile_lengths == 24).all():
            raise ValueError(
                "Every building profile must contain exactly 24 hourly steps."
            )

        self.profile_ids = self.profiles["profile_id"].unique().tolist()

        self.thermal_model = ThermalModel(
            config=thermal_config or ThermalConfig()
        )

        self.reward_calculator = RewardCalculator(
            config=reward_config or RewardConfig()
        )

        self.initial_indoor_temp_c = float(initial_indoor_temp_c)

        # --------------------------------------------------------
        # Continuous HVAC action
        # --------------------------------------------------------
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # --------------------------------------------------------
        # Normalized observation
        #
        # Temperature values are scaled by 40.
        # Price is scaled by 1.
        # Occupancy already lies in [0, 1].
        # Cyclic time features lie in [-1, 1].
        # --------------------------------------------------------
        self.observation_space = spaces.Box(
            low=np.array(
                [-np.inf, -np.inf, -np.inf, 0.0, 0.0, -1.0, -1.0],
                dtype=np.float32,
            ),
            high=np.array(
                [np.inf, np.inf, np.inf, 2.0, 1.0, 1.0, 1.0],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self.current_profile = None
        self.current_profile_id = None
        self.current_step = 0
        self.indoor_temp_c = self.initial_indoor_temp_c

    def _get_observation(self) -> np.ndarray:
        """Construct the current normalized observation."""

        row = self.current_profile.iloc[self.current_step]

        hour = float(row["hour"])

        hour_sin = np.sin(2.0 * np.pi * hour / 24.0)
        hour_cos = np.cos(2.0 * np.pi * hour / 24.0)

        observation = np.array(
            [
                self.indoor_temp_c / 40.0,
                float(row["outdoor_temp_c"]) / 40.0,
                float(row["comfort_setpoint_c"]) / 40.0,
                float(row["electricity_price"]),
                float(row["occupancy"]),
                hour_sin,
                hour_cos,
            ],
            dtype=np.float32,
        )

        return observation

    def reset(self, *, seed=None, options=None):
        """Reset the environment to the beginning of a daily profile."""

        super().reset(seed=seed)

        if options is not None and "profile_id" in options:
            requested_id = options["profile_id"]

            if requested_id not in self.profile_ids:
                raise ValueError(
                    f"Unknown profile_id: {requested_id}"
                )

            profile_id = requested_id

        else:
            # Reproducible profile selection when seed is supplied.
            index = self.np_random.integers(
                0,
                len(self.profile_ids),
            )
            profile_id = self.profile_ids[index]

        self.current_profile_id = profile_id

        self.current_profile = (
            self.profiles[
                self.profiles["profile_id"] == profile_id
            ]
            .sort_values("hour")
            .reset_index(drop=True)
        )

        self.current_step = 0
        self.indoor_temp_c = self.initial_indoor_temp_c

        observation = self._get_observation()

        info = {
            "profile_id": self.current_profile_id,
            "hour": int(self.current_profile.iloc[0]["hour"]),
            "indoor_temperature_c": self.indoor_temp_c,
        }

        return observation, info

    def step(self, action):
        """Apply HVAC action and advance the building simulation."""

        action = np.asarray(action, dtype=np.float32).reshape(-1)

        if action.size != 1:
            raise ValueError("Action must contain exactly one value.")

        action_value = float(action[0])

        # Protect against tiny floating-point excursions.
        action_value = float(np.clip(action_value, -1.0, 1.0))

        row = self.current_profile.iloc[self.current_step]

        outdoor_temp = float(row["outdoor_temp_c"])
        occupancy = float(row["occupancy"])
        electricity_price = float(row["electricity_price"])
        setpoint = float(row["comfort_setpoint_c"])

        previous_indoor_temp = self.indoor_temp_c

        # --------------------------------------------------------
        # Thermal transition
        # --------------------------------------------------------
        self.indoor_temp_c = self.thermal_model.next_temperature(
            indoor_temp_c=self.indoor_temp_c,
            outdoor_temp_c=outdoor_temp,
            occupancy=occupancy,
            action=action_value,
        )

        # --------------------------------------------------------
        # Reward
        # --------------------------------------------------------
        reward_info = self.reward_calculator.calculate(
            indoor_temp_c=self.indoor_temp_c,
            setpoint_c=setpoint,
            action=action_value,
            electricity_price=electricity_price,
        )

        reward = reward_info["reward"]

        # Move to the next hourly state.
        self.current_step += 1

        terminated = False
        truncated = self.current_step >= 24

        # At the end of the 24-hour episode, keep the final
        # observation associated with the last profile row.
        if truncated:
            self.current_step = 23

        observation = self._get_observation()

        info = {
            "profile_id": self.current_profile_id,
            "hour": int(row["hour"]),
            "previous_indoor_temperature_c": previous_indoor_temp,
            "indoor_temperature_c": self.indoor_temp_c,
            "outdoor_temperature_c": outdoor_temp,
            "comfort_setpoint_c": setpoint,
            "occupancy": occupancy,
            "electricity_price": electricity_price,
            "action": action_value,
            "energy": reward_info["energy"],
            "electricity_cost": reward_info["electricity_cost"],
            "comfort_deviation": reward_info["comfort_deviation"],
            "discomfort": reward_info["discomfort"],
            "temperature_violation": reward_info["temperature_violation"],
            "energy_penalty": reward_info["energy_penalty"],
            "cost_penalty": reward_info["cost_penalty"],
            "comfort_penalty": reward_info["comfort_penalty"],
            "violation_penalty": reward_info["violation_penalty"],
        }

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )
