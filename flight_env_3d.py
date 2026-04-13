"""3D attitude-control environment for a morphing-wing aircraft.

The model uses rigid-body rotational dynamics with a time-varying inertia matrix
to represent morphing-wing effects. The state is the full 6D attitude state:
Euler angles and body rates.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class FlightLimits:
    roll_limit: float = 0.7853981633974483  # 45 deg
    pitch_limit: float = 0.5235987755982988  # 30 deg


class FlightEnv3D(gym.Env):
    """Custom Gymnasium environment for 3D attitude stabilization."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        dt: float = 0.05,
        max_episode_steps: int = 400,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.dt = float(dt)
        self.max_episode_steps = int(max_episode_steps)
        self.limits = FlightLimits()

        self.base_inertia = np.array([0.45, 0.55, 0.65], dtype=np.float64)
        self.morph_amplitude = 0.25
        self.morph_frequency = 0.35
        self.disturbance_scale = 1.0
        self.torque_limits = np.array([0.25, 0.25, 0.18], dtype=np.float64)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.array(
                [-np.pi, -np.pi, -np.pi, -10.0, -10.0, -10.0], dtype=np.float32
            ),
            high=np.array(
                [np.pi, np.pi, np.pi, 10.0, 10.0, 10.0], dtype=np.float32
            ),
            dtype=np.float32,
        )

        self._initial_seed = seed
        self.np_random: np.random.Generator | None = None
        self.state = np.zeros(6, dtype=np.float64)
        self.step_count = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._initial_seed = seed
            self.np_random = np.random.default_rng(seed)
        elif self.np_random is None:
            self.np_random = np.random.default_rng(self._initial_seed)

        angle_noise = self.np_random.uniform(low=-0.08, high=0.08, size=3)
        rate_noise = self.np_random.uniform(low=-0.03, high=0.03, size=3)
        self.state = np.concatenate([angle_noise, rate_noise]).astype(np.float64)
        self.step_count = 0

        return self._get_observation(), {}

    def set_state(self, state: np.ndarray) -> np.ndarray:
        """Set the internal state with basic shape validation."""

        state = np.asarray(state, dtype=np.float64)
        if state.shape != (6,):
            raise ValueError(f"Expected state shape (6,), got {state.shape}")
        self.state = state.copy()
        return self._get_observation()

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        transition_step = self.step_count
        next_state = self._propagate_state(self.state, action, transition_step)
        self.state = next_state
        self.step_count += 1

        roll, pitch, yaw, p, q, r = self.state
        terminated = bool(
            abs(roll) > self.limits.roll_limit or abs(pitch) > self.limits.pitch_limit
        )
        truncated = bool(self.step_count >= self.max_episode_steps)

        # Improved reward function: stronger penalties for attitude deviation
        angle_cost = 30.0 * roll**2 + 40.0 * pitch**2 + 1.0 * yaw**2
        rate_cost = 0.8 * (p**2 + q**2 + r**2)
        control_cost = 0.02 * float(np.sum(action**2))
        reward = -(angle_cost + rate_cost + control_cost)

        if terminated:
            reward = -250.0

        info = {
            "step_index": transition_step,
            "inertia": self.current_inertia(transition_step).copy(),
            "wind_disturbance": self.wind_disturbance(transition_step).copy(),
            "terminated_out_of_envelope": terminated,
            "truncated_time_limit": truncated,
        }

        return self._get_observation(), float(reward), terminated, truncated, info

    def render(self):
        return None

    def current_inertia(self, step_index: int) -> np.ndarray:
        """Return the time-varying diagonal inertia matrix."""

        time = step_index * self.dt
        morph_signal = np.sin(self.morph_frequency * time)

        ixx = self.base_inertia[0] * (1.0 + self.morph_amplitude * morph_signal)
        iyy = self.base_inertia[1] * (1.0 - 0.5 * self.morph_amplitude * morph_signal)
        izz = self.base_inertia[2]

        ixx = max(ixx, 0.15)
        iyy = max(iyy, 0.15)
        return np.diag([ixx, iyy, izz]).astype(np.float64)

    def wind_disturbance(self, step_index: int) -> np.ndarray:
        """Deterministic, bounded disturbance used to emulate atmospheric effects."""

        time = step_index * self.dt
        return np.array(
            [
                0.010 * np.sin(0.70 * time),
                0.012 * np.cos(0.90 * time + 0.4),
                0.008 * np.sin(0.50 * time + 0.2),
            ],
            dtype=np.float64,
        ) * self.disturbance_scale

    def predict_next_state(
        self,
        state: np.ndarray,
        action: np.ndarray,
        step_index: int,
    ) -> np.ndarray:
        """Predict the next state from an arbitrary state-action pair."""

        return self._propagate_state(
            np.asarray(state, dtype=np.float64),
            np.asarray(action, dtype=np.float64),
            step_index,
        )

    def predict_state_n_steps(
        self,
        state: np.ndarray,
        action: np.ndarray,
        step_index: int,
        steps_ahead: int,
    ) -> np.ndarray:
        predicted_state = np.asarray(state, dtype=np.float64)
        for offset in range(steps_ahead):
            predicted_state = self._propagate_state(predicted_state, action, step_index + offset)
        return predicted_state

    def _get_observation(self) -> np.ndarray:
        return self.state.astype(np.float32)

    def _propagate_state(
        self,
        state: np.ndarray,
        action: np.ndarray,
        step_index: int,
    ) -> np.ndarray:
        angles = state[:3].copy()
        rates = state[3:].copy()

        inertia = self.current_inertia(step_index)
        angular_momentum = inertia @ rates
        disturbance = self.wind_disturbance(step_index)
        torque = np.clip(action, -1.0, 1.0) * self.torque_limits

        rate_dot = np.linalg.solve(
            inertia,
            torque + disturbance - np.cross(rates, angular_momentum),
        )
        rates_next = rates + self.dt * rate_dot

        roll, pitch, yaw = angles
        p, q, r = rates
        cos_pitch = np.cos(pitch)
        cos_pitch = np.clip(cos_pitch, 1e-4, None)
        tan_pitch = np.tan(pitch)

        angle_dot = np.array(
            [
                p + q * np.sin(roll) * tan_pitch + r * np.cos(roll) * tan_pitch,
                q * np.cos(roll) - r * np.sin(roll),
                q * np.sin(roll) / cos_pitch + r * np.cos(roll) / cos_pitch,
            ],
            dtype=np.float64,
        )
        angles_next = angles + self.dt * angle_dot

        angles_next[0] = self._wrap_to_pi(angles_next[0])
        angles_next[1] = float(np.clip(angles_next[1], -1.45, 1.45))
        angles_next[2] = self._wrap_to_pi(angles_next[2])

        next_state = np.concatenate([angles_next, rates_next]).astype(np.float64)
        return next_state

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)