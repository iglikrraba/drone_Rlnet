"""Safety shield behavior tests."""

from __future__ import annotations

import numpy as np

from drone_rlnet.config import ShieldConfig
from flight_env_3d import FlightEnv3D
from main_safe_rl_3d import PredictiveSafetyShield


def test_shield_intervenes_near_boundary() -> None:
    env = FlightEnv3D(seed=5)
    env.reset(seed=5)

    near_boundary_state = np.array(
        [np.deg2rad(41.0), 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    observation = env.set_state(near_boundary_state)

    shield = PredictiveSafetyShield(env, ShieldConfig(horizon_steps=2, safety_factor=0.90))
    decision = shield.filter_action(observation, np.zeros(3, dtype=np.float64), env.step_count)

    assert decision.intervened
    assert decision.reason in {"fallback", "fallback_predicted_unsafe"}


def test_shield_returns_policy_action_when_safe() -> None:
    env = FlightEnv3D(seed=11)
    observation, _ = env.reset(seed=11)

    shield = PredictiveSafetyShield(env, ShieldConfig(horizon_steps=2, safety_factor=0.90))
    policy_action = np.array([0.05, -0.10, 0.02], dtype=np.float64)
    decision = shield.filter_action(observation, policy_action, env.step_count)

    assert decision.action.shape == (3,)
    assert decision.reason == "policy"
    assert not decision.intervened
    assert np.allclose(decision.action, policy_action)
    assert decision.roll_uncertainty_std >= 0.0
    assert decision.pitch_uncertainty_std >= 0.0
    assert 0.0 <= decision.violation_probability <= 1.0


def test_shield_clips_proposed_action_before_returning_policy_action() -> None:
    env = FlightEnv3D(seed=13)
    observation, _ = env.reset(seed=13)

    shield = PredictiveSafetyShield(env, ShieldConfig(horizon_steps=1, safety_factor=0.99))
    policy_action = np.array([2.0, -1.5, 0.2], dtype=np.float64)
    decision = shield.filter_action(observation, policy_action, env.step_count)

    assert decision.reason == "policy"
    assert np.all(decision.action <= 1.0)
    assert np.all(decision.action >= -1.0)


def test_shield_uses_fallback_action_when_policy_is_predicted_unsafe() -> None:
    env = FlightEnv3D(seed=17)
    env.reset(seed=17)
    observation = env.set_state(
        np.array([np.deg2rad(44.0), 0.0, 0.0, 0.1, 0.0, 0.0], dtype=np.float64)
    )

    shield = PredictiveSafetyShield(env, ShieldConfig(horizon_steps=2, safety_factor=0.90))
    policy_action = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    decision = shield.filter_action(observation, policy_action, env.step_count)

    fallback = shield.fallback_action(observation, env.step_count)
    assert decision.intervened
    assert decision.reason in {"fallback", "fallback_predicted_unsafe"}
    assert np.allclose(decision.action, fallback)
