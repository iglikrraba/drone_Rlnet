"""Environment correctness tests for the morphing-wing attitude model."""

from __future__ import annotations

import numpy as np

from flight_env_3d import FlightEnv3D


def test_reset_seed_is_deterministic() -> None:
    env = FlightEnv3D()

    obs_1, _ = env.reset(seed=123)
    obs_2, _ = env.reset(seed=123)

    assert np.allclose(obs_1, obs_2)


def test_predict_next_state_matches_step_transition() -> None:
    env = FlightEnv3D(seed=7)
    observation, _ = env.reset(seed=7)
    action = np.array([0.2, -0.3, 0.1], dtype=np.float64)

    expected_state = env.predict_next_state(observation, action, env.step_count)
    next_observation, _, _, _, _ = env.step(action)

    assert np.allclose(expected_state, next_observation.astype(np.float64), atol=1e-7)


def test_set_state_validates_shape() -> None:
    env = FlightEnv3D()
    env.reset(seed=0)

    valid_state = np.zeros(6, dtype=np.float64)
    returned_observation = env.set_state(valid_state)

    assert returned_observation.shape == (6,)

    invalid_state = np.zeros(5, dtype=np.float64)
    try:
        env.set_state(invalid_state)
    except ValueError:
        pass
    else:
        raise AssertionError("set_state must raise ValueError for invalid state shape")
