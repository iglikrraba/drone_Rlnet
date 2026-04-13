"""Pipeline-oriented tests for outputs, metrics, and directory helpers."""

from __future__ import annotations

import json

import numpy as np

from drone_rlnet.config import EvaluationConfig, RunConfig, ShieldConfig, TrainingConfig
from flight_env_3d import FlightEnv3D
from main_safe_rl_3d import (
    PredictiveSafetyShield,
    create_run_directories,
    evaluate_with_shield,
    plot_uncertainty_dashboard,
    plot_results,
    save_posthoc_report,
    save_metrics,
)


class _ZeroPolicy:
    def predict(self, observation: np.ndarray, deterministic: bool = True):
        del observation, deterministic
        return np.zeros(3, dtype=np.float64), None


def test_create_run_directories_builds_expected_tree(tmp_path) -> None:
    config = RunConfig(output_root=str(tmp_path))
    paths = create_run_directories(config, seed=7)

    assert paths["run_dir"].is_dir()
    assert paths["models"].is_dir()
    assert paths["figures"].is_dir()
    assert paths["metrics"].is_dir()
    assert "seed7" in paths["run_dir"].name


def test_evaluation_outputs_and_metrics_are_written(tmp_path) -> None:
    env = FlightEnv3D(seed=3)
    shield = PredictiveSafetyShield(env, ShieldConfig(horizon_steps=2, safety_factor=0.90))
    evaluation_config = EvaluationConfig(
        seed=3,
        max_steps=12,
        disturbance_scale=0.0,
        initial_roll_deg=6.0,
        initial_pitch_deg=4.0,
        initial_yaw_deg=0.0,
        initial_rates=(0.0, 0.0, 0.0),
    )

    result = evaluate_with_shield(env, _ZeroPolicy(), shield, evaluation_config)

    figure_path = tmp_path / "safe_rl_results.png"
    uncertainty_figure_path = tmp_path / "shield_uncertainty_dashboard.png"
    metrics_path = tmp_path / "safe_rl_metrics.json"
    report_path = tmp_path / "safe_rl_report.md"
    plot_results(result, figure_path)
    plot_uncertainty_dashboard(result, uncertainty_figure_path)
    metrics = save_metrics(
        metrics_path=metrics_path,
        training_config=TrainingConfig(total_timesteps=128),
        shield_config=ShieldConfig(horizon_steps=2, safety_factor=0.90),
        evaluation_config=evaluation_config,
        shield=shield,
        result=result,
    )
    save_posthoc_report(report_path, metrics, result)

    assert result.time_steps
    assert len(result.roll_history_deg) == len(result.time_steps)
    assert len(result.pitch_history_deg) == len(result.time_steps)
    assert figure_path.is_file()
    assert figure_path.stat().st_size > 0
    assert uncertainty_figure_path.is_file()
    assert uncertainty_figure_path.stat().st_size > 0
    assert metrics_path.is_file()
    assert report_path.is_file()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["steps"] == len(result.time_steps)
    assert metrics["shield_interventions"] == len(result.intervention_steps)
    assert "uncertainty" in metrics
    assert "control_authority" in metrics
    assert "configs" in metrics
    assert metrics["configs"]["evaluation"]["seed"] == evaluation_config.seed
