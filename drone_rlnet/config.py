"""Configuration dataclasses for training and evaluation workflows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for PPO training."""

    seed: int = 42
    total_timesteps: int = 200_000
    n_steps: int = 1024
    batch_size: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 1e-4
    ent_coef: float = 0.01
    model_filename: str = "ppo_morphing_3d.zip"


@dataclass(frozen=True)
class ShieldConfig:
    """Settings for predictive safety filtering."""

    horizon_steps: int = 2
    safety_factor: float = 0.90
    use_gp_uncertainty: bool = False
    uncertainty_sigma_scale: float = 2.0
    uncertainty_std_clip_rad: float = 0.35
    gp_min_samples: int = 64
    gp_retrain_interval: int = 25
    gp_max_samples: int = 1500


@dataclass(frozen=True)
class EvaluationConfig:
    """Scenario configuration for shielded policy rollout."""

    seed: int = 42
    max_steps: int = 300
    disturbance_scale: float = 1.8
    deterministic: bool = True
    initial_roll_deg: float = 18.0
    initial_pitch_deg: float = 12.0
    initial_yaw_deg: float = 0.0
    initial_rates: tuple[float, float, float] = (0.12, 0.08, 0.0)


@dataclass(frozen=True)
class RunConfig:
    """Output naming and directory conventions for each run."""

    output_root: str = "outputs"
    plot_filename: str = "safe_rl_results.png"
    training_plot_filename: str = "training_dashboard.png"
    uncertainty_plot_filename: str = "shield_uncertainty_dashboard.png"
    metrics_filename: str = "safe_rl_metrics.json"
    report_filename: str = "safe_rl_report.md"
