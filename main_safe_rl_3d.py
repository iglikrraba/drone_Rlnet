"""Training and shielded evaluation for safe RL attitude control.

This script trains a PPO policy on the morphing-wing attitude environment and
then evaluates it through a predictive safety shield before actions are applied
to the aircraft model.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from drone_rlnet.config import EvaluationConfig, RunConfig, ShieldConfig, TrainingConfig
from flight_env_3d import FlightEnv3D

try:
    import gpytorch
    import torch
except Exception:  # pragma: no cover - optional dependency path
    gpytorch = None
    torch = None


@dataclass
class ShieldDecision:
    action: np.ndarray
    intervened: bool
    reason: str
    fallback_predicted_unsafe: bool = False
    roll_uncertainty_std: float = 0.0
    pitch_uncertainty_std: float = 0.0
    violation_probability: float = 0.0
    safety_margin: float = 0.0


if gpytorch is not None:

    class _SingleOutputExactGP(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood) -> None:
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPDynamicsUncertaintyEstimator:
    """Online GP model for one-step roll/pitch residual uncertainty."""

    def __init__(self, env: FlightEnv3D, config: ShieldConfig) -> None:
        self.env = env
        self.enabled = bool(config.use_gp_uncertainty and gpytorch is not None and torch is not None)
        self.min_samples = int(config.gp_min_samples)
        self.retrain_interval = int(config.gp_retrain_interval)
        self.max_samples = int(config.gp_max_samples)
        self._feature_buffer: list[np.ndarray] = []
        self._target_buffer: list[np.ndarray] = []
        self._transitions_since_train = 0
        self._is_trained = False
        self._backend = "gpytorch" if self.enabled else "deterministic"
        self._roll_model = None
        self._roll_likelihood = None
        self._pitch_model = None
        self._pitch_likelihood = None

    def status(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "backend": self._backend,
            "is_trained": self._is_trained,
            "samples": len(self._feature_buffer),
            "min_samples": self.min_samples,
            "retrain_interval": self.retrain_interval,
        }

    def observe_transition(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        step_index: int,
        next_observation: np.ndarray,
    ) -> None:
        if not self.enabled:
            return

        obs = np.asarray(observation, dtype=np.float64)
        act = np.asarray(action, dtype=np.float64)
        next_obs = np.asarray(next_observation, dtype=np.float64)
        predicted = self.env.predict_next_state(obs, act, step_index)

        feature = np.concatenate([obs, np.clip(act, -1.0, 1.0)]).astype(np.float64)
        residual = np.array(
            [next_obs[0] - predicted[0], next_obs[1] - predicted[1]],
            dtype=np.float64,
        )

        self._feature_buffer.append(feature)
        self._target_buffer.append(residual)
        if len(self._feature_buffer) > self.max_samples:
            overflow = len(self._feature_buffer) - self.max_samples
            self._feature_buffer = self._feature_buffer[overflow:]
            self._target_buffer = self._target_buffer[overflow:]

        self._transitions_since_train += 1
        if len(self._feature_buffer) >= self.min_samples and self._transitions_since_train >= self.retrain_interval:
            self._fit_models()
            self._transitions_since_train = 0

    def predict_std(self, state: np.ndarray, action: np.ndarray) -> tuple[float, float]:
        if not self.enabled or not self._is_trained:
            return 0.0, 0.0

        feature = np.concatenate(
            [np.asarray(state, dtype=np.float64), np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)]
        ).astype(np.float64)
        test_x = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)

        self._roll_model.eval()
        self._roll_likelihood.eval()
        self._pitch_model.eval()
        self._pitch_likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            roll_pred = self._roll_likelihood(self._roll_model(test_x))
            pitch_pred = self._pitch_likelihood(self._pitch_model(test_x))

        roll_std = float(torch.sqrt(torch.clamp(roll_pred.variance, min=1e-12)).item())
        pitch_std = float(torch.sqrt(torch.clamp(pitch_pred.variance, min=1e-12)).item())
        return roll_std, pitch_std

    def _fit_models(self) -> None:
        train_x = torch.tensor(np.asarray(self._feature_buffer), dtype=torch.float32)
        targets = torch.tensor(np.asarray(self._target_buffer), dtype=torch.float32)

        self._roll_likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self._roll_model = _SingleOutputExactGP(train_x, targets[:, 0], self._roll_likelihood)

        self._pitch_likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self._pitch_model = _SingleOutputExactGP(train_x, targets[:, 1], self._pitch_likelihood)

        self._train_exact_gp(self._roll_model, self._roll_likelihood, train_x, targets[:, 0])
        self._train_exact_gp(self._pitch_model, self._pitch_likelihood, train_x, targets[:, 1])
        self._is_trained = True

    @staticmethod
    def _train_exact_gp(model, likelihood, train_x, train_y) -> None:
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.08)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()


class PredictiveSafetyShield:
    """Simplex-style safety filter with a predictive reachability check."""

    def __init__(
        self,
        env: FlightEnv3D,
        config: ShieldConfig,
    ) -> None:
        self.config = config
        self.env = env
        self.horizon_steps = int(config.horizon_steps)
        self.safety_factor = float(config.safety_factor)
        self.uncertainty_sigma_scale = float(config.uncertainty_sigma_scale)
        self.uncertainty_std_clip = float(config.uncertainty_std_clip_rad)
        self.roll_threshold = self.safety_factor * env.limits.roll_limit
        self.pitch_threshold = self.safety_factor * env.limits.pitch_limit
        self.uncertainty_estimator = GPDynamicsUncertaintyEstimator(env, config)

        self.kp_angles = np.array([7.5, 8.0, 3.0], dtype=np.float64)
        self.kd_rates = np.array([4.5, 4.5, 2.5], dtype=np.float64)

    def filter_action(
        self,
        observation: np.ndarray,
        proposed_action: np.ndarray,
        step_index: int,
    ) -> ShieldDecision:
        proposed_action = np.asarray(proposed_action, dtype=np.float64)
        proposed_action = np.clip(proposed_action, -1.0, 1.0)

        proposed_risk = self._evaluate_risk(observation, proposed_action, step_index)
        if proposed_risk["unsafe"]:
            fallback = self.fallback_action(observation, step_index)
            fallback_risk = self._evaluate_risk(observation, fallback, step_index)
            fallback_is_unsafe = bool(fallback_risk["unsafe"])
            reason = "fallback" if not fallback_is_unsafe else "fallback_predicted_unsafe"
            return ShieldDecision(
                action=fallback,
                intervened=True,
                reason=reason,
                fallback_predicted_unsafe=fallback_is_unsafe,
                roll_uncertainty_std=float(fallback_risk["roll_uncertainty_std"]),
                pitch_uncertainty_std=float(fallback_risk["pitch_uncertainty_std"]),
                violation_probability=float(fallback_risk["violation_probability"]),
                safety_margin=float(fallback_risk["safety_margin"]),
            )

        return ShieldDecision(
            action=proposed_action,
            intervened=False,
            reason="policy",
            roll_uncertainty_std=float(proposed_risk["roll_uncertainty_std"]),
            pitch_uncertainty_std=float(proposed_risk["pitch_uncertainty_std"]),
            violation_probability=float(proposed_risk["violation_probability"]),
            safety_margin=float(proposed_risk["safety_margin"]),
        )

    def observe_transition(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        step_index: int,
        next_observation: np.ndarray,
    ) -> None:
        self.uncertainty_estimator.observe_transition(
            observation=observation,
            action=action,
            step_index=step_index,
            next_observation=next_observation,
        )

    def uncertainty_status(self) -> dict[str, object]:
        return self.uncertainty_estimator.status()

    def fallback_action(self, observation: np.ndarray, step_index: int) -> np.ndarray:
        """Inverse-dynamics PD controller for aggressive attitude stabilization."""

        angles = np.asarray(observation[:3], dtype=np.float64)
        rates = np.asarray(observation[3:], dtype=np.float64)

        desired_angular_acceleration = -self.kp_angles * angles - self.kd_rates * rates
        inertia = self.env.current_inertia(step_index)
        coupling = np.cross(rates, inertia @ rates)
        torque_command = inertia @ desired_angular_acceleration + coupling
        normalized_action = torque_command / self.env.torque_limits
        return np.clip(normalized_action, -1.0, 1.0)

    def _evaluate_risk(self, observation: np.ndarray, action: np.ndarray, step_index: int) -> dict[str, float | bool]:
        predicted_state = np.asarray(observation, dtype=np.float64)
        max_roll_std = 0.0
        max_pitch_std = 0.0
        max_violation_probability = 0.0
        min_margin = float("inf")

        for offset in range(self.horizon_steps):
            predicted_state = self.env.predict_next_state(predicted_state, action, step_index + offset)
            roll = float(predicted_state[0])
            pitch = float(predicted_state[1])
            abs_roll = abs(roll)
            abs_pitch = abs(pitch)
            margin = min(self.roll_threshold - abs_roll, self.pitch_threshold - abs_pitch)
            min_margin = min(min_margin, margin)

            roll_std, pitch_std = self.uncertainty_estimator.predict_std(predicted_state, action)
            scaled_roll_std = min(self.uncertainty_sigma_scale * roll_std, self.uncertainty_std_clip)
            scaled_pitch_std = min(self.uncertainty_sigma_scale * pitch_std, self.uncertainty_std_clip)
            max_roll_std = max(max_roll_std, scaled_roll_std)
            max_pitch_std = max(max_pitch_std, scaled_pitch_std)

            violation_probability = self._combined_violation_probability(
                roll=roll,
                pitch=pitch,
                roll_std=scaled_roll_std,
                pitch_std=scaled_pitch_std,
            )
            max_violation_probability = max(max_violation_probability, violation_probability)

            if abs_roll + scaled_roll_std > self.roll_threshold or abs_pitch + scaled_pitch_std > self.pitch_threshold:
                return {
                    "unsafe": True,
                    "roll_uncertainty_std": max_roll_std,
                    "pitch_uncertainty_std": max_pitch_std,
                    "violation_probability": max_violation_probability,
                    "safety_margin": min_margin,
                }

        return {
            "unsafe": False,
            "roll_uncertainty_std": max_roll_std,
            "pitch_uncertainty_std": max_pitch_std,
            "violation_probability": max_violation_probability,
            "safety_margin": min_margin if np.isfinite(min_margin) else 0.0,
        }

    def _combined_violation_probability(
        self,
        roll: float,
        pitch: float,
        roll_std: float,
        pitch_std: float,
    ) -> float:
        p_roll = self._two_sided_tail_probability(roll, self.roll_threshold, roll_std)
        p_pitch = self._two_sided_tail_probability(pitch, self.pitch_threshold, pitch_std)
        return float(1.0 - (1.0 - p_roll) * (1.0 - p_pitch))

    @staticmethod
    def _normal_cdf(value: float) -> float:
        return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))

    def _two_sided_tail_probability(self, mean: float, threshold: float, std_dev: float) -> float:
        if std_dev <= 1e-9:
            return 1.0 if abs(mean) > threshold else 0.0
        z_upper = (threshold - mean) / std_dev
        z_lower = (-threshold - mean) / std_dev
        return float((1.0 - self._normal_cdf(z_upper)) + self._normal_cdf(z_lower))


@dataclass
class EvaluationResult:
    time_steps: list[int]
    roll_history_deg: list[float]
    pitch_history_deg: list[float]
    reward_history: list[float]
    intervention_steps: list[int]
    fallback_predicted_unsafe_steps: list[int]
    decision_reasons: list[str]
    roll_uncertainty_std_deg: list[float]
    pitch_uncertainty_std_deg: list[float]
    violation_probability_history: list[float]
    safety_margin_deg: list[float]
    action_delta_norm_history: list[float]
    terminated: bool
    truncated: bool


def train_policy(env: FlightEnv3D, config: TrainingConfig, model_path: Path) -> PPO:
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=config.seed,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        learning_rate=config.learning_rate,
        ent_coef=config.ent_coef,
    )
    model.learn(total_timesteps=config.total_timesteps)
    model.save(str(model_path.with_suffix("")))
    return model


def build_initial_state(config: EvaluationConfig) -> np.ndarray:
    return np.array(
        [
            np.deg2rad(config.initial_roll_deg),
            np.deg2rad(config.initial_pitch_deg),
            np.deg2rad(config.initial_yaw_deg),
            *config.initial_rates,
        ],
        dtype=np.float64,
    )


def evaluate_with_shield(
    env: FlightEnv3D,
    model: PPO,
    shield: PredictiveSafetyShield,
    config: EvaluationConfig,
) -> EvaluationResult:
    observation, _ = env.reset(seed=config.seed)

    env.disturbance_scale = config.disturbance_scale
    observation = env.set_state(build_initial_state(config))

    time_steps: list[int] = []
    roll_history: list[float] = []
    pitch_history: list[float] = []
    reward_history: list[float] = []
    intervention_steps: list[int] = []
    fallback_predicted_unsafe_steps: list[int] = []
    decision_reasons: list[str] = []
    roll_uncertainty_std_deg: list[float] = []
    pitch_uncertainty_std_deg: list[float] = []
    violation_probability_history: list[float] = []
    safety_margin_deg: list[float] = []
    action_delta_norm_history: list[float] = []

    terminated = False
    truncated = False

    for _ in range(config.max_steps):
        step_index = env.step_count
        raw_action, _ = model.predict(observation, deterministic=config.deterministic)
        raw_action = np.asarray(raw_action, dtype=np.float64)
        clipped_raw_action = np.clip(raw_action, -1.0, 1.0)
        observation_before_step = np.asarray(observation, dtype=np.float64)
        decision = shield.filter_action(observation_before_step, clipped_raw_action, step_index)

        if decision.intervened:
            intervention_steps.append(step_index)
            if decision.fallback_predicted_unsafe:
                fallback_predicted_unsafe_steps.append(step_index)

        observation, reward, terminated, truncated, _info = env.step(decision.action)
        shield.observe_transition(
            observation=observation_before_step,
            action=decision.action,
            step_index=step_index,
            next_observation=np.asarray(observation, dtype=np.float64),
        )

        time_steps.append(step_index)
        roll_history.append(float(np.degrees(observation[0])))
        pitch_history.append(float(np.degrees(observation[1])))
        reward_history.append(float(reward))
        decision_reasons.append(decision.reason)
        roll_uncertainty_std_deg.append(float(np.degrees(decision.roll_uncertainty_std)))
        pitch_uncertainty_std_deg.append(float(np.degrees(decision.pitch_uncertainty_std)))
        violation_probability_history.append(float(decision.violation_probability))
        safety_margin_deg.append(float(np.degrees(decision.safety_margin)))
        action_delta_norm_history.append(float(np.linalg.norm(decision.action - clipped_raw_action, ord=2)))

        if terminated or truncated:
            break

    return EvaluationResult(
        time_steps=time_steps,
        roll_history_deg=roll_history,
        pitch_history_deg=pitch_history,
        reward_history=reward_history,
        intervention_steps=intervention_steps,
        fallback_predicted_unsafe_steps=fallback_predicted_unsafe_steps,
        decision_reasons=decision_reasons,
        roll_uncertainty_std_deg=roll_uncertainty_std_deg,
        pitch_uncertainty_std_deg=pitch_uncertainty_std_deg,
        violation_probability_history=violation_probability_history,
        safety_margin_deg=safety_margin_deg,
        action_delta_norm_history=action_delta_norm_history,
        terminated=terminated,
        truncated=truncated,
    )


def create_run_directories(config: RunConfig, seed: int) -> dict[str, Path]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    run_dir = Path(config.output_root) / f"run_{timestamp}_seed{seed}"
    models_dir = run_dir / "models"
    figures_dir = run_dir / "figures"
    metrics_dir = run_dir / "metrics"

    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": run_dir,
        "models": models_dir,
        "figures": figures_dir,
        "metrics": metrics_dir,
    }


def save_metrics(
    metrics_path: Path,
    training_config: TrainingConfig,
    shield_config: ShieldConfig,
    evaluation_config: EvaluationConfig,
    shield: PredictiveSafetyShield,
    result: EvaluationResult,
) -> dict[str, object]:
    reason_counts = Counter(result.decision_reasons)
    metrics: dict[str, object] = {
        "steps": len(result.time_steps),
        "shield_interventions": len(result.intervention_steps),
        "shield_intervention_rate": (
            len(result.intervention_steps) / len(result.time_steps)
            if result.time_steps
            else 0.0
        ),
        "fallback_predicted_unsafe_count": len(result.fallback_predicted_unsafe_steps),
        "terminated": result.terminated,
        "truncated": result.truncated,
        "final_roll_deg": result.roll_history_deg[-1] if result.roll_history_deg else None,
        "final_pitch_deg": result.pitch_history_deg[-1] if result.pitch_history_deg else None,
        "max_abs_roll_deg": (
            float(np.max(np.abs(result.roll_history_deg))) if result.roll_history_deg else None
        ),
        "max_abs_pitch_deg": (
            float(np.max(np.abs(result.pitch_history_deg))) if result.pitch_history_deg else None
        ),
        "mean_reward": float(np.mean(result.reward_history)) if result.reward_history else None,
        "intervention_reason_counts": dict(reason_counts),
        "uncertainty": {
            "mean_roll_std_deg": (
                float(np.mean(result.roll_uncertainty_std_deg))
                if result.roll_uncertainty_std_deg
                else 0.0
            ),
            "mean_pitch_std_deg": (
                float(np.mean(result.pitch_uncertainty_std_deg))
                if result.pitch_uncertainty_std_deg
                else 0.0
            ),
            "max_roll_std_deg": (
                float(np.max(result.roll_uncertainty_std_deg))
                if result.roll_uncertainty_std_deg
                else 0.0
            ),
            "max_pitch_std_deg": (
                float(np.max(result.pitch_uncertainty_std_deg))
                if result.pitch_uncertainty_std_deg
                else 0.0
            ),
            "mean_violation_probability": (
                float(np.mean(result.violation_probability_history))
                if result.violation_probability_history
                else 0.0
            ),
            "max_violation_probability": (
                float(np.max(result.violation_probability_history))
                if result.violation_probability_history
                else 0.0
            ),
            "mean_safety_margin_deg": (
                float(np.mean(result.safety_margin_deg)) if result.safety_margin_deg else 0.0
            ),
            "min_safety_margin_deg": (
                float(np.min(result.safety_margin_deg)) if result.safety_margin_deg else 0.0
            ),
            "estimator": shield.uncertainty_status(),
        },
        "control_authority": {
            "mean_action_delta_l2": (
                float(np.mean(result.action_delta_norm_history))
                if result.action_delta_norm_history
                else 0.0
            ),
            "max_action_delta_l2": (
                float(np.max(result.action_delta_norm_history))
                if result.action_delta_norm_history
                else 0.0
            ),
        },
        "configs": {
            "training": asdict(training_config),
            "shield": asdict(shield_config),
            "evaluation": asdict(evaluation_config),
        },
    }

    with metrics_path.open("w", encoding="utf-8") as file_handle:
        json.dump(metrics, file_handle, indent=2)
    return metrics


def plot_results(result: EvaluationResult, figure_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        result.time_steps,
        result.roll_history_deg,
        label="Roll angle [deg]",
        linewidth=2.0,
        color="#1f77b4",
    )
    ax.plot(
        result.time_steps,
        result.pitch_history_deg,
        label="Pitch angle [deg]",
        linewidth=2.0,
        color="#2ca02c",
    )

    ax.axhline(45.0, color="red", linestyle="--", linewidth=1.2, alpha=0.9, label="Roll limit (+/-45 deg)")
    ax.axhline(-45.0, color="red", linestyle="--", linewidth=1.2, alpha=0.9, label="_nolegend_")
    ax.axhline(30.0, color="red", linestyle="--", linewidth=1.2, alpha=0.6, label="Pitch limit (+/-30 deg)")
    ax.axhline(-30.0, color="red", linestyle="--", linewidth=1.2, alpha=0.6, label="_nolegend_")

    first_intervention = True
    for step_index in result.intervention_steps:
        ax.axvline(
            step_index,
            color="orange",
            linestyle=":",
            linewidth=1.3,
            alpha=0.8,
            label="Shield intervention" if first_intervention else "_nolegend_",
        )
        first_intervention = False

    ax.set_title("Safe Reinforcement Learning for 3D Attitude Control of a Morphing Wing Aircraft")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Angle [deg]")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)


def plot_uncertainty_dashboard(result: EvaluationResult, figure_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    ax_decision, ax_uncertainty, ax_risk, ax_control = axes.flatten()

    color_map = {
        "policy": "#2ca02c",
        "fallback": "#ff7f0e",
        "fallback_predicted_unsafe": "#d62728",
    }
    decision_y = [1 if reason == "policy" else 0 for reason in result.decision_reasons]
    decision_colors = [color_map.get(reason, "#7f7f7f") for reason in result.decision_reasons]
    ax_decision.scatter(result.time_steps, decision_y, c=decision_colors, s=14, alpha=0.9)
    ax_decision.set_yticks([0, 1])
    ax_decision.set_yticklabels(["shield", "policy"])
    ax_decision.set_title("Decision Timeline")
    ax_decision.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)

    ax_uncertainty.plot(
        result.time_steps,
        result.roll_uncertainty_std_deg,
        linewidth=1.8,
        color="#1f77b4",
        label="Roll std [deg]",
    )
    ax_uncertainty.plot(
        result.time_steps,
        result.pitch_uncertainty_std_deg,
        linewidth=1.8,
        color="#9467bd",
        label="Pitch std [deg]",
    )
    ax_uncertainty.set_title("Uncertainty Envelope")
    ax_uncertainty.set_ylabel("Std dev [deg]")
    ax_uncertainty.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax_uncertainty.legend(loc="best")

    ax_risk.plot(
        result.time_steps,
        result.violation_probability_history,
        linewidth=1.8,
        color="#d62728",
        label="Violation probability",
    )
    ax_risk.plot(
        result.time_steps,
        result.safety_margin_deg,
        linewidth=1.8,
        color="#17becf",
        label="Safety margin [deg]",
    )
    ax_risk.axhline(0.0, color="#111111", linestyle=":", linewidth=1.2, alpha=0.7)
    ax_risk.set_title("Risk and Safety Margin")
    ax_risk.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax_risk.legend(loc="best")

    ax_control.plot(
        result.time_steps,
        result.action_delta_norm_history,
        linewidth=1.8,
        color="#8c564b",
        label="Action delta L2",
    )
    ax_control.plot(
        result.time_steps,
        result.reward_history,
        linewidth=1.2,
        color="#bcbd22",
        alpha=0.75,
        label="Reward",
    )
    ax_control.set_title("Control Authority and Reward")
    ax_control.set_xlabel("Time step")
    ax_control.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax_control.legend(loc="best")

    fig.suptitle("Shield + GP Uncertainty Dashboard", fontsize=13)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)


def save_posthoc_report(report_path: Path, metrics: dict[str, object], result: EvaluationResult) -> None:
    uncertainty = metrics.get("uncertainty", {})
    control = metrics.get("control_authority", {})
    reason_counts = metrics.get("intervention_reason_counts", {})

    lines = [
        "# Safe RL Shield Report",
        "",
        "## Summary",
        f"- Steps: {metrics.get('steps', 0)}",
        f"- Shield interventions: {metrics.get('shield_interventions', 0)}",
        f"- Intervention rate: {metrics.get('shield_intervention_rate', 0.0):.4f}",
        f"- Mean reward: {metrics.get('mean_reward', 0.0):.4f}",
        "",
        "## Uncertainty",
        f"- Mean roll std [deg]: {float(uncertainty.get('mean_roll_std_deg', 0.0)):.4f}",
        f"- Mean pitch std [deg]: {float(uncertainty.get('mean_pitch_std_deg', 0.0)):.4f}",
        f"- Max violation probability: {float(uncertainty.get('max_violation_probability', 0.0)):.4f}",
        f"- Min safety margin [deg]: {float(uncertainty.get('min_safety_margin_deg', 0.0)):.4f}",
        "",
        "## Control Authority",
        f"- Mean action delta L2: {float(control.get('mean_action_delta_l2', 0.0)):.4f}",
        f"- Max action delta L2: {float(control.get('max_action_delta_l2', 0.0)):.4f}",
        "",
        "## Decision Counts",
        f"- policy: {reason_counts.get('policy', 0)}",
        f"- fallback: {reason_counts.get('fallback', 0)}",
        f"- fallback_predicted_unsafe: {reason_counts.get('fallback_predicted_unsafe', 0)}",
        "",
        "## Notes",
        f"- Episode terminated: {result.terminated}",
        f"- Episode truncated: {result.truncated}",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")


def plot_training_dashboard(model: PPO, figure_path: Path) -> None:
    ep_buffer = list(model.ep_info_buffer)
    if not ep_buffer:
        return

    episode_indices = list(range(1, len(ep_buffer) + 1))
    rewards = [float(item["r"]) for item in ep_buffer if "r" in item]
    lengths = [float(item["l"]) for item in ep_buffer if "l" in item]
    if not rewards or not lengths:
        return

    fig, (ax_reward, ax_length) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax_reward.plot(episode_indices[: len(rewards)], rewards, color="#1f77b4", linewidth=1.8)
    ax_reward.set_ylabel("Episode reward")
    ax_reward.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax_reward.set_title("Training Dashboard")

    ax_length.plot(episode_indices[: len(lengths)], lengths, color="#ff7f0e", linewidth=1.8)
    ax_length.set_ylabel("Episode length")
    ax_length.set_xlabel("Episode index")
    ax_length.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    training_defaults = TrainingConfig()
    shield_defaults = ShieldConfig()
    evaluation_defaults = EvaluationConfig()
    run_defaults = RunConfig()

    parser = argparse.ArgumentParser(
        description="Train PPO and evaluate with a predictive safety shield.",
    )
    parser.add_argument("--seed", type=int, default=training_defaults.seed)
    parser.add_argument("--total-timesteps", type=int, default=training_defaults.total_timesteps)
    parser.add_argument("--n-steps", type=int, default=training_defaults.n_steps)
    parser.add_argument("--batch-size", type=int, default=training_defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=training_defaults.learning_rate)
    parser.add_argument("--horizon-steps", type=int, default=shield_defaults.horizon_steps)
    parser.add_argument("--safety-factor", type=float, default=shield_defaults.safety_factor)
    parser.add_argument("--use-gp-uncertainty", action="store_true")
    parser.add_argument(
        "--uncertainty-sigma-scale",
        type=float,
        default=shield_defaults.uncertainty_sigma_scale,
    )
    parser.add_argument(
        "--uncertainty-std-clip-rad",
        type=float,
        default=shield_defaults.uncertainty_std_clip_rad,
    )
    parser.add_argument("--gp-min-samples", type=int, default=shield_defaults.gp_min_samples)
    parser.add_argument("--gp-retrain-interval", type=int, default=shield_defaults.gp_retrain_interval)
    parser.add_argument("--gp-max-samples", type=int, default=shield_defaults.gp_max_samples)
    parser.add_argument("--eval-steps", type=int, default=evaluation_defaults.max_steps)
    parser.add_argument("--disturbance-scale", type=float, default=evaluation_defaults.disturbance_scale)
    parser.add_argument("--output-root", type=str, default=run_defaults.output_root)
    parser.add_argument("--disable-uncertainty-dashboard", action="store_true")
    parser.add_argument("--disable-posthoc-report", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--model-path", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    training_config = TrainingConfig(
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    shield_config = ShieldConfig(
        horizon_steps=args.horizon_steps,
        safety_factor=args.safety_factor,
        use_gp_uncertainty=args.use_gp_uncertainty,
        uncertainty_sigma_scale=args.uncertainty_sigma_scale,
        uncertainty_std_clip_rad=args.uncertainty_std_clip_rad,
        gp_min_samples=args.gp_min_samples,
        gp_retrain_interval=args.gp_retrain_interval,
        gp_max_samples=args.gp_max_samples,
    )
    evaluation_config = EvaluationConfig(
        seed=args.seed,
        max_steps=args.eval_steps,
        disturbance_scale=args.disturbance_scale,
    )
    run_config = RunConfig(output_root=args.output_root)

    paths = create_run_directories(run_config, training_config.seed)
    model_output_path = paths["models"] / training_config.model_filename

    env = FlightEnv3D(seed=training_config.seed)

    if args.skip_training:
        if args.model_path is None:
            raise ValueError("--model-path is required when --skip-training is used")
        model = PPO.load(args.model_path)
        model_output_path = Path(args.model_path)
    else:
        model = train_policy(env, training_config, model_output_path)

    training_figure_path = paths["figures"] / run_config.training_plot_filename
    if not args.skip_training:
        plot_training_dashboard(model, training_figure_path)

    shield = PredictiveSafetyShield(env, shield_config)
    result = evaluate_with_shield(env, model, shield, evaluation_config)

    figure_path = paths["figures"] / run_config.plot_filename
    plot_results(result, figure_path)

    uncertainty_figure_path = paths["figures"] / run_config.uncertainty_plot_filename
    if not args.disable_uncertainty_dashboard:
        plot_uncertainty_dashboard(result, uncertainty_figure_path)

    metrics_path = paths["metrics"] / run_config.metrics_filename
    metrics = save_metrics(
        metrics_path,
        training_config,
        shield_config,
        evaluation_config,
        shield,
        result,
    )

    report_path = paths["run_dir"] / run_config.report_filename
    if not args.disable_posthoc_report:
        save_posthoc_report(report_path, metrics, result)

    print(f"Run directory: {paths['run_dir']}")
    print(f"Model path: {model_output_path}")
    print(f"Plot path: {figure_path}")
    if not args.skip_training:
        print(f"Training dashboard path: {training_figure_path}")
    if not args.disable_uncertainty_dashboard:
        print(f"Uncertainty dashboard path: {uncertainty_figure_path}")
    print(f"Metrics path: {metrics_path}")
    if not args.disable_posthoc_report:
        print(f"Post-hoc report path: {report_path}")
    print(f"Shield interventions: {len(result.intervention_steps)}")
    print(f"Uncertainty estimator status: {shield.uncertainty_status()}")
    if result.intervention_steps:
        print(f"Intervention steps: {result.intervention_steps}")
    if result.fallback_predicted_unsafe_steps:
        print("Warning: fallback action remained near unsafe boundary at steps:")
        print(result.fallback_predicted_unsafe_steps)


if __name__ == "__main__":
    main()