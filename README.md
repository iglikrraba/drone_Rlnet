# Safe RL for 3D Attitude Control

I built this as a personal project to explore safe reinforcement learning for a simplified morphing-wing aircraft model. The goal was to keep the controller stable, add a lightweight safety check, and make the training pipeline easy to reproduce.

This is a simulation project, not a flight-ready controller.

## What’s in it

- A 6D rotational state: `[roll, pitch, yaw, p, q, r]`
- A time-varying diagonal inertia model to mimic morphing effects
- A bounded disturbance term
- PPO training with shielded evaluation
- A simple predictive safety shield
- Optional GP-based uncertainty estimation in the shield

## Why I made it

- To get practice with reinforcement learning in a control setting
- To learn how reward shaping changes behavior
- To compare a learned policy with a simple safety fallback
- To keep the code small enough that I could test and explain it

## Repo layout

- `flight_env_3d.py`: environment dynamics, reward, and limits
- `main_safe_rl_3d.py`: training, evaluation, plotting, and metrics export
- `train.py`: command-line entrypoint
- `drone_rlnet/config.py`: config dataclasses
- `tests/`: unit and pipeline tests

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

Full run:

```bash
./venv/bin/python train.py --total-timesteps 200000 --eval-steps 300 --output-root outputs
```

Smoke run:

```bash
./venv/bin/python train.py --total-timesteps 1024 --eval-steps 80 --output-root outputs_smoke
```

GP uncertainty smoke run:

```bash
./venv/bin/python train.py --total-timesteps 1024 --eval-steps 80 --output-root outputs_gp --use-gp-uncertainty --gp-min-samples 16 --gp-retrain-interval 8
```

Each run writes:

- figures/safe_rl_results.png
- metrics/safe_rl_metrics.json

## A sample result

Under one stress scenario (disturbance scale `1.8`, initial `roll=18deg`, `pitch=12deg`), one run produced:

- Final roll: `4.38deg`
- Final pitch: `1.23deg`
- Max roll during rollout: `20.35deg`
- Max pitch during rollout: `13.70deg`
- Shield interventions: `0`

These numbers vary with seed and training budget, so I treat them as one reference run rather than a promise.

## What I learned

- Reward scaling matters a lot.
- A stronger controller is not always a better controller.
- Longer training helps, but only up to a point.
- The safety shield is useful, but it also shows where the policy is still weak.

## Limitations

- Environment is simulation-only and intentionally simplified.
- No formal safety proof is provided.
- No hardware-in-the-loop or real flight validation yet.

## Future work

- Try a few more seeds and compare stability
- Tighten the final-pitch behavior without increasing oscillation
- Replace the current report-style outputs with a single concise results table if needed

## License

MIT (see `LICENSE`).
