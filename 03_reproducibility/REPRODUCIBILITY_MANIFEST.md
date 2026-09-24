# MAI-07 — Reproducibility Manifest

## Model artifact
`rl/checkpoints/ppo_marl_500k/ppo_marl_500k_final.zip`

## Training
- Algorithm: PPO
- Timesteps: 500,000
- Seed: 42
- n_steps: 256
- batch_size: 64
- learning_rate: 3e-4
- gamma: 0.99
- gae_lambda: 0.95
- clip_range: 0.2
- entropy coefficient: 0.01
- value coefficient: 0.5
- max_grad_norm: 0.5

## Environment
- 4 intersections: A, B, C, D
- 120-dimensional global observation
- MultiDiscrete([2,2,2,2]) action space
- 10 scenarios
- episode length: 300 steps

## Required artifact set
- source code
- scenario route/config files
- trained model
- requirements/dependency lock information
- evaluation scripts
- result CSV/JSON
- safety audit
- test logs
- README and user guide

## Reproduction principle
The final model must be evaluated from the saved artifact without retraining. Training and evaluation are separate stages.
