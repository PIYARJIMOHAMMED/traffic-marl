# MAI-07 — Model and System Card

## Model
PPO-based Multi-Agent Reinforcement Learning controller.

## Intended use
Simulation-based adaptive traffic-signal control research.

## Inputs
120-dimensional graph-aware traffic state.

## Outputs
One binary signal-control action per intersection.

## Safety
Actions pass through a Safety Validator and Traffic Controller. Human governance can require approval.

## Training artifact
`ppo_marl_500k_final.zip`, trained for 500,000 timesteps.

## Evaluation
Ten traffic-demand scenarios including baseline, rush-hour, heavy-demand and stress conditions.

## Known limitations
Performance is simulation-dependent. The model should not be represented as validated for physical deployment. Statistical uncertainty needs stronger independent traffic realizations than the initial three-episode run.
