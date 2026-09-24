# MAI-07 — Environment and Runtime Specification

## Core runtime
- Windows development environment
- Python virtual environment
- SUMO + TraCI
- Gymnasium
- Stable-Baselines3
- PyTorch

## Environment contract
Observation: 120-D graph-aware vector.

Action: `MultiDiscrete([2,2,2,2])`.

Intersections:
A, B, C, D.

Graph:
A-B, A-C, B-D, C-D.

Safety timing:
- minimum green: 10 s
- yellow: 3 s

Evaluation scenarios:
baseline, low_demand, balanced, morning_rush, evening_rush, high_demand, north_south_heavy, east_west_heavy, random, stress_test.
