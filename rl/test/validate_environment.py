"""
=========================================================
Traffic MARL Project

Module:
Environment Validation

Description:
Checks whether the Gymnasium environment
follows Stable-Baselines3 requirements.

Author:
Mohammed Piyarji

Week:
4

Task:
4.2
=========================================================
"""

from stable_baselines3.common.env_checker import check_env

from rl.environments.traffic_env import TrafficEnvironment


print("=" * 70)
print("Environment Validation")
print("=" * 70)

env = TrafficEnvironment()

check_env(
    env,
    warn=True
)

print()

print("=" * 70)
print("Environment Passed Validation")
print("=" * 70)

env.close()