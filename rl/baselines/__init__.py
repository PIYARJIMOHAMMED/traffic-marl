"""
=========================================================
Traffic MARL Project

Module:
Baseline Policies

Description:
Baseline controllers used for comparison with
the Multi-Agent PPO traffic-light controller.

Policies:

1. Fixed-Time
2. Random
3. Queue-Based

These policies provide reproducible non-RL
reference points for the research evaluation.

Week:
6

Task:
6.1

Author:
Mohammed Piyarji
=========================================================
"""

from rl.baselines.fixed_time import FixedTimePolicy
from rl.baselines.random_policy import RandomPolicy
from rl.baselines.queue_based import QueueBasedPolicy


__all__ = [
    "FixedTimePolicy",
    "RandomPolicy",
    "QueueBasedPolicy",
]