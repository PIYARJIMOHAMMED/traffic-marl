"""
=========================================================
Traffic MARL Project

Module:
MARL Evaluation

Description:
Evaluate the trained Multi-Agent PPO model
on the four-intersection traffic network.

Computes

• Reward
• Waiting Time
• Queue Length
• Speed
• Throughput

Week:
5

Task:
5.3

Author:
Mohammed Piyarji
=========================================================
"""

from pathlib import Path

import pandas as pd
import numpy as np

from stable_baselines3 import PPO

from rl.environments.config import EnvironmentConfig
from rl.environments.traffic_env import TrafficEnvironment


# =====================================================
# MODEL
# =====================================================

MODEL_PATH = (

    EnvironmentConfig.ROOT_DIR

    / "models"

    / "ppo_multi_intersection.zip"

)

OUTPUT_DIR = (

    EnvironmentConfig.ROOT_DIR

    / "evaluation"

)

OUTPUT_DIR.mkdir(

    exist_ok=True

)

CSV_FILE = (

    OUTPUT_DIR

    / "marl_results.csv"

)

# =====================================================

print()

print("=" * 70)

print("Traffic MARL Evaluation")

print("=" * 70)

print()

env = TrafficEnvironment()

model = PPO.load(

    str(MODEL_PATH)

)

EPISODES = 10

episode_rewards = []

episode_waiting = []

episode_queue = []

episode_speed = []

episode_steps = []

print("Evaluating...")

print()


results = []

for episode in range(EPISODES):

    observation, _ = env.reset()

    terminated = False

    truncated = False

    total_reward = 0

    total_waiting = []

    total_queue = []

    total_speed = []

    steps = 0

    while not terminated and not truncated:

        action, _ = model.predict(

            observation,

            deterministic=True

        )

        observation, reward, terminated, truncated, info = (

            env.step(action)

        )

        total_reward += reward

        state = observation.reshape(

            EnvironmentConfig.NUMBER_OF_AGENTS,

            EnvironmentConfig.OBSERVATION_SIZE

        )

        waiting = state[:, 2].mean()

        queue = (

            state[:, 5]

            + state[:, 6]

            + state[:, 7]

            + state[:, 8]

        ).mean()

        speed = state[:, 1].mean()

        total_waiting.append(waiting)

        total_queue.append(queue)

        total_speed.append(speed)

        steps += 1

    episode_rewards.append(total_reward)

    episode_waiting.append(np.mean(total_waiting))

    episode_queue.append(np.mean(total_queue))

    episode_speed.append(np.mean(total_speed))

    episode_steps.append(steps)

    results.append({

        "Episode": episode + 1,

        "Reward": total_reward,

        "Average Waiting": np.mean(total_waiting),

        "Average Queue": np.mean(total_queue),

        "Average Speed": np.mean(total_speed),

        "Episode Length": steps

    })

df = pd.DataFrame(results)

df.to_csv(

    CSV_FILE,

    index=False

)

print()

print("=" * 70)

print("Research Statistics")

print("=" * 70)

print(

    "Average Reward :",

    np.mean(episode_rewards)

)

print(

    "Average Waiting :",

    np.mean(episode_waiting)

)

print(

    "Average Queue :",

    np.mean(episode_queue)

)

print(

    "Average Speed :",

    np.mean(episode_speed)

)

print(

    "Average Episode Length :",

    np.mean(episode_steps)

)

print()

print("CSV Saved To")

print(CSV_FILE)

print()

print("=" * 70)

print("Evaluation Finished")

print("=" * 70)

env.close()