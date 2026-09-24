"""
=========================================================
Traffic MARL Project

Module:
PPO Multi-Agent Training

Description:
Train the PPO agent on the
four-intersection SUMO network.

Week:
5

Task:
5.2

Author:
Mohammed Piyarji
=========================================================
"""

from stable_baselines3 import PPO

import mlflow

from rl.environments.config import EnvironmentConfig
from rl.environments.traffic_env import TrafficEnvironment


# =====================================================
# MLFLOW
# =====================================================

mlflow.set_tracking_uri(

    "http://localhost:5000"

)

mlflow.set_experiment(

    "Traffic MARL Multi-Agent PPO"

)

# =====================================================
# TRAINING SETTINGS
# =====================================================

MODEL_NAME = "ppo_multi_intersection"

TOTAL_TIMESTEPS = 50000

LEARNING_RATE = 3e-4

GAMMA = 0.99

BATCH_SIZE = 128

N_STEPS = 1024

# =====================================================

print()

print("=" * 70)

print("Traffic MARL")

print()

print("Multi-Agent PPO Training")

print("=" * 70)

print()

env = TrafficEnvironment()

with mlflow.start_run():

    print(

        "Creating PPO Model..."

    )

    model = PPO(

        policy="MlpPolicy",

        env=env,

        learning_rate=LEARNING_RATE,

        gamma=GAMMA,

        batch_size=BATCH_SIZE,

        n_steps=N_STEPS,

        verbose=1

    )

    print()

    print("=" * 70)

    print("Training Started")

    print("=" * 70)

    print()

    model.learn(

        total_timesteps=TOTAL_TIMESTEPS

    )

    print()

    print("=" * 70)

    print("Training Finished")

    print("=" * 70)

    print()

    model.save(

        MODEL_NAME

    )


    # =====================================================
    # LOG PARAMETERS
    # =====================================================

    mlflow.log_param(

        "Algorithm",

        "PPO"

    )

    mlflow.log_param(

        "Policy",

        "MlpPolicy"

    )

    mlflow.log_param(

        "Learning Rate",

        LEARNING_RATE

    )

    mlflow.log_param(

        "Gamma",

        GAMMA

    )

    mlflow.log_param(

        "Batch Size",

        BATCH_SIZE

    )

    mlflow.log_param(

        "Rollout Steps",

        N_STEPS

    )

    mlflow.log_param(

        "Training Timesteps",

        TOTAL_TIMESTEPS

    )

    mlflow.log_param(

        "Number Of Agents",

        EnvironmentConfig.NUMBER_OF_AGENTS

    )

    mlflow.log_param(

        "Traffic Lights",

        ",".join(

            EnvironmentConfig.TRAFFIC_LIGHT_IDS

        )

    )

    # =====================================================
    # LOG METRICS
    # =====================================================

    mlflow.log_metric(

        "Observation Size",

        env.observation_space.shape[0]

    )

    mlflow.log_metric(

        "Number Of Agents",

        EnvironmentConfig.NUMBER_OF_AGENTS

    )

    mlflow.log_metric(

        "Action Dimensions",

        len(

            env.action_space.nvec

        )

    )

    mlflow.log_metric(

        "Actions Per Agent",

        EnvironmentConfig.NUMBER_OF_ACTIONS

    )

    mlflow.log_metric(

        "Episode Length",

        EnvironmentConfig.MAX_EPISODE_STEPS

    )

    mlflow.log_metric(

        "Training Timesteps",

        TOTAL_TIMESTEPS

    )

    # =====================================================
    # SAVE MODEL
    # =====================================================

    mlflow.log_artifact(

        MODEL_NAME + ".zip"

    )

    env.close()

    print()

    print("=" * 70)

    print("Training Completed Successfully")

    print("=" * 70)

    print()

    print("Model")

    print(

        MODEL_NAME + ".zip"

    )

    print()

    print("Observation Size :", env.observation_space.shape[0])

    print(

        "Number Of Agents :", EnvironmentConfig.NUMBER_OF_AGENTS

    )

    print(

        "Traffic Lights   :",

        EnvironmentConfig.TRAFFIC_LIGHT_IDS

    )

    print("=" * 70)