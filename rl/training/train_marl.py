"""
=========================================================
Traffic MARL Project

Module:
MARL Training

Description:
Train the Multi-Agent PPO controller on the
4-intersection SUMO network.

Uses:
    PPO
    Gymnasium
    MLflow
    SUMO
    Checkpoints

=========================================================
"""

from pathlib import Path

import mlflow

from stable_baselines3 import PPO

from stable_baselines3.common.callbacks import (
    CheckpointCallback
)

from stable_baselines3.common.monitor import (
    Monitor
)

from rl.environments.config import (
    EnvironmentConfig
)

from rl.environments.traffic_env import (
    TrafficEnvironment
)


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(
    EnvironmentConfig.ROOT_DIR
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "rl"
    / "checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = (
    MODEL_DIR
    / EnvironmentConfig.MODEL_NAME
)


# =========================================================
# MLFLOW
# =========================================================

mlflow.set_tracking_uri(
    "http://localhost:5000"
)

mlflow.set_experiment(
    EnvironmentConfig.EXPERIMENT_NAME
)


# =========================================================
# ENVIRONMENT
# =========================================================

def create_environment():

    env = TrafficEnvironment()

    env = Monitor(
        env
    )

    return env


# =========================================================
# CHECKPOINT CALLBACK
# =========================================================

def create_checkpoint_callback():

    return CheckpointCallback(

        save_freq=5000,

        save_path=str(
            CHECKPOINT_DIR
        ),

        name_prefix="ppo_marl"

    )


# =========================================================
# PPO MODEL
# =========================================================

def create_model(env):

    model = PPO(

        policy=EnvironmentConfig.PPO_POLICY,

        env=env,

        learning_rate=(
            EnvironmentConfig.LEARNING_RATE
        ),

        gamma=(
            EnvironmentConfig.GAMMA
        ),

        batch_size=(
            EnvironmentConfig.BATCH_SIZE
        ),

        n_steps=(
            EnvironmentConfig.N_STEPS
        ),

        ent_coef=(
            EnvironmentConfig.ENT_COEF
        ),

        gae_lambda=(
            EnvironmentConfig.GAE_LAMBDA
        ),

        clip_range=(
            EnvironmentConfig.CLIP_RANGE
        ),

        verbose=1,

        tensorboard_log=str(
            PROJECT_ROOT / "tensorboard"
        )

    )

    return model

# =========================================================
# LOG TRAINING PARAMETERS
# =========================================================

def log_parameters():

    mlflow.log_param(
        "Algorithm",
        "PPO"
    )

    mlflow.log_param(
        "Policy",
        EnvironmentConfig.PPO_POLICY
    )

    mlflow.log_param(
        "Agents",
        EnvironmentConfig.NUMBER_OF_AGENTS
    )

    mlflow.log_param(
        "Traffic Lights",
        ",".join(
            EnvironmentConfig.TRAFFIC_LIGHT_IDS
        )
    )

    mlflow.log_param(
        "Observation Size",
        EnvironmentConfig.OBSERVATION_SIZE
    )

    mlflow.log_param(
        "Action Space",
        EnvironmentConfig.NUMBER_OF_ACTIONS
    )

    mlflow.log_param(
        "Learning Rate",
        EnvironmentConfig.LEARNING_RATE
    )

    mlflow.log_param(
        "Gamma",
        EnvironmentConfig.GAMMA
    )

    mlflow.log_param(
        "Batch Size",
        EnvironmentConfig.BATCH_SIZE
    )

    mlflow.log_param(
        "N Steps",
        EnvironmentConfig.N_STEPS
    )

    mlflow.log_param(
        "Entropy Coefficient",
        EnvironmentConfig.ENT_COEF
    )

    mlflow.log_param(
        "GAE Lambda",
        EnvironmentConfig.GAE_LAMBDA
    )

    mlflow.log_param(
        "Clip Range",
        EnvironmentConfig.CLIP_RANGE
    )

    mlflow.log_param(
        "Timesteps",
        EnvironmentConfig.TOTAL_TIMESTEPS
    )


# =========================================================
# TRAINING
# =========================================================

def train_model(
    model,
    checkpoint_callback
):

    print()
    print("=" * 70)
    print("Traffic MARL")
    print("=" * 70)

    print()
    print(
        "Neighbour-aware Multi-Agent PPO"
    )

    print()
    print(
        "Training Started"
    )

    print()

    with mlflow.start_run():

        # -------------------------------------------------
        # Parameters
        # -------------------------------------------------

        log_parameters()

        # -------------------------------------------------
        # Training
        # -------------------------------------------------

        model.learn(

            total_timesteps=(
                EnvironmentConfig.TOTAL_TIMESTEPS
            ),

            callback=(
                checkpoint_callback
            ),

            progress_bar=True

        )

        # -------------------------------------------------
        # Save model
        # -------------------------------------------------

        model.save(
            str(
                MODEL_PATH
            )
        )

        # -------------------------------------------------
        # Log model
        # -------------------------------------------------

        model_file = (
            str(MODEL_PATH)
            + ".zip"
        )

        mlflow.log_artifact(
            model_file
        )

        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        mlflow.log_metric(
            "Observation Dimension",
            model.observation_space.shape[0]
        )

        mlflow.log_metric(
            "Number Of Agents",
            EnvironmentConfig.NUMBER_OF_AGENTS
        )

        mlflow.log_metric(
            "Episode Length",
            EnvironmentConfig.MAX_EPISODE_STEPS
        )

        mlflow.log_metric(
            "Total Timesteps",
            EnvironmentConfig.TOTAL_TIMESTEPS
        )


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("=" * 70)
    print("TRAFFIC MARL - PPO TRAINING")
    print("=" * 70)

    print()
    print(
        "Project Root:"
    )

    print(
        PROJECT_ROOT
    )

    print()

    print(
        "Model Path:"
    )

    print(
        MODEL_PATH
    )

    print()

    print(
        "Observation Size:",
        EnvironmentConfig.OBSERVATION_SIZE
    )

    print(
        "Number Of Agents:",
        EnvironmentConfig.NUMBER_OF_AGENTS
    )

    print(
        "Traffic Lights:",
        EnvironmentConfig.TRAFFIC_LIGHT_IDS
    )

    print(
        "Actions:",
        EnvironmentConfig.NUMBER_OF_ACTIONS
    )

    print(
        "Configured Timesteps:",
        EnvironmentConfig.TOTAL_TIMESTEPS
    )

    print()

    env = None

    try:

        # -------------------------------------------------
        # Create environment
        # -------------------------------------------------

        env = create_environment()

        # -------------------------------------------------
        # Create callback
        # -------------------------------------------------

        checkpoint_callback = (
            create_checkpoint_callback()
        )

        # -------------------------------------------------
        # Create PPO
        # -------------------------------------------------

        model = create_model(
            env
        )

        # -------------------------------------------------
        # Train
        # -------------------------------------------------

        train_model(
            model,
            checkpoint_callback
        )

        # -------------------------------------------------
        # Finished
        # -------------------------------------------------

        print()
        print("=" * 70)
        print("Training Finished")
        print("=" * 70)

        print()

        print(
            "Model Saved:"
        )

        print(
            MODEL_PATH
        )

        print()

        print(
            "Observation Size:",
            env.observation_space.shape[0]
        )

        print(
            "Number Of Agents:",
            EnvironmentConfig.NUMBER_OF_AGENTS
        )

        print(
            "Traffic Lights:",
            EnvironmentConfig.TRAFFIC_LIGHT_IDS
        )

        print(
            "Training Timesteps:",
            EnvironmentConfig.TOTAL_TIMESTEPS
        )

        print()

        print("=" * 70)
        print(
            "Traffic MARL Training Complete"
        )
        print("=" * 70)

    finally:

        if env is not None:

            env.close()


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()