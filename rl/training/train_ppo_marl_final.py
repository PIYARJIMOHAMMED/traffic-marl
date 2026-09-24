from pathlib import Path
import sys
import os

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# IMPORTS
# ============================================================

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

from rl.environments.traffic_env import TrafficEnvironment


# ============================================================
# CONFIGURATION
# ============================================================

TOTAL_TIMESTEPS = 150_000

N_STEPS = 256

BATCH_SIZE = 64

LEARNING_RATE = 3e-4

GAMMA = 0.99

GAE_LAMBDA = 0.95

CLIP_RANGE = 0.2

ENT_COEF = 0.01

VF_COEF = 0.5

MAX_GRAD_NORM = 0.5

SEED = 42

SCENARIO = "baseline"

# ============================================================
# DIRECTORIES
# ============================================================

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "rl"
    / "checkpoints"
    / "ppo_marl_120d"
)

LOG_DIR = (
    PROJECT_ROOT
    / "rl"
    / "logs"
    / "ppo_marl_120d"
)

BEST_MODEL_DIR = (
    PROJECT_ROOT
    / "rl"
    / "checkpoints"
    / "ppo_marl_120d_best"
)

FINAL_MODEL_PATH = (
    PROJECT_ROOT
    / "rl"
    / "checkpoints"
    / "ppo_marl_120d_final"
)


# ============================================================
# DIRECTORY SETUP
# ============================================================

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# ENVIRONMENT FACTORY
# ============================================================

def make_environment():

    env = TrafficEnvironment(
        scenario_name=SCENARIO
    )

    env = Monitor(env)

    return env


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    print("=" * 70)
    print("FINAL PPO-MARL TRAINING")
    print("=" * 70)

    print(f"Project root:       {PROJECT_ROOT}")
    print(f"Scenario:           {SCENARIO}")
    print(f"Timesteps:          {TOTAL_TIMESTEPS:,}")
    print(f"Observation target: 120")
    print(f"Seed:               {SEED}")
    print("=" * 70)

    # --------------------------------------------------------
    # Create training environment
    # --------------------------------------------------------

    train_env = make_environment()

    print("\nTraining environment:")
    print("Observation space:", train_env.observation_space)
    print("Action space:", train_env.action_space)

    # --------------------------------------------------------
    # HARD VALIDATION
    # --------------------------------------------------------

    if train_env.observation_space.shape != (120,):
        train_env.close()

        raise ValueError(
            "FINAL TRAINING STOPPED: "
            f"Expected observation shape (120,), "
            f"received {train_env.observation_space.shape}"
        )

    expected_actions = np.array(
        [2, 2, 2, 2]
    )

    actual_actions = np.asarray(
        train_env.action_space.nvec
    )

    if not np.array_equal(
        actual_actions,
        expected_actions
    ):
        train_env.close()

        raise ValueError(
            "FINAL TRAINING STOPPED: "
            f"Expected MultiDiscrete([2,2,2,2]), "
            f"received {train_env.action_space}"
        )

    # --------------------------------------------------------
    # Test reset before expensive training
    # --------------------------------------------------------

    observation, info = train_env.reset(
        seed=SEED
    )

    print(
        "\nInitial observation shape:",
        observation.shape
    )

    if observation.shape != (120,):
        train_env.close()

        raise ValueError(
            "Reset returned incorrect observation shape: "
            f"{observation.shape}"
        )

    # --------------------------------------------------------
    # Test one action
    # --------------------------------------------------------

    test_action = np.array(
        [0, 0, 0, 0],
        dtype=np.int64
    )

    next_observation, reward, terminated, truncated, info = (
        train_env.step(test_action)
    )

    print(
        "Test observation shape:",
        next_observation.shape
    )

    print(
        "Test reward:",
        float(reward)
    )

    if next_observation.shape != (120,):
        train_env.close()

        raise ValueError(
            "Step returned incorrect observation shape: "
            f"{next_observation.shape}"
        )

    if not np.isfinite(float(reward)):
        train_env.close()

        raise ValueError(
            f"Invalid reward detected: {reward}"
        )

    print(
        "Environment validation: PASSED"
    )

    # --------------------------------------------------------
    # Reset after validation
    # --------------------------------------------------------

    train_env.reset(
        seed=SEED
    )

    # --------------------------------------------------------
    # PPO model
    # --------------------------------------------------------

    print("\nCreating PPO model...")

    model = PPO(
        policy="MlpPolicy",
        env=train_env,

        learning_rate=LEARNING_RATE,

        n_steps=N_STEPS,

        batch_size=BATCH_SIZE,

        gamma=GAMMA,

        gae_lambda=GAE_LAMBDA,

        clip_range=CLIP_RANGE,

        ent_coef=ENT_COEF,

        vf_coef=VF_COEF,

        max_grad_norm=MAX_GRAD_NORM,

        seed=SEED,

        verbose=1,

        tensorboard_log=str(LOG_DIR),

        device="auto",
    )

    print("\nPPO model created.")

    print(
        "Model observation space:",
        model.observation_space
    )

    print(
        "Model action space:",
        model.action_space
    )

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=str(CHECKPOINT_DIR),
        name_prefix="ppo_marl_120d",
    )

    eval_env = make_environment()

    eval_callback = EvalCallback(
        eval_env,

        best_model_save_path=str(
            BEST_MODEL_DIR
        ),

        log_path=str(
            LOG_DIR
        ),

        eval_freq=10_000,

        n_eval_episodes=3,

        deterministic=True,

        render=False,
    )

    callbacks = CallbackList(
        [
            checkpoint_callback,
            eval_callback,
        ]
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("STARTING 120-D PPO-MARL TRAINING")
    print("=" * 70)

    try:

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,

            callback=callbacks,

            progress_bar=True,

            reset_num_timesteps=True,

            tb_log_name="ppo_marl_120d_final",
        )

    finally:

        # Always close environments cleanly.
        eval_env.close()

        train_env.close()

    # --------------------------------------------------------
    # SAVE FINAL MODEL
    # --------------------------------------------------------

    model.save(
        str(FINAL_MODEL_PATH)
    )

    print("\n")
    print("=" * 70)
    print("FINAL PPO-MARL TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Final model:"
    )

    print(
        FINAL_MODEL_PATH
    )

    print(
        "\nCheckpoints:"
    )

    print(
        CHECKPOINT_DIR
    )

    print(
        "\nBest model:"
    )

    print(
        BEST_MODEL_DIR
    )

    print("=" * 70)


if __name__ == "__main__":
    main()