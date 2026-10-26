from pathlib import Path
import sys
import os


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from pathlib import Path
import sys
import random

import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO

from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    CallbackList,
)

from stable_baselines3.common.monitor import Monitor


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# =========================================================
# IMPORT PROJECT ENVIRONMENT
# =========================================================

from rl.environments.traffic_env import TrafficEnvironment
from rl.environments.config import EnvironmentConfig


# =========================================================
# TRAINING SETTINGS
# =========================================================

TOTAL_TIMESTEPS = 500_000

SEED = 42


# =========================================================
# PPO SETTINGS
# =========================================================
#
# These remain aligned with the existing PPO training
# configuration.
#
# Only the total training budget is increased.
# =========================================================

N_STEPS = 256

BATCH_SIZE = 64

LEARNING_RATE = 3e-4

GAMMA = 0.99

GAE_LAMBDA = 0.95

CLIP_RANGE = 0.2

ENT_COEF = 0.01

VF_COEF = 0.5

MAX_GRAD_NORM = 0.5


# =========================================================
# SCENARIOS
# =========================================================

SCENARIOS = [
    "baseline",
    "low_demand",
    "balanced",
    "morning_rush",
    "evening_rush",
    "high_demand",
    "north_south_heavy",
    "east_west_heavy",
    "random",
    "stress_test",
]


# =========================================================
# OUTPUT DIRECTORIES
# =========================================================

TRAINING_ROOT = (
    PROJECT_ROOT
    / "rl"
    / "checkpoints"
    / "ppo_marl_500k"
)

CHECKPOINT_DIR = (
    TRAINING_ROOT
    / "checkpoints"
)

BEST_DIR = (
    TRAINING_ROOT
    / "best"
)

LOG_DIR = (
    PROJECT_ROOT
    / "rl"
    / "logs"
    / "ppo_marl_500k"
)

FINAL_MODEL_PATH = (
    TRAINING_ROOT
    / "ppo_marl_500k_final"
)


# =========================================================
# CREATE DIRECTORIES
# =========================================================

TRAINING_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# REPRODUCIBILITY
# =========================================================

def set_global_seed(seed):
    """
    Set Python and NumPy random seeds.
    """

    random.seed(seed)

    np.random.seed(seed)


# =========================================================
# SCENARIO ROTATION ENVIRONMENT
# =========================================================

class ScenarioRotationEnv(gym.Wrapper):
    """
    Wrapper that changes the scenario BEFORE every
    new episode reset.

    This is important.

    Stable-Baselines3 automatically resets an environment
    after an episode terminates.

    Therefore scenario selection must happen inside reset()
    rather than inside an SB3 callback.

    Episode order:

        1  baseline
        2  low_demand
        3  balanced
        4  morning_rush
        5  evening_rush
        6  high_demand
        7  north_south_heavy
        8  east_west_heavy
        9  random
        10 stress_test
        11 baseline
        ...

    This guarantees that the next episode really starts
    with the intended scenario.
    """

    def __init__(
        self,
        env,
        scenarios,
    ):

        super().__init__(
            env
        )

        self.scenarios = list(
            scenarios
        )

        if not self.scenarios:
            raise ValueError(
                "Scenario list cannot be empty."
            )

        self.scenario_index = 0

        self.episode_count = 0

        self.first_reset = True

        # Exposed through SB3 VecEnv.get_attr() for callbacks/logging.
        self.scenario_name = self.scenarios[0]

        # Keep a direct accessor as well; useful when debugging wrappers.
        self.get_scenario = lambda: self.scenario_name

    # -----------------------------------------------------
    # RESET
    # -----------------------------------------------------

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):

        # -------------------------------------------------
        # First reset
        # -------------------------------------------------

        if self.first_reset:

            self.first_reset = False

        else:

            self.scenario_index = (
                self.scenario_index + 1
            ) % len(self.scenarios)

        # -------------------------------------------------
        # Select scenario
        # -------------------------------------------------

        scenario = (
            self.scenarios[
                self.scenario_index
            ]
        )

        # Keep the active scenario available to SB3 callbacks.
        self.scenario_name = scenario

        # -------------------------------------------------
        # Change scenario BEFORE reset
        # -------------------------------------------------

        self.env.set_scenario(
            scenario
        )

        # -------------------------------------------------
        # Episode count
        # -------------------------------------------------

        self.episode_count += 1

        # -------------------------------------------------
        # Informative training output
        # -------------------------------------------------

        print()
        print(
            "=" * 70
        )

        print(
            "STARTING TRAINING EPISODE"
        )

        print(
            f"Episode          : "
            f"{self.episode_count}"
        )

        print(
            f"Scenario         : "
            f"{scenario}"
        )

        print(
            f"Scenario position: "
            f"{self.scenario_index + 1}/"
            f"{len(self.scenarios)}"
        )

        print(
            f"Training target   : "
            f"{TOTAL_TIMESTEPS:,} timesteps"
        )

        print(
            "=" * 70
        )

        # -------------------------------------------------
        # Reset underlying SUMO environment
        # -------------------------------------------------

        return self.env.reset(
            seed=seed,
            options=options,
        )


# =========================================================
# TRAINING PROGRESS CALLBACK
# =========================================================

class TrainingProgressCallback(
    BaseCallback
):
    """
    Prints useful progress information without creating
    another SUMO environment.

    This avoids TraCI conflicts during training.
    """

    def __init__(
        self,
        report_frequency=10_000,
        verbose=1,
    ):

        super().__init__(
            verbose
        )

        self.report_frequency = (
            report_frequency
        )

        self.last_report = 0

    def _on_step(self):

        current = (
            self.num_timesteps
        )

        if (
            current - self.last_report
            >= self.report_frequency
        ):

            self.last_report = current

            progress = (
                current
                / TOTAL_TIMESTEPS
            ) * 100.0

            print()
            print(
                "-" * 70
            )

            print(
                "PPO-MARL TRAINING PROGRESS"
            )

            print(
                f"Timesteps : "
                f"{current:,} / "
                f"{TOTAL_TIMESTEPS:,}"
            )

            print(
                f"Progress  : "
                f"{progress:.2f}%"
            )

            # SB3 wraps the Gym environment in Monitor/DummyVecEnv.
            # get_attr() safely retrieves the attribute through wrappers.
            try:
                current_scenario = self.training_env.get_attr(
                    "scenario_name"
                )[0]
            except Exception:
                current_scenario = "unknown"

            print(
                f"Current scenario: "
                f"{current_scenario}"
            )

            print(
                "-" * 70
            )

        return True


# =========================================================
# ENVIRONMENT VALIDATION
# =========================================================

def validate_environment(
    env
):
    """
    Validate the existing environment before beginning
    the expensive 500k training run.
    """

    print()
    print(
        "=" * 70
    )

    print(
        "VALIDATING TRAFFIC MARL ENVIRONMENT"
    )

    print(
        "=" * 70
    )

    # -----------------------------------------------------
    # Observation
    # -----------------------------------------------------

    print(
        f"Observation space: "
        f"{env.observation_space}"
    )

    expected_observation = (
        120,
    )

    if (
        env.observation_space.shape
        != expected_observation
    ):

        raise ValueError(
            "\nInvalid observation space.\n"
            f"Expected: {expected_observation}\n"
            f"Received: "
            f"{env.observation_space.shape}"
        )

    print(
        "[PASS] Observation = 120-D"
    )

    # -----------------------------------------------------
    # Action
    # -----------------------------------------------------

    print(
        f"Action space: "
        f"{env.action_space}"
    )

    expected_actions = np.array(
        [
            2,
            2,
            2,
            2,
        ]
    )

    actual_actions = np.asarray(
        env.action_space.nvec
    )

    if not np.array_equal(
        actual_actions,
        expected_actions,
    ):

        raise ValueError(
            "\nInvalid action space.\n"
            f"Expected: {expected_actions}\n"
            f"Received: {actual_actions}"
        )

    print(
        "[PASS] Action = "
        "[2, 2, 2, 2]"
    )

    # -----------------------------------------------------
    # Scenario validation
    # -----------------------------------------------------

    available = (
        env.available_scenarios()
    )

    for scenario in SCENARIOS:

        if scenario not in available:

            raise ValueError(
                f"Required scenario is not "
                f"available: {scenario}"
            )

    print(
        f"[PASS] All {len(SCENARIOS)} "
        f"scenarios available"
    )

    # -----------------------------------------------------
    # Reset test
    # -----------------------------------------------------

    print()
    print(
        "Testing baseline reset..."
    )

    env.set_scenario(
        "baseline"
    )

    observation, info = (
        env.reset(
            seed=SEED
        )
    )

    observation = np.asarray(
        observation,
        dtype=np.float32,
    )

    if observation.shape != (
        120,
    ):

        raise ValueError(
            "Reset observation is not 120-D."
        )

    if not np.all(
        np.isfinite(
            observation
        )
    ):

        raise ValueError(
            "Reset observation contains "
            "NaN or infinite values."
        )

    print(
        "[PASS] Reset observation"
    )

    # -----------------------------------------------------
    # One action
    # -----------------------------------------------------

    action = np.array(
        [
            0,
            0,
            0,
            0,
        ],
        dtype=np.int32,
    )

    (
        next_observation,
        reward,
        terminated,
        truncated,
        step_info,
    ) = env.step(
        action
    )

    next_observation = np.asarray(
        next_observation,
        dtype=np.float32,
    )

    if next_observation.shape != (
        120,
    ):

        raise ValueError(
            "Step observation is not 120-D."
        )

    if not np.all(
        np.isfinite(
            next_observation
        )
    ):

        raise ValueError(
            "Step observation contains "
            "NaN or infinite values."
        )

    if not np.isfinite(
        float(reward)
    ):

        raise ValueError(
            "Reward is not finite."
        )

    print(
        "[PASS] Step observation"
    )

    print(
        f"[PASS] Reward = "
        f"{float(reward):.6f}"
    )

    print(
        f"[PASS] Scenario = "
        f"{step_info['scenario']}"
    )

    print()
    print(
        "=" * 70
    )

    print(
        "ENVIRONMENT VALIDATION PASSED"
    )

    print(
        "=" * 70
    )

    print()


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print(
        "=" * 80
    )

    print(
        "MULTI-AGENT REINFORCEMENT LEARNING"
    )

    print(
        "FOR NETWORK-WIDE ADAPTIVE TRAFFIC CONTROL"
    )

    print(
        "FINAL 500K MULTI-SCENARIO TRAINING"
    )

    print(
        "=" * 80
    )

    print()

    # -----------------------------------------------------
    # Seed
    # -----------------------------------------------------

    set_global_seed(
        SEED
    )


    EnvironmentConfig.SUMO_BINARY = "sumo"

    print(
        "SUMO mode        : HEADLESS"
    )

    print(
        "SUMO executable  : sumo"
    )

    print(
        "GUI              : False"
    )

    print()

    # -----------------------------------------------------
    # Configuration summary
    # -----------------------------------------------------

    print(
        f"Algorithm        : PPO-MARL"
    )

    print(
        f"Timesteps        : "
        f"{TOTAL_TIMESTEPS:,}"
    )

    print(
        f"Agents           : "
        f"{EnvironmentConfig.NUMBER_OF_AGENTS}"
    )

    print(
        f"Observation      : 120-D"
    )

    print(
        f"Actions           : "
        f"{EnvironmentConfig.NUMBER_OF_ACTIONS}"
    )

    print(
        f"Batch size       : "
        f"{BATCH_SIZE}"
    )

    print(
        f"N steps          : "
        f"{N_STEPS}"
    )

    print(
        f"Learning rate    : "
        f"{LEARNING_RATE}"
    )

    print()

    print(
        "Scenario training order:"
    )

    for index, scenario in enumerate(
        SCENARIOS,
        start=1,
    ):

        print(
            f"  {index:02d}. {scenario}"
        )

    print()

    # -----------------------------------------------------
    # Create base environment
    # -----------------------------------------------------

    base_env = None
    train_env = None
    model = None

    try:

        print(
            "Creating SUMO environment..."
        )

        base_env = TrafficEnvironment(
            scenario_name="baseline"
        )

        # -------------------------------------------------
        # Validate base environment
        # -------------------------------------------------

        validate_environment(
            base_env
        )

        # -------------------------------------------------
        # Re-create clean environment
        # -------------------------------------------------
        #
        # The validation above intentionally runs one SUMO
        # episode. Close it before training begins.
        # -------------------------------------------------

        base_env.close()

        base_env = None

        print(
            "Creating clean training environment..."
        )

        base_env = TrafficEnvironment(
            scenario_name="baseline"
        )

        # -------------------------------------------------
        # Scenario rotation wrapper
        # -------------------------------------------------

        scenario_env = (
            ScenarioRotationEnv(
                base_env,
                SCENARIOS,
            )
        )

        # -------------------------------------------------
        # Monitor
        # -------------------------------------------------

        train_env = Monitor(
            scenario_env,
            filename=str(
                LOG_DIR
                / "training_monitor.csv"
            ),
        )

        print(
            "[PASS] Scenario rotation wrapper created"
        )

        print(
            "[PASS] Monitor created"
        )

        # -------------------------------------------------
        # Initial reset
        # -------------------------------------------------

        train_env.reset(
            seed=SEED
        )

        print(
            "[PASS] Initial scenario = baseline"
        )

        # =================================================
        # PPO
        # =================================================

        print()
        print(
            "=" * 80
        )

        print(
            "CREATING PPO-MARL MODEL"
        )

        print(
            "=" * 80
        )

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

            tensorboard_log=str(
                LOG_DIR
            ),

            device="auto",
        )

        print()
        print(
            "[PASS] PPO-MARL model created"
        )

        print(
            f"Policy observation space:"
        )

        print(
            model.observation_space
        )

        print(
            f"Policy action space:"
        )

        print(
            model.action_space
        )

        # =================================================
        # CALLBACKS
        # =================================================

        checkpoint_callback = (
            CheckpointCallback(
                save_freq=25_000,

                save_path=str(
                    CHECKPOINT_DIR
                ),

                name_prefix=(
                    "ppo_marl_500k"
                ),

                save_replay_buffer=False,

                save_vecnormalize=False,
            )
        )

        progress_callback = (
            TrainingProgressCallback(
                report_frequency=10_000,
                verbose=1,
            )
        )

        callbacks = CallbackList(
            [
                checkpoint_callback,
                progress_callback,
            ]
        )

        # =================================================
        # TRAIN
        # =================================================

        print()
        print(
            "=" * 80
        )

        print(
            "STARTING 500,000 TIMESTEP TRAINING"
        )

        print(
            "=" * 80
        )

        print()

        print(
            "IMPORTANT:"
        )

        print(
            "SUMO is running HEADLESS."
        )

        print(
            "Scenario changes happen at episode reset."
        )

        print(
            "One SUMO environment is used to avoid"
        )

        print(
            "TraCI connection conflicts."
        )

        print()

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,

            callback=callbacks,

            progress_bar=True,

            reset_num_timesteps=True,

            tb_log_name=(
                "ppo_marl_500k"
            ),
        )

        # =================================================
        # FINAL SAVE
        # =================================================

        print()
        print(
            "=" * 80
        )

        print(
            "500K TRAINING COMPLETED"
        )

        print(
            "=" * 80
        )

        model.save(
            FINAL_MODEL_PATH
        )

        print()
        print(
            "[SAVED]"
        )

        print(
            f"Final model:"
        )

        print(
            f"{FINAL_MODEL_PATH}.zip"
        )

        print()
        print(
            f"Checkpoints:"
        )

        print(
            CHECKPOINT_DIR
        )

        print()
        print(
            f"TensorBoard logs:"
        )

        print(
            LOG_DIR
        )

        print()
        print(
            "=" * 80
        )

        print(
            "TRAINING SUCCESSFULLY FINISHED"
        )

        print(
            "=" * 80
        )

    except KeyboardInterrupt:

        print()
        print(
            "=" * 80
        )

        print(
            "TRAINING INTERRUPTED"
        )

        print(
            "=" * 80
        )

        if model is not None:

            interrupted_path = (
                TRAINING_ROOT
                / "ppo_marl_500k_interrupted"
            )

            model.save(
                interrupted_path
            )

            print()
            print(
                "Interrupted model saved:"
            )

            print(
                f"{interrupted_path}.zip"
            )

        raise

    except Exception as error:

        print()
        print(
            "=" * 80
        )

        print(
            "TRAINING ERROR"
        )

        print(
            "=" * 80
        )

        print(
            f"{type(error).__name__}:"
        )

        print(
            error
        )

        raise

    finally:

        # -------------------------------------------------
        # Close environment
        # -------------------------------------------------

        if train_env is not None:

            try:
                train_env.close()
            except Exception:
                pass

        elif base_env is not None:

            try:
                base_env.close()
            except Exception:
                pass

        print()
        print(
            "SUMO environment closed."
        )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()