"""
======================================================================
MAI-07 | DQN 500K BASELINE
Multi-Agent Reinforcement Learning for Network-Wide Adaptive Traffic
Control

PURPOSE
-------
Train a DQN baseline across ALL 10 traffic-demand scenarios.

This is an alternative RL baseline for comparison with:

    PPO-MARL 500K

IMPORTANT EXPERIMENTAL DESIGN
-----------------------------
DQN:
    - 500,000 timesteps
    - all 10 scenarios
    - 120-D environment observation
    - Discrete(16) joint-action adapter
    - same TrafficEnvironment
    - deterministic evaluation later
    - no hardcoded results

PPO-MARL:
    - 500,000 timesteps
    - all 10 scenarios
    - 120-D observation
    - MultiDiscrete([2,2,2,2])

The DQN action adapter converts:

    Discrete(16)

into:

    [A, B, C, D]

where every intersection has two possible actions:

    0 = North/South
    1 = East/West

OUTPUT
------
rl/checkpoints/dqn_500k/
    dqn_500k_final.zip
    checkpoints/

rl/logs/dqn_500k/

======================================================================
"""

import os
import sys
import time
from datetime import datetime, timezone

import gymnasium as gym
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor


# ======================================================================
# PROJECT ROOT
# ======================================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from rl.environments.traffic_env import TrafficEnvironment


# ======================================================================
# CONFIGURATION
# ======================================================================

TOTAL_TIMESTEPS = 500_000

BASE_SEED = 42

MAX_EPISODE_STEPS = 300

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

EXPECTED_OBSERVATION_DIM = 120

NUMBER_OF_AGENTS = 4

ACTIONS_PER_AGENT = 2

JOINT_ACTION_COUNT = (
    ACTIONS_PER_AGENT ** NUMBER_OF_AGENTS
)

assert JOINT_ACTION_COUNT == 16


# ======================================================================
# OUTPUT DIRECTORIES
# ======================================================================

CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT,
    "rl",
    "checkpoints",
    "dqn_500k",
)

CHECKPOINT_SUBDIR = os.path.join(
    CHECKPOINT_DIR,
    "checkpoints",
)

LOG_DIR = os.path.join(
    PROJECT_ROOT,                                              
    "rl",
    "logs",
    "dqn_500k",
)

FINAL_MODEL = os.path.join(
    CHECKPOINT_DIR,
    "dqn_500k_final.zip",
)                 


os.makedirs(
    CHECKPOINT_SUBDIR, 
    exist_ok=True,
)

os.makedirs(
    LOG_DIR,
    exist_ok=True,
)


# ======================================================================
# SUMO CONFIGURATION
# ======================================================================
#
# DQN is trained headlessly. This prevents SUMO-GUI windows from being
# opened 500,000 times worth of simulation.
#
# EnvironmentConfig is imported only for the training process.
# ======================================================================

try:
    from rl.environments.traffic_env import EnvironmentConfig

    EnvironmentConfig.SUMO_BINARY = "sumo"

except Exception:
    pass


# ======================================================================
# ACTION ENCODING
# ======================================================================

def encode_joint_action(actions):
    """
    Convert four binary intersection actions into one integer.

    Example:

        [0,0,0,0] -> 0
        [0,0,0,1] -> 1
        [0,0,1,0] -> 2
        [1,1,1,1] -> 15

    The mapping is deterministic and reversible.
    """

    actions = np.asarray(
        actions,
        dtype=np.int64,
    ).reshape(-1)

    if len(actions) != 4:
        raise ValueError(
            f"Expected 4 actions, got {len(actions)}"
        )

    action_id = 0

    for i, action in enumerate(actions):
        action = int(action)

        if action not in (0, 1):
            raise ValueError(
                f"Invalid binary action {action}"
            )

        action_id |= (
            action << i
        )

    return int(action_id)


def decode_joint_action(action_id):
    """
    Convert one DQN Discrete(16) action into four intersection actions.

    Example:

        0  -> [0,0,0,0]
        1  -> [1,0,0,0]
        5  -> [1,0,1,0]
        15 -> [1,1,1,1]

    The exact ordering is not important as long as it remains
    consistent during training and evaluation.
    """

    action_id = int(action_id)

    if action_id < 0 or action_id >= 16:
        raise ValueError(
            f"DQN action must be in [0,15], got {action_id}"
        )

    return np.asarray(
        [
            (action_id >> i) & 1
            for i in range(4)
        ],
        dtype=np.int64,
    )


# ======================================================================
# SAFETY / ACTION VALIDATION
# ======================================================================

def validate_joint_action(actions):
    """
    Final sanity check before passing the action to the traffic
    environment.

    Safety constraints such as minimum green time and yellow protection
    remain inside the existing TrafficController/SafetyValidator layer.
    This adapter does not bypass those controls.
    """

    actions = np.asarray(
        actions,
        dtype=np.int64,
    ).reshape(-1)

    if actions.shape != (4,):
        raise ValueError(
            f"Invalid joint action shape: {actions.shape}"
        )

    if not np.all(
        np.isin(actions, [0, 1])
    ):
        raise ValueError(
            f"Joint action contains invalid values: {actions}"
        )

    return actions


# ======================================================================
# DQN SCENARIO ROTATION ENVIRONMENT
# ======================================================================

class DQNScenarioRotationEnv(gym.Env):
    """
    One Gymnasium environment that rotates through all ten SUMO
    scenarios.

    IMPORTANT:
    We create a fresh TrafficEnvironment at each episode boundary.

    This avoids keeping multiple TraCI/SUMO environments alive at the
    same time and prevents global TraCI conflicts.

    Observation:
        Box-like 120-D vector returned by TrafficEnvironment.

    Action:
        Discrete(16)

    DQN action is decoded into:

        [action_A, action_B, action_C, action_D]
    """

    metadata = {
        "render_modes": []
    }

    def __init__(self):
        super().__init__()

        self.scenario_index = -1

        self.episode_index = 0

        self.current_scenario = None

        self.current_env = None

        # --------------------------------------------------------------
        # We need the real observation space from the actual environment.
        # --------------------------------------------------------------

        probe_env = TrafficEnvironment(
            scenario_name=SCENARIOS[0],
            render_mode=None,
        )

        try:
            actual_space = probe_env.observation_space

            actual_shape = tuple(
                actual_space.shape
            )

            if actual_shape != (
                EXPECTED_OBSERVATION_DIM,
            ):
                raise ValueError(
                    "\nDQN training stopped.\n"
                    "The current TrafficEnvironment does not "
                    "return the expected 120-D observation.\n"
                    f"Expected: ({EXPECTED_OBSERVATION_DIM},)\n"
                    f"Actual:   {actual_shape}\n"
                )

            self.observation_space = actual_space

        finally:
            try:
                probe_env.close()
            except Exception:
                pass

        # --------------------------------------------------------------
        # DQN requires Discrete action space.
        # --------------------------------------------------------------

        self.action_space = gym.spaces.Discrete(
            JOINT_ACTION_COUNT
        )

    # ------------------------------------------------------------------
    # CREATE NEXT SUMO ENVIRONMENT
    # ------------------------------------------------------------------

    def _create_environment(self):
        if self.current_env is not None:
            try:
                self.current_env.close()
            except Exception:
                pass

        self.current_scenario = SCENARIOS[
            self.scenario_index
        ]

        self.current_env = TrafficEnvironment(
            scenario_name=self.current_scenario,
            render_mode=None,
        )

        actual_shape = tuple(
            self.current_env.observation_space.shape
        )

        if actual_shape != (
            EXPECTED_OBSERVATION_DIM,
        ):
            raise RuntimeError(
                f"Scenario '{self.current_scenario}' "
                f"returned observation shape "
                f"{actual_shape}; expected "
                f"({EXPECTED_OBSERVATION_DIM},)"
            )

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        super().reset(
            seed=seed
        )

        # --------------------------------------------------------------
        # Rotate:
        #
        # baseline
        # low_demand
        # balanced
        # ...
        # stress_test
        # baseline
        # ...
        # --------------------------------------------------------------

        self.scenario_index = (
            self.scenario_index + 1
        ) % len(SCENARIOS)

        self.episode_index += 1

        self._create_environment()

        # --------------------------------------------------------------
        # Deterministic reproducible seed.
        # --------------------------------------------------------------

        if seed is None:
            seed = (
                BASE_SEED
                + self.episode_index
            )

        observation, info = (
            self.current_env.reset(
                seed=seed
            )
        )

        observation = np.asarray(
            observation,
            dtype=np.float32,
        )

        if observation.shape != (
            EXPECTED_OBSERVATION_DIM,
        ):
            raise RuntimeError(
                f"Environment returned "
                f"{observation.shape}; expected "
                f"(120,)"
            )

        info = dict(
            info or {}
        )

        info.update(
            {
                "scenario": self.current_scenario,
                "episode_index": self.episode_index,
                "seed": seed,
                "dqn_action_space": "Discrete(16)",
                "underlying_action_space": (
                    "MultiDiscrete([2,2,2,2])"
                ),
            }
        )

        return observation, info

    # ------------------------------------------------------------------
    # STEP
    # ------------------------------------------------------------------

    def step(self, action):
        if self.current_env is None:
            raise RuntimeError(
                "step() called before reset()."
            )

        # --------------------------------------------------------------
        # DQN action
        # --------------------------------------------------------------

        action_id = int(action)

        # --------------------------------------------------------------
        # Convert:
        #
        # Discrete(16)
        #       ↓
        # [A,B,C,D]
        # --------------------------------------------------------------

        joint_action = decode_joint_action(
            action_id
        )

        joint_action = validate_joint_action(
            joint_action
        )

        # --------------------------------------------------------------
        # Send to existing traffic environment.
        #
        # Existing safety/controller logic remains active.
        # --------------------------------------------------------------

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = self.current_env.step(
            joint_action
        )

        observation = np.asarray(
            observation,
            dtype=np.float32,
        )

        if observation.shape != (
            EXPECTED_OBSERVATION_DIM,
        ):
            raise RuntimeError(
                f"Environment returned "
                f"{observation.shape}; expected "
                f"(120,)"
            )

        info = dict(
            info or {}
        )

        info.update(
            {
                "scenario": self.current_scenario,
                "dqn_action_id": action_id,
                "joint_action": joint_action.tolist(),
            }
        )

        return (
            observation,
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    # ------------------------------------------------------------------
    # CLOSE
    # ------------------------------------------------------------------

    def close(self):
        if self.current_env is not None:
            try:
                self.current_env.close()
            except Exception:
                pass

            self.current_env = None


# ======================================================================
# VALIDATION BEFORE TRAINING
# ======================================================================

def validate_environment():
    print()
    print("=" * 78)
    print("DQN ENVIRONMENT VALIDATION")
    print("=" * 78)

    env = DQNScenarioRotationEnv()

    try:
        print()
        print(
            "Observation space:",
            env.observation_space,
        )

        print(
            "Observation shape:",
            env.observation_space.shape,
        )

        print(
            "DQN action space:",
            env.action_space,
        )

        print(
            "Number of DQN actions:",
            env.action_space.n,
        )

        print()
        print(
            "Scenario rotation:"
        )

        for index, scenario in enumerate(
            SCENARIOS,
            start=1,
        ):
            print(
                f"  {index:02d}. {scenario}"
            )

        # --------------------------------------------------------------
        # Test every possible joint action.
        # --------------------------------------------------------------

        print()
        print(
            "Testing all 16 DQN action mappings:"
        )

        for action_id in range(16):
            decoded = decode_joint_action(
                action_id
            )

            encoded = encode_joint_action(
                decoded
            )

            if encoded != action_id:
                raise AssertionError(
                    f"Action mapping failed: "
                    f"{action_id} -> "
                    f"{decoded} -> "
                    f"{encoded}"
                )

            print(
                f"  {action_id:02d} -> "
                f"{decoded.tolist()}"
            )

        print()
        print(
            "[PASS] All 16 action mappings are reversible."
        )

        # --------------------------------------------------------------
        # Test each scenario once.
        # --------------------------------------------------------------

        print()
        print(
            "Testing all 10 scenario resets:"
        )

        for expected_scenario in SCENARIOS:
            observation, info = env.reset(
                seed=BASE_SEED
            )

            actual_scenario = info.get(
                "scenario"
            )

            if actual_scenario != expected_scenario:
                raise AssertionError(
                    f"Scenario rotation failed. "
                    f"Expected {expected_scenario}, "
                    f"got {actual_scenario}"
                )

            if observation.shape != (
                EXPECTED_OBSERVATION_DIM,
            ):
                raise AssertionError(
                    f"{expected_scenario}: "
                    f"wrong observation shape "
                    f"{observation.shape}"
                )

            print(
                f"  [PASS] {expected_scenario}: "
                f"{observation.shape}"
            )

        print()
        print(
            "[PASS] DQN environment validation complete."
        )

    finally:
        env.close()


# ======================================================================
# TRAINING
# ======================================================================

def train():
    print()
    print("=" * 78)
    print("MAI-07 | DQN 500K BASELINE TRAINING")
    print("=" * 78)

    print()
    print("Training configuration:")
    print(
        f"  Timesteps       : {TOTAL_TIMESTEPS:,}"
    )
    print(
        f"  Observation     : {EXPECTED_OBSERVATION_DIM}-D"
    )
    print(
        f"  DQN actions     : Discrete({JOINT_ACTION_COUNT})"
    )
    print(
        "  Underlying      : MultiDiscrete([2,2,2,2])"
    )
    print(
        f"  Scenarios       : {len(SCENARIOS)}"
    )
    print(
        "  Episode steps   : "
        f"{MAX_EPISODE_STEPS}"
    )
    print(
        f"  Seed            : {BASE_SEED}"
    )

    print()
    print("Scenarios:")
    for scenario in SCENARIOS:
        print(
            f"  - {scenario}"
        )

    # --------------------------------------------------------------
    # Environment
    # --------------------------------------------------------------

    env = DQNScenarioRotationEnv()

    env = Monitor(
        env,
        filename=os.path.join(
            LOG_DIR,
            "monitor.csv",
        ),
    )

    # --------------------------------------------------------------
    # Check spaces one more time before model creation.
    # --------------------------------------------------------------

    if tuple(
        env.observation_space.shape
    ) != (
        EXPECTED_OBSERVATION_DIM,
    ):
        raise RuntimeError(
            "DQN observation space validation failed."
        )

    if env.action_space.n != 16:
        raise RuntimeError(
            "DQN action space must be Discrete(16)."
        )

    # --------------------------------------------------------------
    # Checkpoint callback
    #
    # Save every 25,000 steps.
    # --------------------------------------------------------------

    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path=CHECKPOINT_SUBDIR,
        name_prefix="dqn_500k",
    )

    # --------------------------------------------------------------
    # DQN
    # --------------------------------------------------------------

    model = DQN(
        policy="MlpPolicy",
        env=env,

        # ----------------------------------------------------------
        # Core training
        # ----------------------------------------------------------

        learning_rate=1e-4,

        buffer_size=50_000,

        learning_starts=5_000,

        batch_size=64,

        gamma=0.99,

        train_freq=4,

        gradient_steps=1,

        target_update_interval=2_000,

        # ----------------------------------------------------------
        # Exploration
        # ----------------------------------------------------------

        exploration_initial_eps=1.0,

        exploration_final_eps=0.05,

        exploration_fraction=0.20,

        # ----------------------------------------------------------
        # Neural network
        # ----------------------------------------------------------

        policy_kwargs=dict(
            net_arch=[256, 256],
        ),

        # ----------------------------------------------------------
        # Reproducibility
        # ----------------------------------------------------------

        seed=BASE_SEED,

        device="auto",

        verbose=1,

        tensorboard_log=LOG_DIR,
    )

    print()
    print(
        "[PASS] DQN model created."
    )

    print(
        "Observation space:",
        model.observation_space,
    )

    print(
        "Action space:",
        model.action_space,
    )

    if tuple(
        model.observation_space.shape
    ) != (
        EXPECTED_OBSERVATION_DIM,
    ):
        raise RuntimeError(
            "DQN model was not created with 120-D observation."
        )

    if model.action_space.n != 16:
        raise RuntimeError(
            "DQN model was not created with Discrete(16)."
        )

    # --------------------------------------------------------------
    # Train
    # --------------------------------------------------------------

    start_time = time.time()

    print()
    print("=" * 78)
    print("STARTING DQN 500K TRAINING")
    print("=" * 78)
    print()

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        progress_bar=True,
        reset_num_timesteps=False,
    )

    elapsed = time.time() - start_time

    # --------------------------------------------------------------
    # Save final model
    # --------------------------------------------------------------

    model.save(
        FINAL_MODEL
    )

    print()
    print("=" * 78)
    print("DQN 500K TRAINING COMPLETE")
    print("=" * 78)

    print()
    print(
        f"Timesteps requested : "
        f"{TOTAL_TIMESTEPS:,}"
    )

    print(
        f"Training time       : "
        f"{elapsed / 3600:.2f} hours"
    )

    print(
        "Final model:"
    )

    print(
        FINAL_MODEL
    )

    print()
    print(
        "Checkpoint directory:"
    )

    print(
        CHECKPOINT_SUBDIR
    )

    print()
    print(
        "TensorBoard/log directory:"
    )

    print(
        LOG_DIR
    )

    print()
    print("=" * 78)
    print("DQN BASELINE READY FOR EVALUATION")
    print("=" * 78)

    env.close()


# ======================================================================
# MAIN
# ======================================================================

if __name__ == "__main__":

    validate_environment()

    print()
    print(
        "Environment validation passed."
    )

    print()
    response = input(
        "Start the 500K DQN training now? "
        "Type YES to continue: "
    ).strip()

    if response.upper() != "YES":
        print()
        print(
            "Training cancelled."
        )
        print(
            "No model or experiment was modified."
        )
        sys.exit(0)

    train()