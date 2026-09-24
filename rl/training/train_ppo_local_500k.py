"""
==============================================================================
MAI-07 | PPO-LOCAL 500K BASELINE TRAINING
==============================================================================

Multi-Agent Reinforcement Learning for Network-Wide Adaptive Traffic Control

PURPOSE
-------
Train a PPO-Local baseline across ALL 10 traffic-demand scenarios.

This model is the LOCAL-INFORMATION RL baseline for comparison with:

    1. DQN 500K
    2. PPO-MARL 500K (proposed/core model)

EXPERIMENTAL DESIGN
-------------------
PPO-Local:
    - 500,000 timesteps
    - all 10 scenarios
    - 60-D observation
    - local traffic information only
    - MultiDiscrete([2,2,2,2]) actions
    - same TrafficEnvironment
    - same episode length
    - same seed family
    - deterministic evaluation later
    - no hardcoded results

CURRENT ENVIRONMENT
-------------------
TrafficEnvironment returns:

    120-D global observation

Structure:

    Agent A: 30 = 15 local + 15 neighbour
    Agent B: 30 = 15 local + 15 neighbour
    Agent C: 30 = 15 local + 15 neighbour
    Agent D: 30 = 15 local + 15 neighbour

PPO-Local receives:

    A local 15
    B local 15
    C local 15
    D local 15

Total:

    60-D

The neighbour information is intentionally removed.

ACTION SPACE
------------
Underlying environment:

    MultiDiscrete([2,2,2,2])

Actions:

    0 = North/South
    1 = East/West

SAFETY
------
The wrapper does NOT bypass the existing traffic controller.

Minimum-green-time and yellow-transition protection remain inside
the existing TrafficController / SafetyValidator layer.

SCENARIOS
---------
    baseline
    low_demand
    balanced
    morning_rush
    evening_rush
    high_demand
    north_south_heavy
    east_west_heavy
    random
    stress_test

OUTPUT
------
rl/checkpoints/ppo_local_500k/
    ppo_local_500k_final.zip
    checkpoints/

rl/logs/ppo_local_500k/

==============================================================================
"""

import os
import sys
import time
import json

import gymnasium as gym
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor


# ============================================================================
# PROJECT ROOT
# ============================================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================================
# PROJECT IMPORT
# ============================================================================

from rl.environments.traffic_env import TrafficEnvironment


# ============================================================================
# CONFIGURATION
# ============================================================================

TOTAL_TIMESTEPS = 500_000

BASE_SEED = 42

MAX_EPISODE_STEPS = 300

EXPECTED_GLOBAL_DIM = 120

EXPECTED_LOCAL_DIM = 60

NUMBER_OF_AGENTS = 4

LOCAL_FEATURES_PER_AGENT = 15

GRAPH_FEATURES_PER_AGENT = 30

NEIGHBOUR_FEATURES_PER_AGENT = 15


# ============================================================================
# ALL 10 SCENARIOS
# ============================================================================

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


# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================

CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT,
    "rl",
    "checkpoints",
    "ppo_local_500k",
)

CHECKPOINT_SUBDIR = os.path.join(
    CHECKPOINT_DIR,
    "checkpoints",
)

LOG_DIR = os.path.join(
    PROJECT_ROOT,
    "rl",
    "logs",
    "ppo_local_500k",
)

FINAL_MODEL = os.path.join(
    CHECKPOINT_DIR,
    "ppo_local_500k_final.zip",
)

# Resume from the verified 150K checkpoint.
RESUME_CHECKPOINT = os.path.join(
    CHECKPOINT_SUBDIR,
    "ppo_local_500k_150000_steps.zip",
)

RESUME_FROM_CHECKPOINT = True

# Require NVIDIA CUDA for this resumed PPO-Local run.
# SUMO itself remains CPU-bound; only the PPO neural-network computation
# is moved to the NVIDIA GPU.
REQUIRE_CUDA = True


TRAINING_INFO_FILE = os.path.join(
    CHECKPOINT_DIR,
    "training_info.json",
)


os.makedirs(
    CHECKPOINT_SUBDIR,
    exist_ok=True,
)

os.makedirs(
    LOG_DIR,
    exist_ok=True,
)


# ============================================================================
# SUMO HEADLESS CONFIGURATION
# ============================================================================

try:
    from rl.environments.traffic_env import EnvironmentConfig

    EnvironmentConfig.SUMO_BINARY = "sumo"

except Exception:
    pass


# ============================================================================
# OBSERVATION ADAPTER
# ============================================================================

def convert_120_to_60(observation):
    """
    Convert the current 120-D graph-aware observation into
    the 60-D local-only observation.

    Current structure:

        Agent A = 30
            local    = 15
            neighbour = 15

        Agent B = 30
            local    = 15
            neighbour = 15

        Agent C = 30
            local    = 15
            neighbour = 15

        Agent D = 30
            local    = 15
            neighbour = 15

    PPO-Local receives only:

        A local 15
        B local 15
        C local 15
        D local 15

    Therefore:

        4 x 15 = 60
    """

    observation = np.asarray(
        observation,
        dtype=np.float32,
    ).reshape(-1)

    # ------------------------------------------------------------------
    # Validate input
    # ------------------------------------------------------------------

    if observation.shape != (EXPECTED_GLOBAL_DIM,):
        raise ValueError(
            "\nPPO-Local observation adapter error.\n"
            f"Expected global observation: "
            f"({EXPECTED_GLOBAL_DIM},)\n"
            f"Received: {observation.shape}\n"
        )

    local_parts = []

    # ------------------------------------------------------------------
    # Extract first 15 values from each 30-value agent block.
    # ------------------------------------------------------------------

    for agent_index in range(NUMBER_OF_AGENTS):

        block_start = (
            agent_index
            * GRAPH_FEATURES_PER_AGENT
        )

        block_end = (
            block_start
            + GRAPH_FEATURES_PER_AGENT
        )

        agent_block = observation[
            block_start:block_end
        ]

        if agent_block.shape != (
            GRAPH_FEATURES_PER_AGENT,
        ):
            raise ValueError(
                "Invalid graph block for "
                f"agent {agent_index}: "
                f"{agent_block.shape}"
            )

        local_part = agent_block[
            :LOCAL_FEATURES_PER_AGENT
        ]

        if local_part.shape != (
            LOCAL_FEATURES_PER_AGENT,
        ):
            raise ValueError(
                "Invalid local block for "
                f"agent {agent_index}: "
                f"{local_part.shape}"
            )

        local_parts.append(local_part)

    # ------------------------------------------------------------------
    # Combine all local features
    # ------------------------------------------------------------------

    local_observation = np.concatenate(
        local_parts
    ).astype(
        np.float32
    )

    # ------------------------------------------------------------------
    # Final validation
    # ------------------------------------------------------------------

    if local_observation.shape != (
        EXPECTED_LOCAL_DIM,
    ):
        raise ValueError(
            "\nPPO-Local observation dimension error.\n"
            f"Expected: ({EXPECTED_LOCAL_DIM},)\n"
            f"Received: {local_observation.shape}\n"
        )

    return local_observation


# ============================================================================
# PPO-LOCAL SCENARIO ROTATION ENVIRONMENT
# ============================================================================

class PPOLocalScenarioRotationEnv(gym.Env):
    """
    Gymnasium wrapper for PPO-Local.

    The underlying TrafficEnvironment remains unchanged.

    The wrapper:

        120-D TrafficEnvironment observation
                    |
                    v
             local extraction
                    |
                    v
                 60-D
                    |
                    v
                  PPO

    The environment rotates through all 10 scenarios
    episode by episode.

    Only one SUMO environment is kept alive at a time.
    This avoids global TraCI conflicts.
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
        # Probe actual environment
        # --------------------------------------------------------------

        probe_env = TrafficEnvironment(
            scenario_name=SCENARIOS[0],
            render_mode=None,
        )

        try:

            actual_space = (
                probe_env.observation_space
            )

            actual_shape = tuple(
                actual_space.shape
            )

            if actual_shape != (
                EXPECTED_GLOBAL_DIM,
            ):
                raise ValueError(
                    "\nPPO-Local training stopped.\n"
                    "Current TrafficEnvironment must "
                    "return 120-D observations.\n"
                    f"Expected: "
                    f"({EXPECTED_GLOBAL_DIM},)\n"
                    f"Actual:   {actual_shape}\n"
                )

            # ----------------------------------------------------------
            # PPO sees a 60-D observation.
            # ----------------------------------------------------------

            low = np.full(
                (
                    EXPECTED_LOCAL_DIM,
                ),
                -100000.0,
                dtype=np.float32,
            )

            high = np.full(
                (
                    EXPECTED_LOCAL_DIM,
                ),
                100000.0,
                dtype=np.float32,
            )

            self.observation_space = (
                gym.spaces.Box(
                    low=low,
                    high=high,
                    dtype=np.float32,
                )
            )

            # ----------------------------------------------------------
            # Action space is unchanged.
            # ----------------------------------------------------------

            actual_action_space = (
                probe_env.action_space
            )

            self.action_space = actual_action_space

            # ----------------------------------------------------------
            # Validate expected action space.
            # ----------------------------------------------------------

            if not isinstance(
                self.action_space,
                gym.spaces.MultiDiscrete,
            ):
                raise ValueError(
                    "Expected MultiDiscrete action space, "
                    f"received {self.action_space}"
                )

            expected_actions = np.array(
                [2, 2, 2, 2]
            )

            if not np.array_equal(
                self.action_space.nvec,
                expected_actions,
            ):
                raise ValueError(
                    "Unexpected action space.\n"
                    f"Expected: MultiDiscrete([2,2,2,2])\n"
                    f"Actual:   {self.action_space}"
                )

        finally:

            try:
                probe_env.close()
            except Exception:
                pass

    # ==================================================================
    # CREATE ENVIRONMENT
    # ==================================================================

    def _create_environment(self):

        # --------------------------------------------------------------
        # Close previous environment.
        # --------------------------------------------------------------

        if self.current_env is not None:

            try:
                self.current_env.close()
            except Exception:
                pass

            self.current_env = None

        # --------------------------------------------------------------
        # Select scenario.
        # --------------------------------------------------------------

        self.current_scenario = (
            SCENARIOS[
                self.scenario_index
            ]
        )

        # --------------------------------------------------------------
        # Create fresh SUMO environment.
        # --------------------------------------------------------------

        self.current_env = TrafficEnvironment(
            scenario_name=self.current_scenario,
            render_mode=None,
        )

        # --------------------------------------------------------------
        # Validate actual environment.
        # --------------------------------------------------------------

        actual_shape = tuple(
            self.current_env
            .observation_space
            .shape
        )

        if actual_shape != (
            EXPECTED_GLOBAL_DIM,
        ):
            raise RuntimeError(
                f"Scenario '{self.current_scenario}' "
                "returned an unexpected observation shape.\n"
                f"Expected: "
                f"({EXPECTED_GLOBAL_DIM},)\n"
                f"Actual: {actual_shape}"
            )

        if not isinstance(
            self.current_env.action_space,
            gym.spaces.MultiDiscrete,
        ):
            raise RuntimeError(
                f"Scenario '{self.current_scenario}' "
                "does not use MultiDiscrete actions."
            )

    # ==================================================================
    # RESET
    # ==================================================================

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
        # Rotate through all 10 scenarios.
        # --------------------------------------------------------------

        self.scenario_index = (
            self.scenario_index + 1
        ) % len(SCENARIOS)

        self.episode_index += 1

        self._create_environment()

        # --------------------------------------------------------------
        # Reproducible seed.
        # --------------------------------------------------------------

        if seed is None:

            seed = (
                BASE_SEED
                + self.episode_index
                - 1
            )

        # --------------------------------------------------------------
        # Reset underlying SUMO environment.
        # --------------------------------------------------------------

        observation, info = (
            self.current_env.reset(
                seed=seed
            )
        )

        observation = np.asarray(
            observation,
            dtype=np.float32,
        ).reshape(-1)

        # --------------------------------------------------------------
        # Validate global observation.
        # --------------------------------------------------------------

        if observation.shape != (
            EXPECTED_GLOBAL_DIM,
        ):
            raise RuntimeError(
                "Underlying environment returned "
                f"{observation.shape}; "
                f"expected "
                f"({EXPECTED_GLOBAL_DIM},)"
            )

        # --------------------------------------------------------------
        # Convert 120 -> 60.
        # --------------------------------------------------------------

        local_observation = (
            convert_120_to_60(
                observation
            )
        )

        # --------------------------------------------------------------
        # Add useful metadata.
        # --------------------------------------------------------------

        info = dict(
            info or {}
        )

        info.update(
            {
                "scenario": (
                    self.current_scenario
                ),
                "episode_index": (
                    self.episode_index
                ),
                "seed": seed,
                "observation_type": (
                    "local_only"
                ),
                "global_observation_dim": (
                    EXPECTED_GLOBAL_DIM
                ),
                "local_observation_dim": (
                    EXPECTED_LOCAL_DIM
                ),
                "underlying_action_space": (
                    "MultiDiscrete([2,2,2,2])"
                ),
            }
        )

        return (
            local_observation,
            info,
        )

    # ==================================================================
    # STEP
    # ==================================================================

    def step(self, action):

        if self.current_env is None:
            raise RuntimeError(
                "step() called before reset()."
            )

        # --------------------------------------------------------------
        # Validate PPO action.
        # --------------------------------------------------------------

        action = np.asarray(
            action,
            dtype=np.int64,
        ).reshape(-1)

        if action.shape != (
            NUMBER_OF_AGENTS,
        ):
            raise ValueError(
                "Invalid PPO-Local action shape.\n"
                f"Expected: ({NUMBER_OF_AGENTS},)\n"
                f"Received: {action.shape}"
            )

        if not np.all(
            np.isin(
                action,
                [0, 1],
            )
        ):
            raise ValueError(
                "PPO-Local produced an invalid action:\n"
                f"{action}"
            )

        # --------------------------------------------------------------
        # Send action to underlying environment.
        #
        # Existing TrafficController / SafetyValidator remains active.
        # --------------------------------------------------------------

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = self.current_env.step(
            action
        )

        observation = np.asarray(
            observation,
            dtype=np.float32,
        ).reshape(-1)

        # --------------------------------------------------------------
        # Validate global observation.
        # --------------------------------------------------------------

        if observation.shape != (
            EXPECTED_GLOBAL_DIM,
        ):
            raise RuntimeError(
                "Underlying environment returned "
                f"{observation.shape}; "
                f"expected "
                f"({EXPECTED_GLOBAL_DIM},)"
            )

        # --------------------------------------------------------------
        # Convert 120 -> 60.
        # --------------------------------------------------------------

        local_observation = (
            convert_120_to_60(
                observation
            )
        )

        # --------------------------------------------------------------
        # Add metadata.
        # --------------------------------------------------------------

        info = dict(
            info or {}
        )

        info.update(
            {
                "scenario": (
                    self.current_scenario
                ),
                "ppo_local_action": (
                    action.tolist()
                ),
                "observation_type": (
                    "local_only"
                ),
            }
        )

        return (
            local_observation,
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    # ==================================================================
    # CLOSE
    # ==================================================================

    def close(self):

        if self.current_env is not None:

            try:
                self.current_env.close()
            except Exception:
                pass

            self.current_env = None


# ============================================================================
# VALIDATION
# ============================================================================

def validate_environment():

    print()
    print("=" * 78)
    print("PPO-LOCAL ENVIRONMENT VALIDATION")
    print("=" * 78)

    env = PPOLocalScenarioRotationEnv()

    try:

        print()
        print(
            "PPO observation space:",
            env.observation_space,
        )

        print(
            "PPO observation shape:",
            env.observation_space.shape,
        )

        print(
            "Action space:",
            env.action_space,
        )

        print()
        print(
            "Expected observation transformation:"
        )

        print(
            "  TrafficEnvironment : 120-D"
        )

        print(
            "  PPO-Local          : 60-D"
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
        # Test direct conversion.
        # --------------------------------------------------------------

        print()
        print(
            "Testing 120-D -> 60-D observation adapter..."
        )

        test_observation = np.arange(
            EXPECTED_GLOBAL_DIM,
            dtype=np.float32,
        )

        converted = convert_120_to_60(
            test_observation
        )

        if converted.shape != (
            EXPECTED_LOCAL_DIM,
        ):
            raise AssertionError(
                "Observation conversion failed."
            )

        # Expected:
        #
        # Agent A local = 0:15
        # Agent B local = 30:45
        # Agent C local = 60:75
        # Agent D local = 90:105

        expected = np.concatenate(
            [
                test_observation[
                    0:15
                ],
                test_observation[
                    30:45
                ],
                test_observation[
                    60:75
                ],
                test_observation[
                    90:105
                ],
            ]
        )

        if not np.array_equal(
            converted,
            expected,
        ):
            raise AssertionError(
                "120-D -> 60-D local extraction "
                "mapping failed."
            )

        print(
            "[PASS] 120-D -> 60-D adapter."
        )

        # --------------------------------------------------------------
        # Test every scenario.
        # --------------------------------------------------------------

        print()
        print(
            "Testing all 10 scenario resets..."
        )

        for expected_scenario in SCENARIOS:

            observation, info = env.reset(
                seed=BASE_SEED
            )

            actual_scenario = (
                info.get("scenario")
            )

            if actual_scenario != (
                expected_scenario
            ):
                raise AssertionError(
                    "Scenario rotation failed.\n"
                    f"Expected: {expected_scenario}\n"
                    f"Actual:   {actual_scenario}"
                )

            if observation.shape != (
                EXPECTED_LOCAL_DIM,
            ):
                raise AssertionError(
                    f"{expected_scenario}: "
                    "wrong PPO-Local observation shape.\n"
                    f"Expected: "
                    f"({EXPECTED_LOCAL_DIM},)\n"
                    f"Actual:   {observation.shape}"
                )

            print(
                f"  [PASS] {expected_scenario}: "
                f"{observation.shape}"
            )

        # --------------------------------------------------------------
        # Test action space.
        # --------------------------------------------------------------

        if not isinstance(
            env.action_space,
            gym.spaces.MultiDiscrete,
        ):
            raise AssertionError(
                "Action space is not MultiDiscrete."
            )

        if not np.array_equal(
            env.action_space.nvec,
            np.array([2, 2, 2, 2]),
        ):
            raise AssertionError(
                "Action space is not "
                "MultiDiscrete([2,2,2,2])."
            )

        print()
        print(
            "[PASS] Action space: "
            "MultiDiscrete([2,2,2,2])"
        )

        print()
        print(
            "[PASS] PPO-Local environment "
            "validation complete."
        )

    finally:

        env.close()


# ============================================================================
# SAVE TRAINING INFORMATION
# ============================================================================

def save_training_information():

    information = {
        "project": "MAI-07",
        "algorithm": "PPO",
        "baseline": "PPO-Local",
        "total_timesteps": TOTAL_TIMESTEPS,
        "seed": BASE_SEED,
        "episode_steps": MAX_EPISODE_STEPS,

        "scenarios": SCENARIOS,

        "network": {
            "intersections": 4,
            "agents": 4,
            "topology": "2x2 grid",
        },

        "observation": {
            "underlying_dimension": 120,
            "local_dimension": 60,
            "local_features_per_agent": 15,
            "agents": 4,
            "neighbour_information_used": False,
        },

        "action_space": {
            "type": "MultiDiscrete",
            "agents": 4,
            "actions_per_agent": 2,
            "values": [0, 1],
        },

        "hyperparameters": {
            "learning_rate": 0.0003,
            "n_steps": 256,
            "batch_size": 64,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        },

        "model_path": FINAL_MODEL,

        "description": (
            "PPO-Local baseline trained for "
            "500,000 timesteps across all ten "
            "traffic-demand scenarios. The model "
            "receives only local traffic features "
            "from the four intersections and does "
            "not receive neighbour features."
        ),
    }

    with open(
        TRAINING_INFO_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            information,
            file,
            indent=4,
        )


# ============================================================================
# GPU / CUDA VALIDATION
# ============================================================================

def get_training_device():
    """
    Select and validate the NVIDIA CUDA device.

    This does not move SUMO to the GPU. SUMO/TraCI remains CPU-based.
    The PPO policy and optimizer are placed on CUDA.
    """
    print()
    print("=" * 78)
    print("NVIDIA CUDA VALIDATION")
    print("=" * 78)

    print("PyTorch version:", torch.__version__)
    print("CUDA build:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        message = (
            "\nNVIDIA CUDA is NOT available to PyTorch.\n"
            "The script is configured to require CUDA and will not silently "
            "fall back to CPU.\n"
            "Install a CUDA-enabled PyTorch build and verify the NVIDIA "
            "driver before starting PPO-Local."
        )
        raise RuntimeError(message)

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory_gb = (
        torch.cuda.get_device_properties(0).total_memory
        / (1024 ** 3)
    )

    print("Selected device:", device)
    print("GPU:", gpu_name)
    print(f"GPU memory: {gpu_memory_gb:.2f} GB")

    if REQUIRE_CUDA and device.type != "cuda":
        raise RuntimeError("CUDA is required but a CUDA device was not selected.")

    # Small CUDA operation verifies actual execution, not just detection.
    test_tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
    test_result = (test_tensor * 2.0).sum().item()
    torch.cuda.synchronize()

    if test_result != 12.0:
        raise RuntimeError("CUDA execution validation failed.")

    print("[PASS] CUDA device is available and executing PyTorch operations.")
    return device


# ============================================================================
# CREATE / RESUME MODEL
# ============================================================================

def create_model(env, device):

    print()
    print("=" * 78)
    print("CREATING PPO-LOCAL 500K MODEL")
    print("=" * 78)

    model = PPO(
        policy="MlpPolicy",
        env=env,

        # --------------------------------------------------------------
        # Same PPO family / main hyperparameters as proposed model
        # --------------------------------------------------------------

        learning_rate=3e-4,

        n_steps=256,

        batch_size=64,

        gamma=0.99,

        gae_lambda=0.95,

        clip_range=0.2,

        ent_coef=0.01,

        vf_coef=0.5,

        max_grad_norm=0.5,

        # --------------------------------------------------------------
        # Reproducibility
        # --------------------------------------------------------------

        seed=BASE_SEED,

        # Explicitly use NVIDIA CUDA.
        device=device,

        verbose=1,

        tensorboard_log=LOG_DIR,
    )

    print()
    print(
        "[PASS] PPO-Local model created."
    )

    print(
        "Observation space:",
        model.observation_space,
    )

    print(
        "Action space:",
        model.action_space,
    )

    # --------------------------------------------------------------
    # Final model validation
    # --------------------------------------------------------------

    if tuple(
        model.observation_space.shape
    ) != (
        EXPECTED_LOCAL_DIM,
    ):
        raise RuntimeError(
            "PPO-Local model was not created "
            "with 60-D observation."
        )

    if not isinstance(
        model.action_space,
        gym.spaces.MultiDiscrete,
    ):
        raise RuntimeError(
            "PPO-Local model does not use "
            "MultiDiscrete action space."
        )

    if not np.array_equal(
        model.action_space.nvec,
        np.array([2, 2, 2, 2]),
    ):
        raise RuntimeError(
            "PPO-Local model does not use "
            "MultiDiscrete([2,2,2,2])."
        )

    return model


# ============================================================================
# RESUME FROM VERIFIED 150K CHECKPOINT
# ============================================================================

def load_resume_model(env, device):
    """
    Load the verified 150K PPO-Local checkpoint.

    The checkpoint is expected to contain PPO's policy, optimizer state,
    running timestep counter, and training state. We then continue with
    reset_num_timesteps=False so the global timestep count continues toward
    500,000 rather than restarting at zero.
    """
    if not os.path.isfile(RESUME_CHECKPOINT):
        raise FileNotFoundError(
            "\nRequired PPO-Local resume checkpoint was not found:\n"
            f"{RESUME_CHECKPOINT}\n"
            "Refusing to start from zero."
        )

    print()
    print("=" * 78)
    print("LOADING PPO-LOCAL 150K CHECKPOINT")
    print("=" * 78)
    print("Checkpoint:")
    print(RESUME_CHECKPOINT)

    model = PPO.load(
        RESUME_CHECKPOINT,
        env=env,
        device=device,
        print_system_info=True,
    )

    if model.env is None:
        raise RuntimeError("Loaded PPO-Local checkpoint has no training environment.")

    if tuple(model.observation_space.shape) != (EXPECTED_LOCAL_DIM,):
        raise RuntimeError(
            "Loaded checkpoint observation dimension is incorrect: "
            f"{model.observation_space.shape}"
        )

    if not isinstance(model.action_space, gym.spaces.MultiDiscrete):
        raise RuntimeError(
            "Loaded checkpoint action space is not MultiDiscrete."
        )

    if not np.array_equal(
        model.action_space.nvec,
        np.array([2, 2, 2, 2]),
    ):
        raise RuntimeError(
            "Loaded checkpoint action space is not "
            "MultiDiscrete([2,2,2,2])."
        )

    loaded_steps = int(model.num_timesteps)

    print("Loaded model device:", model.device)
    print("Loaded model timesteps:", f"{loaded_steps:,}")

    if loaded_steps != 150_000:
        raise RuntimeError(
            "SAFETY CHECK FAILED: expected exactly 150,000 timesteps "
            f"but checkpoint reports {loaded_steps:,}."
        )

    if model.device.type != "cuda":
        raise RuntimeError(
            f"SAFETY CHECK FAILED: model loaded on {model.device}, not CUDA."
        )

    print("[PASS] Correct 150K checkpoint loaded.")
    print("[PASS] PPO-Local remains 60-D.")
    print("[PASS] Action space remains MultiDiscrete([2,2,2,2]).")
    print("[PASS] PPO model is running on CUDA.")
    return model


# ============================================================================
# CHECKPOINT CALLBACK
# ============================================================================

def create_checkpoint_callback():

    callback = CheckpointCallback(
        save_freq=25_000,
        save_path=CHECKPOINT_SUBDIR,
        name_prefix="ppo_local_500k",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    return callback


# ============================================================================
# TRAIN
# ============================================================================

def train():

    print()
    print("=" * 78)
    print("MAI-07 | PPO-LOCAL 500K BASELINE TRAINING")
    print("=" * 78)

    print()
    print("Training configuration:")
    print(
        f"  Timesteps           : "
        f"{TOTAL_TIMESTEPS:,}"
    )
    print(
        f"  Global observation  : "
        f"{EXPECTED_GLOBAL_DIM}-D"
    )
    print(
        f"  Local observation   : "
        f"{EXPECTED_LOCAL_DIM}-D"
    )
    print(
        "  Action space        : "
        "MultiDiscrete([2,2,2,2])"
    )
    print(
        f"  Scenarios           : "
        f"{len(SCENARIOS)}"
    )
    print(
        f"  Episode steps       : "
        f"{MAX_EPISODE_STEPS}"
    )
    print(
        f"  Seed                : "
        f"{BASE_SEED}"
    )

    print()
    print("Scenarios:")

    for scenario in SCENARIOS:
        print(
            f"  - {scenario}"
        )

    # ----------------------------------------------------------------
    # Save metadata
    # ----------------------------------------------------------------

    save_training_information()

    # ----------------------------------------------------------------
    # Create environment
    # ----------------------------------------------------------------

    env = PPOLocalScenarioRotationEnv()

    env = Monitor(
        env,
        filename=os.path.join(
            LOG_DIR,
            "monitor.csv",
        ),
    )

    # ----------------------------------------------------------------
    # Validate spaces
    # ----------------------------------------------------------------

    if tuple(
        env.observation_space.shape
    ) != (
        EXPECTED_LOCAL_DIM,
    ):
        raise RuntimeError(
            "Training environment does not "
            "provide the required 60-D observation."
        )

    if not isinstance(
        env.action_space,
        gym.spaces.MultiDiscrete,
    ):
        raise RuntimeError(
            "Training environment action space "
            "must be MultiDiscrete."
        )

    if not np.array_equal(
        env.action_space.nvec,
        np.array([2, 2, 2, 2]),
    ):
        raise RuntimeError(
            "Training environment action space "
            "must be MultiDiscrete([2,2,2,2])."
        )

    # ----------------------------------------------------------------
    # GPU validation
    # ----------------------------------------------------------------

    device = get_training_device()

    # ----------------------------------------------------------------
    # Create OR resume PPO model
    # ----------------------------------------------------------------

    if RESUME_FROM_CHECKPOINT:
        model = load_resume_model(
            env,
            device,
        )
        training_steps_remaining = TOTAL_TIMESTEPS - int(model.num_timesteps)

        if training_steps_remaining <= 0:
            raise RuntimeError(
                "Checkpoint already contains 500,000 or more timesteps. "
                "No additional training is required."
            )

        print()
        print(
            f"Additional timesteps required: "
            f"{training_steps_remaining:,}"
        )
    else:
        model = create_model(
            env,
            device,
        )
        training_steps_remaining = TOTAL_TIMESTEPS

    # ----------------------------------------------------------------
    # Checkpointing
    # ----------------------------------------------------------------

    checkpoint_callback = (
        create_checkpoint_callback()
    )

    # ----------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------

    print()
    print("=" * 78)
    print("STARTING PPO-LOCAL 500K TRAINING")
    print("=" * 78)

    print()
    print(
        "Observation: 120-D environment "
        "-> 60-D local PPO input"
    )

    print(
        "Checkpoint interval: 25,000 timesteps"
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
        "Final model:"
    )

    print(
        FINAL_MODEL
    )

    print()

    start_time = time.time()

    try:

        model.learn(
            total_timesteps=training_steps_remaining,
            callback=checkpoint_callback,
            progress_bar=True,
            # CRITICAL: preserve the loaded 150K timestep counter.
            reset_num_timesteps=False,
        )

    except KeyboardInterrupt:

        print()
        print("=" * 78)
        print("PPO-LOCAL TRAINING INTERRUPTED")
        print("=" * 78)

        interrupted_path = os.path.join(
            CHECKPOINT_DIR,
            "ppo_local_500k_interrupted",
        )

        model.save(
            interrupted_path
        )

        print()
        print(
            "Interrupted model saved to:"
        )

        print(
            interrupted_path + ".zip"
        )

        env.close()

        return

    except Exception as error:

        print()
        print("=" * 78)
        print("PPO-LOCAL TRAINING STOPPED")
        print("=" * 78)

        print()
        print(
            type(error).__name__,
            ":",
            error,
        )

        env.close()

        raise

    # ----------------------------------------------------------------
    # Save final model
    # ----------------------------------------------------------------

    model.save(
        FINAL_MODEL
    )

    elapsed = (
        time.time()
        - start_time
    )

    # ----------------------------------------------------------------
    # Update metadata
    # ----------------------------------------------------------------

    save_training_information()

    # ----------------------------------------------------------------
    # Final output
    # ----------------------------------------------------------------

    print()
    print("=" * 78)
    print("PPO-LOCAL 500K TRAINING COMPLETE")
    print("=" * 78)

    print()
    print(
        f"Timesteps requested : "
        f"{TOTAL_TIMESTEPS:,}"
    )

    print(
        f"Training time       : "
        f"{elapsed / 3600.0:.2f} hours"
    )

    print()
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
        "TensorBoard / logs:"
    )

    print(
        LOG_DIR
    )

    print()
    print("=" * 78)
    print("PPO-LOCAL BASELINE READY FOR EVALUATION")
    print("=" * 78)

    env.close()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    validate_environment()

    print()
    print(
        "=" * 78
    )

    print(
        "Environment validation passed."
    )

    print(
        "=" * 78
    )

    print()
    print(
        "Starting PPO-Local 500K training..."
    )

    print()

    train()