"""
=========================================================
Traffic MARL Project

Module:
Environment Configuration

Description:
Central configuration for the Multi-Agent
Traffic Reinforcement Learning Environment.

Supports:

• Single Agent
• Multi Agent
• Future Network Scaling

Author:
Mohammed Piyarji

Week:
5

Task:
5.2

=========================================================
"""

from pathlib import Path


class EnvironmentConfig:
    """
    Central configuration class.

    Every environment module imports values from here.
    """

    # ==========================================================
    # PROJECT
    # ==========================================================

    PROJECT_NAME = "Traffic MARL"

    VERSION = "2.0"

    # ==========================================================
    # ROOT DIRECTORY
    # ==========================================================

    ROOT_DIR = Path(__file__).resolve().parents[2]

    # ==========================================================
    # SUMO
    # ==========================================================

    SUMO_BINARY = "sumo"

    NETWORK_FOLDER = (
        ROOT_DIR
        / "simulation"
        / "networks"
        / "four_intersection"
    )

    SUMO_CONFIG = (
        NETWORK_FOLDER
        / "simulation.sumocfg"
    )

    NETWORK_FILE = (
        NETWORK_FOLDER
        / "network.net.xml"
    )

    ROUTE_FILE = (
        NETWORK_FOLDER
        / "routes.rou.xml"
    )

    # ==========================================================
    # SIMULATION
    # ==========================================================

    MAX_EPISODE_STEPS = 300

    STEP_LENGTH = 1

    RANDOM_SEED = 42

    GUI = False

    # ==========================================================
    # MULTI AGENT SETTINGS
    # ==========================================================

    MULTI_AGENT = True

    # ==========================================================
    # TRAFFIC LIGHT IDS
    # (Verified from SUMO)
    # ==========================================================

    TRAFFIC_LIGHT_IDS = [
        "A",
        "B",
        "C",
        "D"
    ]

    NUMBER_OF_AGENTS = len(TRAFFIC_LIGHT_IDS)

    # ==========================================================
    # NEIGHBOUR GRAPH
    #
    #      A ----- B
    #      |       |
    #      |       |
    #      C ----- D
    #
    # ==========================================================

    NEIGHBOURS = {
        "A": [
            "B",
            "C"
        ],
        "B": [
            "A",
            "D"
        ],
        "C": [
            "A",
            "D"
        ],
        "D": [
            "B",
            "C"
        ]
    }

    # ==========================================================
    # OBSERVATION SPACE
    #
    # vehicle_count
    # average_speed
    # waiting_time
    # stopped
    # phase
    # north_queue
    # south_queue
    # east_queue
    # west_queue
    #
    # ==========================================================

    OBSERVATION_SIZE = 15

    # ==========================================================
    # ACTION SPACE
    # ==========================================================

    ACTION_NORTH_SOUTH = 0

    ACTION_EAST_WEST = 1

    NUMBER_OF_ACTIONS = 2

    ACTIONS = [
        ACTION_NORTH_SOUTH,
        ACTION_EAST_WEST
    ]

    # ==========================================================
    # TRAFFIC LIGHT PHASES
    # ==========================================================

    GREEN_PHASE_NS = 0

    YELLOW_PHASE_NS = 1

    GREEN_PHASE_EW = 2

    YELLOW_PHASE_EW = 3

    # ==========================================================
    # SAFETY CONSTRAINTS
    # ==========================================================

    MIN_GREEN_TIME = 10

    YELLOW_TIME = 3

    MAX_GREEN_TIME = 60

    # ==========================================================
    # REWARD SETTINGS
    # ==========================================================

    REWARD_SCALE = 1.0

    WAITING_TIME_WEIGHT = 1.0

    QUEUE_WEIGHT = 1.0

    SPEED_WEIGHT = 0.50

    STOPPED_WEIGHT = 0.50

    # ==========================================================
    # TRAINING
    # ==========================================================

    PPO_POLICY = "MlpPolicy"

    LEARNING_RATE = 3e-4

    GAMMA = 0.99

    N_STEPS = 1024

    BATCH_SIZE = 128

    TOTAL_TIMESTEPS = 250000

    ENT_COEF = 0.01

    GAE_LAMBDA = 0.95

    CLIP_RANGE = 0.2

    # ==========================================================
    # MODEL PATH
    # ==========================================================

    MODEL_NAME = "ppo_multi_intersection"

    MODEL_DIR = (
        ROOT_DIR
        / "rl"
        / "checkpoints"
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MODEL_PATH = (
        MODEL_DIR
        / f"{MODEL_NAME}.zip"
    )

    # ==========================================================
    # LOGGING
    # ==========================================================

    LOG_DIR = (
        ROOT_DIR
        / "rl"
        / "logs"
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    EVALUATION_DIR = (
        ROOT_DIR
        / "rl"
        / "evaluation"
    )

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_FILE = (
        EVALUATION_DIR
        / "ppo_results.csv"
    )

    # ==========================================================
    # MLFLOW
    # ==========================================================

    EXPERIMENT_NAME = "Traffic MARL Multi-Agent PPO"

    # ==========================================================
    # DEBUG
    # ==========================================================

    PRINT_ACTIONS = False

    PRINT_OBSERVATIONS = False

    PRINT_REWARDS = False

    PRINT_PHASE_CHANGES = False