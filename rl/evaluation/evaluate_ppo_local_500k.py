"""
==============================================================================
MAI-07 | PPO-LOCAL 500K BASELINE EVALUATION
==============================================================================

PPO-Local:
    - 500,000 training timesteps
    - CURRENT TrafficEnvironment observation = 120-D
    - PPO-Local model observation = 60-D
    - 120-D -> 60-D local observation adapter
    - MultiDiscrete([2,2,2,2]) action space
    - deterministic inference

Evaluation:
    - 10 scenarios
    - 3 episodes per scenario
    - 30 total episodes
    - 300 maximum steps per episode
    - seeds 42, 43, 44
    - no hardcoded performance values

IMPORTANT
---------
The current TrafficEnvironment returns:

    Agent A = 15 local + 15 neighbour = 30
    Agent B = 15 local + 15 neighbour = 30
    Agent C = 15 local + 15 neighbour = 30
    Agent D = 15 local + 15 neighbour = 30

Total = 120-D

PPO-Local was trained using only:

    A local 15
    B local 15
    C local 15
    D local 15

Total = 60-D

Therefore:

    120-D environment observation
                  |
                  v
        remove neighbour features
                  |
                  v
        60-D PPO-Local observation
                  |
                  v
             PPO-Local

Run from project root:

    python .\rl\evaluation\evaluate_ppo_local_500k.py
==============================================================================
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


# ============================================================================
# PROJECT ROOT
# ============================================================================

# This file is:
#
# traffic-marl/
#     rl/
#         evaluation/
#             evaluate_ppo_local_500k.py
#
# parents[0] = evaluation
# parents[1] = rl
# parents[2] = traffic-marl
#
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# PROJECT IMPORT
# ============================================================================

from rl.environments.traffic_env import TrafficEnvironment


# ============================================================================
# PATHS
# ============================================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "rl"
    / "checkpoints"
    / "ppo_local_500k"
    / "ppo_local_500k_final.zip"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "rl"
    / "results"
    / "ppo_local_500k"
)

RAW_FILE = (
    OUTPUT_DIR
    / "ppo_local_500k_raw.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "ppo_local_500k_summary.csv"
)

SCENARIO_FILE = (
    OUTPUT_DIR
    / "ppo_local_500k_scenario_summary.csv"
)

JSON_FILE = (
    OUTPUT_DIR
    / "ppo_local_500k_report.json"
)


# ============================================================================
# EXPERIMENT CONFIGURATION
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

EPISODES_PER_SCENARIO = 3
MAX_STEPS = 300
BASE_SEED = 42

# Current TrafficEnvironment
EXPECTED_GLOBAL_DIM = 120

# PPO-Local trained model
EXPECTED_LOCAL_DIM = 60

NUMBER_OF_AGENTS = 4
LOCAL_FEATURES_PER_AGENT = 15
GRAPH_FEATURES_PER_AGENT = 30

EXPECTED_ACTION_NVEC = np.array(
    [2, 2, 2, 2],
    dtype=np.int64,
)


# ============================================================================
# OUTPUT DIRECTORY
# ============================================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================================
# HELPER
# ============================================================================

def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def numeric(
    value: Any,
    default: float = 0.0,
) -> float:

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return default


# ============================================================================
# 120-D -> 60-D PPO-LOCAL ADAPTER
# ============================================================================

def convert_120_to_60(
    observation: Any,
) -> np.ndarray:
    """
    Convert current 120-D environment observation
    into the 60-D observation expected by PPO-Local.

    Layout:

        Agent 0:
            0:15   = local
            15:30  = neighbour

        Agent 1:
            30:45  = local
            45:60  = neighbour

        Agent 2:
            60:75  = local
            75:90  = neighbour

        Agent 3:
            90:105 = local
            105:120 = neighbour

    PPO-Local receives:

        0:15
        30:45
        60:75
        90:105

    Total = 60 features.
    """

    observation = np.asarray(
        observation,
        dtype=np.float32,
    ).reshape(-1)

    # --------------------------------------------------------------
    # Raw environment must be 120-D
    # --------------------------------------------------------------

    if observation.shape != (
        EXPECTED_GLOBAL_DIM,
    ):
        raise RuntimeError(
            "PPO-Local observation adapter error.\n"
            f"Expected environment observation: "
            f"({EXPECTED_GLOBAL_DIM},)\n"
            f"Received: {observation.shape}"
        )

    local_parts = []

    # --------------------------------------------------------------
    # Extract local 15 features from each 30-D agent block
    # --------------------------------------------------------------

    for agent_index in range(
        NUMBER_OF_AGENTS
    ):

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
            raise RuntimeError(
                f"Invalid agent block for agent "
                f"{agent_index}.\n"
                f"Expected: "
                f"({GRAPH_FEATURES_PER_AGENT},)\n"
                f"Received: {agent_block.shape}"
            )

        # First 15 = local information
        local_part = agent_block[
            :LOCAL_FEATURES_PER_AGENT
        ]

        if local_part.shape != (
            LOCAL_FEATURES_PER_AGENT,
        ):
            raise RuntimeError(
                f"Invalid local block for agent "
                f"{agent_index}.\n"
                f"Expected: "
                f"({LOCAL_FEATURES_PER_AGENT},)\n"
                f"Received: {local_part.shape}"
            )

        local_parts.append(
            local_part
        )

    # --------------------------------------------------------------
    # Combine four local blocks
    # --------------------------------------------------------------

    local_observation = np.concatenate(
        local_parts
    ).astype(
        np.float32
    )

    # --------------------------------------------------------------
    # Final validation
    # --------------------------------------------------------------

    if local_observation.shape != (
        EXPECTED_LOCAL_DIM,
    ):
        raise RuntimeError(
            "PPO-Local observation conversion failed.\n"
            f"Expected: "
            f"({EXPECTED_LOCAL_DIM},)\n"
            f"Received: {local_observation.shape}"
        )

    return local_observation


# ============================================================================
# ACTION VALIDATION
# ============================================================================

def validate_action(
    action: Any,
) -> np.ndarray:
    """
    Validate PPO-Local MultiDiscrete action.

    Expected:

        [0, 0, 0, 0]
        [0, 1, 0, 1]
        [1, 1, 1, 1]

    etc.
    """

    actions = np.asarray(
        action,
        dtype=np.int64,
    ).reshape(-1)

    if actions.shape != (
        NUMBER_OF_AGENTS,
    ):
        raise RuntimeError(
            "Invalid PPO-Local action shape.\n"
            f"Expected: "
            f"({NUMBER_OF_AGENTS},)\n"
            f"Received: {actions.shape}\n"
            f"Action: {actions}"
        )

    if not np.all(
        np.isin(
            actions,
            [0, 1],
        )
    ):
        raise RuntimeError(
            "Invalid PPO-Local action values.\n"
            "Expected only 0 or 1.\n"
            f"Received: {actions}"
        )

    return actions


# ============================================================================
# MODEL VALIDATION
# ============================================================================

def load_and_validate_model() -> PPO:

    print()
    print("=" * 78)
    print("PPO-LOCAL MODEL VALIDATION")
    print("=" * 78)

    print()
    print("Model:")
    print(MODEL_PATH)

    # --------------------------------------------------------------
    # Check model file
    # --------------------------------------------------------------

    if not MODEL_PATH.is_file():

        raise FileNotFoundError(
            "\nPPO-Local 500K final model was not found:\n"
            f"{MODEL_PATH}\n"
        )

    # --------------------------------------------------------------
    # Load model
    # --------------------------------------------------------------

    # CPU is intentional.
    # SUMO/TraCI is CPU based and PPO inference is lightweight.
    model = PPO.load(
        str(MODEL_PATH),
        device="cpu",
    )

    # --------------------------------------------------------------
    # Model observation space
    # --------------------------------------------------------------

    observation_shape = tuple(
        model.observation_space.shape
    )

    if observation_shape != (
        EXPECTED_LOCAL_DIM,
    ):

        raise RuntimeError(
            "PPO-Local model observation space mismatch.\n"
            f"Expected trained model input: "
            f"({EXPECTED_LOCAL_DIM},)\n"
            f"Detected: {observation_shape}"
        )

    # --------------------------------------------------------------
    # Model action space
    # --------------------------------------------------------------

    if not isinstance(
        model.action_space,
        gym.spaces.MultiDiscrete,
    ):

        raise RuntimeError(
            "PPO-Local model action space is not "
            "MultiDiscrete.\n"
            f"Detected: {model.action_space}"
        )

    action_nvec = np.asarray(
        model.action_space.nvec,
        dtype=np.int64,
    )

    if not np.array_equal(
        action_nvec,
        EXPECTED_ACTION_NVEC,
    ):

        raise RuntimeError(
            "PPO-Local model action space mismatch.\n"
            f"Expected: "
            f"MultiDiscrete({EXPECTED_ACTION_NVEC.tolist()})\n"
            f"Detected: "
            f"MultiDiscrete({action_nvec.tolist()})"
        )

    print()
    print(
        "[PASS] PPO-Local model file found."
    )

    print(
        "[PASS] Model observation: "
        f"{observation_shape}"
    )

    print(
        "[PASS] Model action space: "
        f"MultiDiscrete({action_nvec.tolist()})"
    )

    print(
        "[PASS] Model loaded successfully."
    )

    return model


# ============================================================================
# ENVIRONMENT VALIDATION
# ============================================================================

def validate_environment() -> None:
    """
    IMPORTANT:

    Raw TrafficEnvironment = 120-D.

    PPO-Local model input = 60-D.

    We validate BOTH sides and the adapter.
    """

    print()
    print("=" * 78)
    print(
        "PPO-LOCAL EVALUATION ENVIRONMENT VALIDATION"
    )
    print("=" * 78)

    for scenario in SCENARIOS:

        env = TrafficEnvironment(
            scenario_name=scenario,
            render_mode=None,
        )

        try:

            # ----------------------------------------------------------
            # Environment observation space
            # ----------------------------------------------------------

            env_shape = tuple(
                env.observation_space.shape
            )

            if env_shape != (
                EXPECTED_GLOBAL_DIM,
            ):

                raise RuntimeError(
                    f"{scenario}: "
                    "TrafficEnvironment observation mismatch.\n"
                    f"Expected: "
                    f"({EXPECTED_GLOBAL_DIM},)\n"
                    f"Detected: {env_shape}"
                )

            # ----------------------------------------------------------
            # Reset
            # ----------------------------------------------------------

            raw_observation, info = (
                env.reset(
                    seed=BASE_SEED
                )
            )

            raw_observation = np.asarray(
                raw_observation,
                dtype=np.float32,
            )

            # ----------------------------------------------------------
            # Check raw 120-D
            # ----------------------------------------------------------

            if raw_observation.shape != (
                EXPECTED_GLOBAL_DIM,
            ):

                raise RuntimeError(
                    f"{scenario}: "
                    "environment reset returned "
                    f"{raw_observation.shape}.\n"
                    f"Expected: "
                    f"({EXPECTED_GLOBAL_DIM},)"
                )

            # ----------------------------------------------------------
            # Convert 120 -> 60
            # ----------------------------------------------------------

            local_observation = (
                convert_120_to_60(
                    raw_observation
                )
            )

            # ----------------------------------------------------------
            # Check converted observation
            # ----------------------------------------------------------

            if local_observation.shape != (
                EXPECTED_LOCAL_DIM,
            ):

                raise RuntimeError(
                    f"{scenario}: "
                    "adapter returned "
                    f"{local_observation.shape}.\n"
                    f"Expected: "
                    f"({EXPECTED_LOCAL_DIM},)"
                )

            print(
                f"  [PASS] {scenario}: "
                f"environment "
                f"{raw_observation.shape} "
                f"-> PPO-Local "
                f"{local_observation.shape}"
            )

        finally:

            try:
                env.close()

            except Exception:
                pass

    print()
    print(
        "[PASS] All 10 scenarios validated."
    )

    print(
        "[PASS] 120-D -> 60-D PPO-Local "
        "observation adapter validated."
    )


# ============================================================================
# SUMO METRICS
# ============================================================================

def extract_metrics() -> dict[str, float]:
    """
    Read live traffic metrics directly from SUMO/TraCI.
    """

    import traci

    vehicle_ids = (
        traci.vehicle.getIDList()
    )

    if not vehicle_ids:

        return {
            "waiting_time": 0.0,
            "queue_length": 0.0,
            "average_speed": 0.0,
        }

    speeds = []
    waiting_times = []

    stopped = 0

    for vehicle_id in vehicle_ids:

        try:

            speed = numeric(
                traci.vehicle.getSpeed(
                    vehicle_id
                )
            )

            waiting = numeric(
                traci.vehicle.getWaitingTime(
                    vehicle_id
                )
            )

        except Exception:

            continue

        speeds.append(
            speed
        )

        waiting_times.append(
            waiting
        )

        if speed <= 0.1:

            stopped += 1

    if not speeds:

        return {
            "waiting_time": 0.0,
            "queue_length": 0.0,
            "average_speed": 0.0,
        }

    return {
        "waiting_time": float(
            np.sum(
                waiting_times
            )
        ),
        "queue_length": float(
            stopped
        ),
        "average_speed": float(
            np.mean(
                speeds
            )
        ),
    }


# ============================================================================
# SINGLE EPISODE
# ============================================================================

def evaluate_episode(
    model: PPO,
    scenario: str,
    episode: int,
) -> dict[str, Any]:

    seed = (
        BASE_SEED
        + episode
        - 1
    )

    print()
    print(
        f"  Episode {episode}/"
        f"{EPISODES_PER_SCENARIO}"
        f" | Scenario: {scenario}"
        f" | Seed: {seed}"
    )

    env = TrafficEnvironment(
        scenario_name=scenario,
        render_mode=None,
    )

    start_time = time.perf_counter()

    try:

        # --------------------------------------------------------------
        # RESET
        # --------------------------------------------------------------

        raw_observation, info = (
            env.reset(
                seed=seed
            )
        )

        # CRITICAL:
        #
        # raw_observation = 120-D
        #
        # PPO-Local requires:
        #
        # observation = 60-D
        #
        observation = convert_120_to_60(
            raw_observation
        )

        # --------------------------------------------------------------
        # Accumulators
        # --------------------------------------------------------------

        total_reward = 0.0

        waiting_values = []
        queue_values = []
        speed_values = []

        throughput = 0

        steps = 0

        terminated = False
        truncated = False

        # --------------------------------------------------------------
        # SIMULATION LOOP
        # --------------------------------------------------------------

        for step_number in range(
            1,
            MAX_STEPS + 1,
        ):

            steps = step_number

            # ==========================================================
            # PPO-LOCAL PREDICTION
            # ==========================================================

            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            # ==========================================================
            # Validate action
            # ==========================================================

            joint_action = validate_action(
                action
            )

            # ==========================================================
            # Environment step
            # ==========================================================

            (
                raw_next_observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                joint_action
            )

            # ==========================================================
            # CRITICAL:
            #
            # Environment returns 120-D again.
            #
            # Convert it back to 60-D for next PPO prediction.
            # ==========================================================

            observation = convert_120_to_60(
                raw_next_observation
            )

            # ==========================================================
            # Reward
            # ==========================================================

            total_reward += numeric(
                reward
            )

            # ==========================================================
            # SUMO metrics
            # ==========================================================

            metrics = extract_metrics()

            waiting_values.append(
                metrics[
                    "waiting_time"
                ]
            )

            queue_values.append(
                metrics[
                    "queue_length"
                ]
            )

            speed_values.append(
                metrics[
                    "average_speed"
                ]
            )

            # ==========================================================
            # Throughput
            # ==========================================================

            try:

                import traci

                arrived = int(
                    traci.simulation.getArrivedNumber()
                )

                throughput = max(
                    throughput,
                    arrived,
                )

            except Exception:

                pass

            # ==========================================================
            # End episode
            # ==========================================================

            if terminated or truncated:

                break

        evaluation_time = (
            time.perf_counter()
            - start_time
        )

        # --------------------------------------------------------------
        # Build result
        # --------------------------------------------------------------

        result = {

            "policy":
                "ppo_local_500k",

            "model":
                "ppo_local_500k_final",

            "observation_type":
                "Local-60D",

            "environment_observation_type":
                "Global-120D",

            "action_space":
                "MultiDiscrete([2,2,2,2])",

            "scenario":
                scenario,

            "episode":
                episode,

            "seed":
                seed,

            "reward":
                float(
                    total_reward
                ),

            "waiting_time":
                float(
                    np.mean(
                        waiting_values
                    )
                )
                if waiting_values
                else 0.0,

            "queue_length":
                float(
                    np.mean(
                        queue_values
                    )
                )
                if queue_values
                else 0.0,

            "average_speed":
                float(
                    np.mean(
                        speed_values
                    )
                )
                if speed_values
                else 0.0,

            "throughput":
                int(
                    throughput
                ),

            "steps":
                int(
                    steps
                ),

            "evaluation_time":
                float(
                    evaluation_time
                ),

            "terminated":
                bool(
                    terminated
                ),

            "truncated":
                bool(
                    truncated
                ),

            "error_type":
                "",

            "error":
                "",

            "timestamp":
                utc_now(),
        }

        print(
            f"    reward="
            f"{result['reward']:.2f}"
            f" | wait="
            f"{result['waiting_time']:.2f}"
            f" | queue="
            f"{result['queue_length']:.2f}"
            f" | speed="
            f"{result['average_speed']:.2f}"
            f" | throughput="
            f"{result['throughput']}"
            f" | steps="
            f"{result['steps']}"
        )

        return result

    finally:

        try:
            env.close()

        except Exception:
            pass


# ============================================================================
# FULL EVALUATION
# ============================================================================

def evaluate_all(
    model: PPO,
) -> list[dict[str, Any]]:

    print()
    print("=" * 78)
    print("PPO-LOCAL 500K EVALUATION")
    print("=" * 78)

    print()
    print(
        "Model:"
    )
    print(
        MODEL_PATH
    )

    print()
    print(
        "Evaluation configuration:"
    )

    print(
        f"  Scenarios             : "
        f"{len(SCENARIOS)}"
    )

    print(
        f"  Episodes/scenario    : "
        f"{EPISODES_PER_SCENARIO}"
    )

    print(
        f"  Total episodes       : "
        f"{len(SCENARIOS) * EPISODES_PER_SCENARIO}"
    )

    print(
        f"  Max episode steps    : "
        f"{MAX_STEPS}"
    )

    print(
        f"  Seeds                : "
        f"{BASE_SEED}, "
        f"{BASE_SEED + 1}, "
        f"{BASE_SEED + 2}"
    )

    print(
        "  Inference             : deterministic"
    )

    print(
        "  Environment input     : 120-D"
    )

    print(
        "  PPO-Local input       : 60-D"
    )

    print(
        "  Observation adapter   : 120-D -> 60-D"
    )

    print(
        "  Action                : "
        "MultiDiscrete([2,2,2,2])"
    )

    print(
        "  Hardcoded results     : none"
    )

    all_results = []

    # --------------------------------------------------------------
    # Scenarios
    # --------------------------------------------------------------

    for scenario_number, scenario in enumerate(
        SCENARIOS,
        start=1,
    ):

        print()
        print("-" * 78)

        print(
            f"[{scenario_number}/"
            f"{len(SCENARIOS)}] "
            f"{scenario}"
        )

        print(
            "-" * 78
        )

        # ----------------------------------------------------------
        # Episodes
        # ----------------------------------------------------------

        for episode in range(
            1,
            EPISODES_PER_SCENARIO + 1,
        ):

            try:

                result = evaluate_episode(
                    model=model,
                    scenario=scenario,
                    episode=episode,
                )

                all_results.append(
                    result
                )

            except Exception as exc:

                error_result = {

                    "policy":
                        "ppo_local_500k",

                    "model":
                        "ppo_local_500k_final",

                    "observation_type":
                        "Local-60D",

                    "environment_observation_type":
                        "Global-120D",

                    "action_space":
                        "MultiDiscrete([2,2,2,2])",

                    "scenario":
                        scenario,

                    "episode":
                        episode,

                    "seed":
                        BASE_SEED
                        + episode
                        - 1,

                    "reward":
                        np.nan,

                    "waiting_time":
                        np.nan,

                    "queue_length":
                        np.nan,

                    "average_speed":
                        np.nan,

                    "throughput":
                        np.nan,

                    "steps":
                        0,

                    "evaluation_time":
                        np.nan,

                    "terminated":
                        False,

                    "truncated":
                        False,

                    "error_type":
                        type(exc).__name__,

                    "error":
                        str(exc),

                    "timestamp":
                        utc_now(),
                }

                all_results.append(
                    error_result
                )

                print(
                    f"    [ERROR] "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

    return all_results


# ============================================================================
# RESULT VALIDATION
# ============================================================================

def validate_results(
    results: list[dict[str, Any]],
) -> None:

    expected_count = (
        len(SCENARIOS)
        * EPISODES_PER_SCENARIO
    )

    if len(results) != expected_count:

        raise RuntimeError(
            f"Expected {expected_count} "
            f"episode results, "
            f"received {len(results)}"
        )

    errors = [
        result
        for result in results
        if result.get("error")
    ]

    if errors:

        details = "\n".join(
            (
                f"  {result['scenario']} "
                f"episode {result['episode']}: "
                f"{result['error_type']} - "
                f"{result['error']}"
            )
            for result in errors
        )

        raise RuntimeError(
            "Evaluation completed with errors. "
            "Results were NOT accepted as a clean evaluation.\n"
            + details
        )

    actual_scenarios = {
        result["scenario"]
        for result in results
    }

    if actual_scenarios != set(
        SCENARIOS
    ):

        raise RuntimeError(
            "Evaluation did not cover exactly "
            "the configured 10 scenarios."
        )

    # --------------------------------------------------------------
    # Numeric validation
    # --------------------------------------------------------------

    for result in results:

        for metric in (
            "reward",
            "waiting_time",
            "queue_length",
            "average_speed",
            "throughput",
        ):

            if not np.isfinite(
                float(
                    result[metric]
                )
            ):

                raise RuntimeError(
                    f"Invalid metric "
                    f"'{metric}' in "
                    f"{result['scenario']} "
                    f"episode "
                    f"{result['episode']}"
                )

    print()
    print(
        "[PASS] All 30 episode results are valid."
    )


# ============================================================================
# SAVE RAW RESULTS
# ============================================================================

def save_raw_results(
    results: list[dict[str, Any]],
) -> None:

    field_names = [

        "policy",
        "model",

        "observation_type",
        "environment_observation_type",

        "action_space",

        "scenario",
        "episode",
        "seed",

        "reward",
        "waiting_time",
        "queue_length",
        "average_speed",
        "throughput",

        "steps",
        "evaluation_time",

        "terminated",
        "truncated",

        "error_type",
        "error",

        "timestamp",
    ]

    with RAW_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=field_names,
            extrasaction="ignore",
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    print()
    print(
        f"Raw results saved:\n"
        f"{RAW_FILE}"
    )


# ============================================================================
# CREATE SUMMARIES
# ============================================================================

def create_summaries(
    results: list[dict[str, Any]],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:

    scenario_summary = []

    # --------------------------------------------------------------
    # Scenario summaries
    # --------------------------------------------------------------

    for scenario in SCENARIOS:

        rows = [
            result
            for result in results
            if result["scenario"]
            == scenario
        ]

        scenario_summary.append(
            {

                "policy":
                    "ppo_local_500k",

                "scenario":
                    scenario,

                "episodes":
                    len(rows),

                "reward_mean":
                    float(
                        np.mean(
                            [
                                row["reward"]
                                for row in rows
                            ]
                        )
                    ),

                "reward_std":
                    float(
                        np.std(
                            [
                                row["reward"]
                                for row in rows
                            ]
                        )
                    ),

                "waiting_time_mean":
                    float(
                        np.mean(
                            [
                                row[
                                    "waiting_time"
                                ]
                                for row in rows
                            ]
                        )
                    ),

                "queue_length_mean":
                    float(
                        np.mean(
                            [
                                row[
                                    "queue_length"
                                ]
                                for row in rows
                            ]
                        )
                    ),

                "average_speed_mean":
                    float(
                        np.mean(
                            [
                                row[
                                    "average_speed"
                                ]
                                for row in rows
                            ]
                        )
                    ),

                "throughput_mean":
                    float(
                        np.mean(
                            [
                                row[
                                    "throughput"
                                ]
                                for row in rows
                            ]
                        )
                    ),

                "steps_mean":
                    float(
                        np.mean(
                            [
                                row["steps"]
                                for row in rows
                            ]
                        )
                    ),
            }
        )

    # --------------------------------------------------------------
    # Overall
    # --------------------------------------------------------------

    overall = {

        "policy":
            "ppo_local_500k",

        "model":
            "ppo_local_500k_final",

        "scenarios":
            len(SCENARIOS),

        "episodes_per_scenario":
            EPISODES_PER_SCENARIO,

        "total_episodes":
            len(results),

        "reward_mean":
            float(
                np.mean(
                    [
                        row["reward"]
                        for row in results
                    ]
                )
            ),

        "reward_std":
            float(
                np.std(
                    [
                        row["reward"]
                        for row in results
                    ]
                )
            ),

        "waiting_time_mean":
            float(
                np.mean(
                    [
                        row["waiting_time"]
                        for row in results
                    ]
                )
            ),

        "queue_length_mean":
            float(
                np.mean(
                    [
                        row["queue_length"]
                        for row in results
                    ]
                )
            ),

        "average_speed_mean":
            float(
                np.mean(
                    [
                        row["average_speed"]
                        for row in results
                    ]
                )
            ),

        "throughput_mean":
            float(
                np.mean(
                    [
                        row["throughput"]
                        for row in results
                    ]
                )
            ),

        "steps_mean":
            float(
                np.mean(
                    [
                        row["steps"]
                        for row in results
                    ]
                )
            ),

        "timestamp":
            utc_now(),
    }

    return (
        overall,
        scenario_summary,
    )


# ============================================================================
# SAVE SUMMARY CSV FILES
# ============================================================================

def save_summary_files(
    overall: dict[str, Any],
    scenario_summary: list[dict[str, Any]],
) -> None:

    # --------------------------------------------------------------
    # Overall
    # --------------------------------------------------------------

    with SUMMARY_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                overall.keys()
            ),
        )

        writer.writeheader()

        writer.writerow(
            overall
        )

    # --------------------------------------------------------------
    # Scenario
    # --------------------------------------------------------------

    with SCENARIO_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                scenario_summary[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            scenario_summary
        )

    print(
        f"Overall summary saved:\n"
        f"{SUMMARY_FILE}"
    )

    print(
        f"Scenario summary saved:\n"
        f"{SCENARIO_FILE}"
    )


# ============================================================================
# JSON REPORT
# ============================================================================

def save_json_report(
    results: list[dict[str, Any]],
    overall: dict[str, Any],
    scenario_summary: list[dict[str, Any]],
    elapsed_seconds: float,
) -> None:

    report = {

        "project":
            "MAI-07 | Traffic MARL",

        "evaluation":
            "PPO-Local 500K Baseline Evaluation",

        "model_path":
            str(MODEL_PATH),

        "model":
            "ppo_local_500k_final",

        "algorithm":
            "PPO-Local",

        "training_timesteps":
            500_000,

        "environment_observation_dimension":
            EXPECTED_GLOBAL_DIM,

        "model_observation_dimension":
            EXPECTED_LOCAL_DIM,

        "observation_type":
            "Local-60D",

        "observation_adapter":
            (
                "120-D current TrafficEnvironment "
                "observation converted to 60-D by "
                "retaining the first 15 local features "
                "from each of the four 30-D agent blocks "
                "and removing neighbour features."
            ),

        "action_space":
            "MultiDiscrete([2,2,2,2])",

        "number_of_agents":
            NUMBER_OF_AGENTS,

        "local_features_per_agent":
            LOCAL_FEATURES_PER_AGENT,

        "graph_features_per_agent":
            GRAPH_FEATURES_PER_AGENT,

        "scenarios":
            SCENARIOS,

        "episodes_per_scenario":
            EPISODES_PER_SCENARIO,

        "total_episodes":
            len(results),

        "max_episode_steps":
            MAX_STEPS,

        "base_seed":
            BASE_SEED,

        "seeds":
            [
                BASE_SEED + i
                for i in range(
                    EPISODES_PER_SCENARIO
                )
            ],

        "deterministic_inference":
            True,

        "metrics": {

            "reward":
                "Cumulative environment reward.",

            "waiting_time":
                (
                    "Mean per-step sum of current "
                    "vehicle waiting times measured "
                    "directly from SUMO/TraCI."
                ),

            "queue_length":
                (
                    "Mean per-step number of vehicles "
                    "with speed <= 0.1 m/s measured "
                    "directly from SUMO/TraCI."
                ),

            "average_speed":
                (
                    "Mean per-step average vehicle "
                    "speed measured directly from "
                    "SUMO/TraCI."
                ),

            "throughput":
                (
                    "Maximum cumulative SUMO arrived-"
                    "vehicle count observed during "
                    "each episode."
                ),
        },

        "overall":
            overall,

        "scenario_summary":
            scenario_summary,

        "episode_results":
            results,

        "evaluation_time_seconds":
            elapsed_seconds,

        "timestamp":
            utc_now(),
    }

    with JSON_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            allow_nan=False,
        )

    print(
        f"JSON report saved:\n"
        f"{JSON_FILE}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print()
    print("=" * 78)

    print(
        "MAI-07 | PPO-LOCAL 500K BASELINE EVALUATION"
    )

    print("=" * 78)

    print()
    print("Model:")
    print(
        MODEL_PATH
    )

    print()
    print("Evaluation:")
    print(
        "  10 scenarios"
    )
    print(
        "  3 episodes per scenario"
    )
    print(
        "  30 total episodes"
    )
    print(
        "  300 maximum steps per episode"
    )
    print(
        "  Seeds: 42, 43, 44"
    )
    print(
        "  Deterministic PPO inference"
    )
    print(
        "  Environment observation: 120-D"
    )
    print(
        "  PPO-Local model input: 60-D"
    )
    print(
        "  Observation adapter: 120-D -> 60-D"
    )
    print(
        "  Action: MultiDiscrete([2,2,2,2])"
    )
    print(
        "  No hardcoded performance values"
    )

    # ------------------------------------------------------------------------
    # Validate environment and adapter
    # ------------------------------------------------------------------------

    validate_environment()

    # ------------------------------------------------------------------------
    # Load and validate model
    # ------------------------------------------------------------------------

    model = load_and_validate_model()

    # ------------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------------

    start_time = time.perf_counter()

    results = evaluate_all(
        model
    )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    # ------------------------------------------------------------------------
    # Validate all results
    # ------------------------------------------------------------------------

    validate_results(
        results
    )

    # ------------------------------------------------------------------------
    # Save raw results
    # ------------------------------------------------------------------------

    save_raw_results(
        results
    )

    # ------------------------------------------------------------------------
    # Create summaries
    # ------------------------------------------------------------------------

    overall, scenario_summary = (
        create_summaries(
            results
        )
    )

    # ------------------------------------------------------------------------
    # Save CSV summaries
    # ------------------------------------------------------------------------

    save_summary_files(
        overall,
        scenario_summary,
    )

    # ------------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------------

    save_json_report(
        results,
        overall,
        scenario_summary,
        elapsed_seconds,
    )

    # ------------------------------------------------------------------------
    # Final output
    # ------------------------------------------------------------------------

    print()
    print("=" * 78)

    print(
        "PPO-LOCAL 500K BASELINE "
        "EVALUATION COMPLETE"
    )

    print("=" * 78)

    print()
    print(
        f"Evaluation time: "
        f"{elapsed_seconds / 60:.2f} minutes"
    )

    print()
    print(
        "OVERALL RESULTS"
    )

    print(
        "-" * 78
    )

    print(
        f"Reward          : "
        f"{overall['reward_mean']:.4f}"
    )

    print(
        f"Waiting Time    : "
        f"{overall['waiting_time_mean']:.4f}"
    )

    print(
        f"Queue Length    : "
        f"{overall['queue_length_mean']:.4f}"
    )

    print(
        f"Average Speed   : "
        f"{overall['average_speed_mean']:.4f}"
    )

    print(
        f"Throughput      : "
        f"{overall['throughput_mean']:.4f}"
    )

    print(
        f"Episode Steps   : "
        f"{overall['steps_mean']:.2f}"
    )

    # ------------------------------------------------------------------------
    # Scenario table
    # ------------------------------------------------------------------------

    print()
    print(
        "Scenario results:"
    )

    print(
        f"{'Scenario':<24}"
        f"{'Reward':>12}"
        f"{'Wait':>12}"
        f"{'Queue':>12}"
        f"{'Speed':>12}"
        f"{'Throughput':>12}"
    )

    print(
        "-" * 84
    )

    for row in scenario_summary:

        print(
            f"{row['scenario']:<24}"
            f"{row['reward_mean']:>12.2f}"
            f"{row['waiting_time_mean']:>12.2f}"
            f"{row['queue_length_mean']:>12.2f}"
            f"{row['average_speed_mean']:>12.2f}"
            f"{row['throughput_mean']:>12.2f}"
        )

    # ------------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------------

    print()
    print(
        "Files:"
    )

    print(
        f"  {RAW_FILE}"
    )

    print(
        f"  {SUMMARY_FILE}"
    )

    print(
        f"  {SCENARIO_FILE}"
    )

    print(
        f"  {JSON_FILE}"
    )

    print()
    print(
        "[PASS] PPO-Local 500K baseline "
        "is ready for comparison."
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()