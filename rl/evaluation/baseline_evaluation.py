"""
=========================================================
Traffic MARL Project

Module:
Baseline Policy Evaluation

Purpose:
Evaluate classical traffic-control baselines before
comparing them with the MARL/PPO controller.

Baseline Policies:

    1. Fixed-Time
    2. Random
    3. Queue-Based

Scenarios:

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

Metrics:

    Reward
    Waiting Time
    Queue Length
    Average Speed
    Throughput
    Episode Length

No PPO model is used.

=========================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import csv
import json
import time
from typing import Dict, Any, List

import numpy as np
import traci

from rl.environments.config import EnvironmentConfig
from rl.environments.traffic_env import TrafficEnvironment


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(
    EnvironmentConfig.ROOT_DIR
)

EVALUATION_DIR = (
    PROJECT_ROOT / "evaluation"
)

EVALUATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


RESULTS_CSV = (
    EVALUATION_DIR
    / "baseline_evaluation_results.csv"
)

SUMMARY_CSV = (
    EVALUATION_DIR
    / "baseline_evaluation_summary.csv"
)

RESULTS_JSON = (
    EVALUATION_DIR
    / "baseline_evaluation_results.json"
)


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
# BASELINE POLICIES
# =========================================================

POLICIES = [

    "fixed_time",

    "random",

    "queue_based",

]


# =========================================================
# SETTINGS
# =========================================================

EPISODES_PER_SCENARIO = 3

RANDOM_SEED = 42

MAX_STEPS = int(
    EnvironmentConfig.MAX_EPISODE_STEPS
)


# =========================================================
# AGENT IDS
# =========================================================

AGENTS = list(
    EnvironmentConfig.TRAFFIC_LIGHT_IDS
)


# =========================================================
# ACTION CONSTANTS
# =========================================================

ACTION_NS = int(
    EnvironmentConfig.ACTION_NORTH_SOUTH
)

ACTION_EW = int(
    EnvironmentConfig.ACTION_EAST_WEST
)


# =========================================================
# NUMERIC HELPER
# =========================================================

def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:

        value = float(value)

        if np.isfinite(value):

            return value

    except Exception:

        pass

    return default


# =========================================================
# EXTRACT AGENT OBSERVATIONS
# =========================================================

def split_global_observation(
    observation: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Split global observation into one observation
    per traffic-light agent.

    Current project:

        Global observation = 60

        A = 15
        B = 15
        C = 15
        D = 15
    """

    observation = np.asarray(
        observation,
        dtype=np.float32
    ).flatten()

    number_of_agents = len(
        AGENTS
    )

    if number_of_agents == 0:

        raise ValueError(
            "No traffic-light agents configured."
        )

    if (
        observation.size
        % number_of_agents
        != 0
    ):

        raise ValueError(
            "Global observation cannot be "
            "evenly divided between agents. "
            f"Size={observation.size}, "
            f"Agents={number_of_agents}"
        )

    agent_size = (
        observation.size
        // number_of_agents
    )

    result = {}

    for index, agent_id in enumerate(
        AGENTS
    ):

        start = (
            index
            * agent_size
        )

        end = (
            start
            + agent_size
        )

        result[agent_id] = (
            observation[start:end].copy()
        )

    return result


# =========================================================
# FIXED-TIME POLICY
# =========================================================

def fixed_time_actions(
    step: int,
) -> np.ndarray:
    """
    Fixed-time policy.

    The existing project test demonstrates:

        First cycle:
            A=0
            B=1
            C=0
            D=1

        Second cycle:
            A=1
            B=0
            C=1
            D=0

    Cycle length:
        10 steps
    """

    cycle_length = 10

    cycle = (
        step
        // cycle_length
    ) % 2

    actions = []

    for index, _agent in enumerate(
        AGENTS
    ):

        base_action = (
            ACTION_NS
            if index % 2 == 0
            else ACTION_EW
        )

        if cycle == 0:

            action = base_action

        else:

            if base_action == ACTION_NS:

                action = ACTION_EW

            else:

                action = ACTION_NS

        actions.append(
            action
        )

    return np.asarray(
        actions,
        dtype=np.int32
    )


# =========================================================
# RANDOM POLICY
# =========================================================

def random_actions(
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Reproducible random baseline.

    Only valid project actions are generated:

        ACTION_NORTH_SOUTH
        ACTION_EAST_WEST
    """

    return rng.integers(
        low=0,
        high=EnvironmentConfig.NUMBER_OF_ACTIONS,
        size=len(AGENTS),
        dtype=np.int32,
    )


# =========================================================
# QUEUE-BASED POLICY
# =========================================================

def queue_based_actions(
    observation: np.ndarray,
) -> np.ndarray:
    """
    Adaptive queue-based controller.

    Observation layout from the project:

        [0] vehicle count
        [1] average speed
        [2] waiting time
        [3] stopped vehicles
        [4] traffic-light phase
        [5] north queue
        [6] east queue
        [7] south queue
        [8] west queue

    Decision:

        NS queue > EW queue
            -> ACTION_NORTH_SOUTH

        EW queue > NS queue
            -> ACTION_EAST_WEST
    """

    agent_observations = (
        split_global_observation(
            observation
        )
    )

    actions = []

    for agent_id in AGENTS:

        agent_obs = (
            agent_observations[
                agent_id
            ]
        )

        if agent_obs.size < 9:

            raise ValueError(
                f"Agent {agent_id} observation "
                "contains fewer than 9 values."
            )

        north = safe_float(
            agent_obs[5]
        )

        east = safe_float(
            agent_obs[6]
        )

        south = safe_float(
            agent_obs[7]
        )

        west = safe_float(
            agent_obs[8]
        )

        ns_queue = (
            north
            + south
        )

        ew_queue = (
            east
            + west
        )

        if ns_queue >= ew_queue:

            action = ACTION_NS

        else:

            action = ACTION_EW

        actions.append(
            action
        )

    return np.asarray(
        actions,
        dtype=np.int32
    )


# =========================================================
# GENERATE POLICY ACTION
# =========================================================

def get_actions(
    policy_name: str,
    step: int,
    observation: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate actions for the requested baseline.
    """

    if policy_name == "fixed_time":

        return fixed_time_actions(
            step
        )

    if policy_name == "random":

        return random_actions(
            rng
        )

    if policy_name == "queue_based":

        return queue_based_actions(
            observation
        )

    raise ValueError(
        f"Unknown policy: {policy_name}"
    )


# =========================================================
# VALIDATE ACTIONS
# =========================================================

def validate_actions(
    actions: np.ndarray,
) -> np.ndarray:
    """
    Ensure actions match the project's
    MultiDiscrete action space.
    """

    actions = np.asarray(
        actions,
        dtype=np.int32
    ).flatten()

    expected = (
        EnvironmentConfig.NUMBER_OF_AGENTS
    )

    if actions.size != expected:

        raise ValueError(
            "Invalid action count. "
            f"Expected {expected}, "
            f"received {actions.size}."
        )

    if np.any(
        actions < 0
    ):

        raise ValueError(
            "Negative action detected."
        )

    if np.any(
        actions
        >= EnvironmentConfig.NUMBER_OF_ACTIONS
    ):

        raise ValueError(
            "Action outside configured action space."
        )

    return actions


# =========================================================
# CREATE METRIC RECORD
# =========================================================

def empty_result(
    scenario: str,
    policy: str,
    episode: int,
) -> Dict[str, Any]:

    return {

        "scenario":
            scenario,

        "policy":
            policy,

        "episode":
            episode,

        "reward":
            0.0,

        "waiting_time":
            0.0,

        "queue_length":
            0.0,

        "average_speed":
            0.0,

        "throughput":
            0,

        "steps":
            0,

        "evaluation_time":
            0.0,

        "terminated":
            False,

        "truncated":
            False,

    }


# =========================================================
# RUN ONE EPISODE
# =========================================================

# =========================================================
# RUN ONE EPISODE
# =========================================================

def run_episode(
    scenario: str,
    policy_name: str,
    episode: int,
) -> Dict[str, Any]:
    """
    Run one baseline-policy episode.

    Important:
    SUMO's getArrivedNumber() reports vehicles that
    arrived during the CURRENT simulation step.

    Therefore throughput must be accumulated over
    every simulation step.

    This prevents the previous incorrect throughput
    values such as 0, 1 or 2 vehicles.
    """

    result = empty_result(
        scenario,
        policy_name,
        episode,
    )

    start_time = time.perf_counter()

    seed = (
        RANDOM_SEED
        + episode
    )

    rng = np.random.default_rng(
        seed
    )

    env = TrafficEnvironment(
        scenario_name=scenario
    )

    # =====================================================
    # EPISODE ACCUMULATORS
    # =====================================================

    total_reward = 0.0

    total_waiting = 0.0

    total_queue = 0.0

    total_speed = 0.0

    total_arrived = 0

    step = 0

    terminated = False

    truncated = False

    try:

        # =================================================
        # RESET
        # =================================================

        observation, reset_info = (
            env.reset(
                seed=seed
            )
        )

        # =================================================
        # SIMULATION LOOP
        # =================================================

        while (
            not terminated
            and
            not truncated
            and
            step < MAX_STEPS
        ):

            # ---------------------------------------------
            # Generate baseline action
            # ---------------------------------------------

            actions = get_actions(
                policy_name=policy_name,
                step=step,
                observation=observation,
                rng=rng,
            )

            # ---------------------------------------------
            # Validate action
            # ---------------------------------------------

            actions = validate_actions(
                actions
            )

            # ---------------------------------------------
            # Environment step
            # ---------------------------------------------

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                actions
            )

            # =================================================
            # REWARD
            # =================================================

            total_reward += safe_float(
                reward
            )

            # =================================================
            # NETWORK WAITING TIME
            # =================================================

            total_waiting += safe_float(
                info.get(
                    "network_waiting_time",
                    0.0
                )
            )

            # =================================================
            # NETWORK QUEUE
            # =================================================

            total_queue += safe_float(
                info.get(
                    "network_queue_length",
                    0.0
                )
            )

            # =================================================
            # NETWORK SPEED
            # =================================================

            total_speed += safe_float(
                info.get(
                    "network_average_speed",
                    0.0
                )
            )

            # =================================================
            # THROUGHPUT
            # =================================================
            #
            # IMPORTANT:
            #
            # getArrivedNumber() is the number of vehicles
            # that arrived DURING THIS STEP.
            #
            # Therefore:
            #
            # total_arrived += current_step_arrivals
            #
            # =================================================

            try:

                arrived_this_step = int(
                    traci.simulation.getArrivedNumber()
                )

            except Exception:

                arrived_this_step = 0

            if arrived_this_step < 0:

                arrived_this_step = 0

            total_arrived += (
                arrived_this_step
            )

            # =================================================
            # STEP COUNTER
            # =================================================

            step += 1

        # =====================================================
        # FINAL VALUES
        # =====================================================

        result["reward"] = (
            total_reward
        )

        result["steps"] = (
            step
        )

        result["terminated"] = bool(
            terminated
        )

        result["truncated"] = bool(
            truncated
        )

        # -----------------------------------------------------
        # Average network metrics
        # -----------------------------------------------------

        denominator = max(
            1,
            step
        )

        result["waiting_time"] = (
            total_waiting
            / denominator
        )

        result["queue_length"] = (
            total_queue
            / denominator
        )

        result["average_speed"] = (
            total_speed
            / denominator
        )

        # -----------------------------------------------------
        # Correct episode throughput
        # -----------------------------------------------------

        result["throughput"] = int(
            total_arrived
        )

    finally:

        result["evaluation_time"] = (
            time.perf_counter()
            - start_time
        )

        env.close()

    return result

# =========================================================
# RUN ALL BASELINE EVALUATIONS
# =========================================================

def run_all_evaluations() -> List[Dict[str, Any]]:
    """
    Run every baseline policy across every scenario.

    Returns
    -------
    list
        Detailed episode-level evaluation results.
    """

    all_results = []

    total_runs = (
        len(SCENARIOS)
        * len(POLICIES)
        * EPISODES_PER_SCENARIO
    )

    current_run = 0

    print()
    print("=" * 90)
    print("BASELINE POLICY EVALUATION")
    print("=" * 90)
    print()
    print("Policies:")
    
    for policy in POLICIES:
        print(f"  - {policy}")

    print()
    print("Scenarios:")

    for scenario in SCENARIOS:
        print(f"  - {scenario}")

    print()
    print(
        f"Episodes per scenario/policy: "
        f"{EPISODES_PER_SCENARIO}"
    )

    print(
        f"Total evaluations: {total_runs}"
    )

    print()
    print("=" * 90)

    overall_start = time.perf_counter()

    # =====================================================
    # POLICY LOOP
    # =====================================================

    for policy_name in POLICIES:

        print()
        print(
            f"### POLICY: {policy_name}"
        )
        print("-" * 90)

        # =================================================
        # SCENARIO LOOP
        # =================================================

        for scenario in SCENARIOS:

            episode_results = []

            # =============================================
            # EPISODE LOOP
            # =============================================

            for episode in range(
                1,
                EPISODES_PER_SCENARIO + 1
            ):

                current_run += 1

                print(
                    f"[{current_run}/{total_runs}] "
                    f"{policy_name:12s} | "
                    f"{scenario:20s} | "
                    f"Episode {episode}"
                )

                try:

                    result = run_episode(
                        scenario=scenario,
                        policy_name=policy_name,
                        episode=episode,
                    )

                    all_results.append(
                        result
                    )

                    episode_results.append(
                        result
                    )

                except Exception as error:

                    print(
                        f"    [ERROR] "
                        f"{type(error).__name__}: "
                        f"{error}"
                    )

                    # -------------------------------------
                    # Store failed evaluation
                    # -------------------------------------

                    failed_result = (
                        empty_result(
                            scenario,
                            policy_name,
                            episode,
                        )
                    )

                    failed_result[
                        "error"
                    ] = str(error)

                    failed_result[
                        "error_type"
                    ] = type(error).__name__

                    all_results.append(
                        failed_result
                    )

            # =============================================
            # SCENARIO SUMMARY
            # =============================================

            successful = [
                r
                for r in episode_results
                if "error" not in r
            ]

            if successful:

                avg_reward = float(
                    np.mean([
                        r["reward"]
                        for r in successful
                    ])
                )

                avg_waiting = float(
                    np.mean([
                        r["waiting_time"]
                        for r in successful
                    ])
                )

                avg_queue = float(
                    np.mean([
                        r["queue_length"]
                        for r in successful
                    ])
                )

                avg_speed = float(
                    np.mean([
                        r["average_speed"]
                        for r in successful
                    ])
                )

                avg_throughput = float(
                    np.mean([
                        r["throughput"]
                        for r in successful
                    ])
                )

                print(
                    f"    Result -> "
                    f"Reward: {avg_reward:.2f}, "
                    f"Waiting: {avg_waiting:.2f}, "
                    f"Queue: {avg_queue:.2f}, "
                    f"Speed: {avg_speed:.2f}, "
                    f"Throughput: {avg_throughput:.2f}"
                )

    total_time = (
        time.perf_counter()
        - overall_start
    )

    print()
    print("=" * 90)
    print(
        "BASELINE EVALUATION COMPLETE"
    )
    print("=" * 90)

    print(
        f"Total evaluation time: "
        f"{total_time:.2f} seconds"
    )

    print()

    return all_results


# =========================================================
# SAVE DETAILED CSV
# =========================================================

def save_detailed_csv(
    results: List[Dict[str, Any]]
) -> None:
    """
    Save episode-level results.
    """

    if not results:

        return

    # -----------------------------------------------------
    # Collect all possible fields
    # -----------------------------------------------------

    field_names = [

        "scenario",

        "policy",

        "episode",

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

    ]

    with open(
        RESULTS_CSV,
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

        for result in results:

            row = {}

            for field in field_names:

                row[field] = result.get(
                    field,
                    ""
                )

            writer.writerow(row)

    print(
        f"Detailed CSV saved to:\n"
        f"{RESULTS_CSV}"
    )


# =========================================================
# CREATE SUMMARY
# =========================================================

def create_summary(
    results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Aggregate episode results by:

        Policy + Scenario
    """

    summary = []

    for policy in POLICIES:

        for scenario in SCENARIOS:

            matching = [

                result

                for result in results

                if (
                    result.get("policy")
                    == policy
                    and
                    result.get("scenario")
                    == scenario
                )

                and
                "error" not in result

            ]

            if not matching:

                summary.append({

                    "policy":
                        policy,

                    "scenario":
                        scenario,

                    "episodes":
                        0,

                    "reward":
                        0.0,

                    "waiting_time":
                        0.0,

                    "queue_length":
                        0.0,

                    "average_speed":
                        0.0,

                    "throughput":
                        0.0,

                    "steps":
                        0.0,

                    "evaluation_time":
                        0.0,

                })

                continue

            summary.append({

                "policy":
                    policy,

                "scenario":
                    scenario,

                "episodes":
                    len(matching),

                "reward":
                    float(
                        np.mean([
                            r["reward"]
                            for r in matching
                        ])
                    ),

                "waiting_time":
                    float(
                        np.mean([
                            r["waiting_time"]
                            for r in matching
                        ])
                    ),

                "queue_length":
                    float(
                        np.mean([
                            r["queue_length"]
                            for r in matching
                        ])
                    ),

                "average_speed":
                    float(
                        np.mean([
                            r["average_speed"]
                            for r in matching
                        ])
                    ),

                "throughput":
                    float(
                        np.mean([
                            r["throughput"]
                            for r in matching
                        ])
                    ),

                "steps":
                    float(
                        np.mean([
                            r["steps"]
                            for r in matching
                        ])
                    ),

                "evaluation_time":
                    float(
                        np.sum([
                            r["evaluation_time"]
                            for r in matching
                        ])
                    ),

            })

    return summary


# =========================================================
# SAVE SUMMARY CSV
# =========================================================

def save_summary_csv(
    summary: List[Dict[str, Any]]
) -> None:
    """
    Save policy/scenario summary.
    """

    if not summary:

        return

    field_names = [

        "policy",

        "scenario",

        "episodes",

        "reward",

        "waiting_time",

        "queue_length",

        "average_speed",

        "throughput",

        "steps",

        "evaluation_time",

    ]

    with open(
        SUMMARY_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=field_names,
        )

        writer.writeheader()

        writer.writerows(
            summary
        )

    print(
        f"Summary CSV saved to:\n"
        f"{SUMMARY_CSV}"
    )


# =========================================================
# SAVE JSON
# =========================================================

def save_json(
    results: List[Dict[str, Any]],
    summary: List[Dict[str, Any]],
) -> None:
    """
    Save complete evaluation results.
    """

    payload = {

        "project":
            "Traffic MARL",

        "evaluation":
            "Baseline Policy Evaluation",

        "policies":
            POLICIES,

        "scenarios":
            SCENARIOS,

        "episodes_per_scenario":
            EPISODES_PER_SCENARIO,

        "random_seed":
            RANDOM_SEED,

        "max_episode_steps":
            MAX_STEPS,

        "detailed_results":
            results,

        "summary":
            summary,

    }

    with open(
        RESULTS_JSON,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            indent=4,
        )

    print(
        f"JSON saved to:\n"
        f"{RESULTS_JSON}"
    )


# =========================================================
# PRINT SUMMARY TABLE
# =========================================================

def print_summary_table(
    summary: List[Dict[str, Any]]
) -> None:
    """
    Print a readable summary.
    """

    print()
    print("=" * 125)
    print("BASELINE EVALUATION SUMMARY")
    print("=" * 125)

    header = (

        f"{'Policy':<15}"

        f"{'Scenario':<22}"

        f"{'Reward':>14}"

        f"{'Waiting':>12}"

        f"{'Queue':>12}"

        f"{'Speed':>12}"

        f"{'Throughput':>14}"

    )

    print(header)

    print("-" * 125)

    for row in summary:

        print(

            f"{row['policy']:<15}"

            f"{row['scenario']:<22}"

            f"{row['reward']:>14.2f}"

            f"{row['waiting_time']:>12.2f}"

            f"{row['queue_length']:>12.2f}"

            f"{row['average_speed']:>12.2f}"

            f"{row['throughput']:>14.2f}"

        )

    print("=" * 125)


# =========================================================
# FIND BEST POLICY
# =========================================================

def calculate_best_policies(
    summary: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Determine best baseline policies.

    Lower is better for:

        Waiting time
        Queue length

    Higher is better for:

        Speed
        Throughput

    Reward:

        Higher is better
    """

    valid = [

        row

        for row in summary

        if row["episodes"] > 0

    ]

    if not valid:

        return {}

    # -----------------------------------------------------
    # Overall policy averages
    # -----------------------------------------------------

    policy_scores = {}

    for policy in POLICIES:

        rows = [

            row

            for row in valid

            if row["policy"] == policy

        ]

        if not rows:

            continue

        policy_scores[policy] = {

            "reward":
                float(
                    np.mean([
                        r["reward"]
                        for r in rows
                    ])
                ),

            "waiting_time":
                float(
                    np.mean([
                        r["waiting_time"]
                        for r in rows
                    ])
                ),

            "queue_length":
                float(
                    np.mean([
                        r["queue_length"]
                        for r in rows
                    ])
                ),

            "average_speed":
                float(
                    np.mean([
                        r["average_speed"]
                        for r in rows
                    ])
                ),

            "throughput":
                float(
                    np.mean([
                        r["throughput"]
                        for r in rows
                    ])
                ),

        }

    if not policy_scores:

        return {}

    best_reward = max(
        policy_scores,
        key=lambda p:
            policy_scores[p]["reward"]
    )

    best_waiting = min(
        policy_scores,
        key=lambda p:
            policy_scores[p]["waiting_time"]
    )

    best_queue = min(
        policy_scores,
        key=lambda p:
            policy_scores[p]["queue_length"]
    )

    best_speed = max(
        policy_scores,
        key=lambda p:
            policy_scores[p]["average_speed"]
    )

    best_throughput = max(
        policy_scores,
        key=lambda p:
            policy_scores[p]["throughput"]
    )

    return {

        "policy_scores":
            policy_scores,

        "best_reward":
            best_reward,

        "best_waiting_time":
            best_waiting,

        "best_queue":
            best_queue,

        "best_speed":
            best_speed,

        "best_throughput":
            best_throughput,

    }


# =========================================================
# PRINT BEST POLICY RESULTS
# =========================================================

def print_best_policies(
    comparison: Dict[str, Any]
) -> None:
    """
    Print the best baseline according to
    each evaluation metric.
    """

    if not comparison:

        print(
            "\nNo valid baseline comparison available."
        )

        return

    print()
    print("=" * 90)
    print("BEST BASELINE POLICIES")
    print("=" * 90)

    print(
        "Best Reward:        ",
        comparison[
            "best_reward"
        ]
    )

    print(
        "Best Waiting Time:  ",
        comparison[
            "best_waiting_time"
        ]
    )

    print(
        "Best Queue:         ",
        comparison[
            "best_queue"
        ]
    )

    print(
        "Best Speed:         ",
        comparison[
            "best_speed"
        ]
    )

    print(
        "Best Throughput:    ",
        comparison[
            "best_throughput"
        ]
    )

    print()
    print(
        "Overall Policy Averages:"
    )

    for policy, scores in (
        comparison[
            "policy_scores"
        ].items()
    ):

        print()

        print(
            f"{policy}:"
        )

        print(
            f"  Reward:     "
            f"{scores['reward']:.2f}"
        )

        print(
            f"  Waiting:    "
            f"{scores['waiting_time']:.2f}"
        )

        print(
            f"  Queue:      "
            f"{scores['queue_length']:.2f}"
        )

        print(
            f"  Speed:      "
            f"{scores['average_speed']:.2f}"
        )

        print(
            f"  Throughput: "
            f"{scores['throughput']:.2f}"
        )

    print()
    print("=" * 90)


# =========================================================
# FAST BASELINE VERIFICATION
# =========================================================

def run_fast_verification():
    """
    Small verification run before the full experiment.

    Purpose:
        Verify that the corrected throughput metric
        is working before spending ~20 minutes on the
        complete 90-episode evaluation.

    Uses:

        3 scenarios
        3 policies
        1 episode

    Total:

        9 episodes
    """

    verification_scenarios = [

        "baseline",

        "morning_rush",

        "stress_test",

    ]

    verification_policies = [

        "fixed_time",

        "random",

        "queue_based",

    ]

    print()
    print("=" * 100)
    print("FAST BASELINE VERIFICATION")
    print("=" * 100)

    print()
    print(
        "Purpose:"
    )

    print(
        "Verify corrected throughput before "
        "running the complete evaluation."
    )

    print()

    results = []

    for policy in verification_policies:

        print()
        print(
            f"Policy: {policy}"
        )

        print(
            "-" * 80
        )

        for scenario in verification_scenarios:

            print(
                f"Running: "
                f"{policy:12s} | "
                f"{scenario}"
            )

            try:

                result = run_episode(
                    scenario=scenario,
                    policy_name=policy,
                    episode=1,
                )

                results.append(
                    result
                )

                print(
                    f"  Reward: "
                    f"{result['reward']:.2f}"
                )

                print(
                    f"  Waiting: "
                    f"{result['waiting_time']:.2f}"
                )

                print(
                    f"  Queue: "
                    f"{result['queue_length']:.2f}"
                )

                print(
                    f"  Speed: "
                    f"{result['average_speed']:.2f}"
                )

                print(
                    f"  Throughput: "
                    f"{result['throughput']}"
                )

                print(
                    f"  Steps: "
                    f"{result['steps']}"
                )

            except Exception as error:

                print(
                    f"  [ERROR] "
                    f"{type(error).__name__}: "
                    f"{error}"
                )

    # =====================================================
    # VERIFICATION TABLE
    # =====================================================

    print()
    print("=" * 100)
    print("VERIFICATION RESULTS")
    print("=" * 100)

    print()

    print(
        f"{'Policy':<15}"
        f"{'Scenario':<20}"
        f"{'Reward':>15}"
        f"{'Waiting':>12}"
        f"{'Queue':>12}"
        f"{'Speed':>12}"
        f"{'Throughput':>14}"
    )

    print(
        "-" * 100
    )

    for result in results:

        print(

            f"{result['policy']:<15}"

            f"{result['scenario']:<20}"

            f"{result['reward']:>15.2f}"

            f"{result['waiting_time']:>12.2f}"

            f"{result['queue_length']:>12.2f}"

            f"{result['average_speed']:>12.2f}"

            f"{result['throughput']:>14}"

        )

    print()
    print("=" * 100)

    # =====================================================
    # BASIC VALIDATION
    # =====================================================

    valid_results = [

        result

        for result in results

        if "error" not in result

    ]

    if len(valid_results) == 0:

        print(
            "[FAIL] No baseline verification "
            "episodes completed."
        )

        return False

    # -----------------------------------------------------
    # Throughput check
    # -----------------------------------------------------

    throughput_values = [

        result["throughput"]

        for result in valid_results

    ]

    positive_throughput = [

        value

        for value in throughput_values

        if value > 0

    ]

    print()

    print(
        f"Completed episodes: "
        f"{len(valid_results)}"
    )

    print(
        f"Positive-throughput episodes: "
        f"{len(positive_throughput)}"
    )

    if positive_throughput:

        print(
            "[PASS] Throughput is being "
            "accumulated across simulation steps."
        )

    else:

        print(
            "[WARNING] No positive throughput "
            "was observed."
        )

        print(
            "Check SUMO demand and route files."
        )

        return False

    # -----------------------------------------------------
    # Step check
    # -----------------------------------------------------

    step_values = [

        result["steps"]

        for result in valid_results

    ]

    max_steps = max(
        step_values
    )

    print(
        f"Maximum episode steps: "
        f"{max_steps}"
    )

    if max_steps > 0:

        print(
            "[PASS] Simulation advanced."
        )

    else:

        print(
            "[FAIL] Simulation did not advance."
        )

        return False

    print()
    print(
        "[PASS] Fast baseline verification complete."
    )

    print(
        "Full 90-episode evaluation is now safe to run."
    )

    print()

    return True
    
# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("=" * 90)
    print("Traffic MARL - Baseline Evaluation")
    print("=" * 90)
    print()

    print(
        "Project Root:"
    )

    print(
        PROJECT_ROOT
    )

    print()

    print(
        "Evaluation Directory:"
    )

    print(
        EVALUATION_DIR
    )

    print()

    print(
        "Policies:"
    )

    for policy in POLICIES:

        print(
            f"  - {policy}"
        )

    print()

    print(
        "Scenarios:"
    )

    for scenario in SCENARIOS:

        print(
            f"  - {scenario}"
        )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "This evaluation uses classical "
        "baseline policies only."
    )

    print(
        "PPO/MARL model is NOT used."
    )

    print()

    # =====================================================
    # RUN
    # =====================================================

    results = run_all_evaluations()

    # =====================================================
    # SAVE DETAILED RESULTS
    # =====================================================

    save_detailed_csv(
        results
    )

    # =====================================================
    # CREATE SUMMARY
    # =====================================================

    summary = create_summary(
        results
    )

    # =====================================================
    # SAVE SUMMARY
    # =====================================================

    save_summary_csv(
        summary
    )

    # =====================================================
    # SAVE JSON
    # =====================================================

    save_json(
        results,
        summary
    )

    # =====================================================
    # PRINT TABLE
    # =====================================================

    print_summary_table(
        summary
    )

    # =====================================================
    # COMPARE POLICIES
    # =====================================================

    comparison = (
        calculate_best_policies(
            summary
        )
    )

    print_best_policies(
        comparison
    )

    # =====================================================
    # FINAL OUTPUT
    # =====================================================

    print()
    print("=" * 90)
    print("FILES CREATED")
    print("=" * 90)

    print()
    print(
        "Detailed Results:"
    )

    print(
        RESULTS_CSV
    )

    print()
    print(
        "Summary Results:"
    )

    print(
        SUMMARY_CSV
    )

    print()
    print(
        "JSON Results:"
    )

    print(
        RESULTS_JSON
    )

    print()
    print("=" * 90)
    print(
        "BASELINE EVALUATION FINISHED"
    )
    print("=" * 90)
    print()


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()

