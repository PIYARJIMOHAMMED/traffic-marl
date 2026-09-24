"""
======================================================================
MAI-07 | PPO-MARL 500K FINAL EVALUATION
Multi-Agent Reinforcement Learning for Network-Wide Adaptive Traffic
Control

PROPOSED SYSTEM ONLY

This file evaluates ONLY:
    PPO-MARL 500K

It does NOT evaluate:
    DQN
    PPO-Local
    Random
    Fixed-Time
    Queue-Based

Evaluation:
    - 10 SUMO traffic scenarios
    - 3 episodes per scenario
    - seeds 42, 43, 44
    - maximum episode steps from EnvironmentConfig
    - deterministic PPO inference
    - CPU inference
    - live SUMO/TraCI metrics
    - no hardcoded performance results

Model:
    Observation = 120-D
    Action      = MultiDiscrete([2,2,2,2])

Output:
    rl/results/ppo_marl_500k_evaluation/
        ppo_marl_500k_raw.csv
        ppo_marl_500k_summary.csv
        ppo_marl_500k_scenario_summary.csv
        ppo_marl_500k_report.json
======================================================================
"""

from pathlib import Path
import sys
import json
import time
import random

import numpy as np
import pandas as pd
import traci

from stable_baselines3 import PPO


# ======================================================================
# PROJECT ROOT
# ======================================================================

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ======================================================================
# PROJECT IMPORTS
# ======================================================================

from rl.environments.traffic_env import TrafficEnvironment
from rl.environments.config import EnvironmentConfig


# ======================================================================
# CONFIGURATION
# ======================================================================

MODEL_PATH = (
    ROOT
    / "rl"
    / "checkpoints"
    / "ppo_marl_500k"
    / "ppo_marl_500k_final.zip"
)

OUTPUT_DIR = (
    ROOT
    / "rl"
    / "results"
    / "ppo_marl_500k_evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ======================================================================
# EVALUATION PROTOCOL
# ======================================================================

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

SEEDS = [42, 43, 44]

EPISODES_PER_SCENARIO = 3

MAX_STEPS = int(
    EnvironmentConfig.MAX_EPISODE_STEPS
)


# ======================================================================
# EXPECTED PPO-MARL ARCHITECTURE
# ======================================================================

EXPECTED_OBSERVATION_DIM = 120

EXPECTED_ACTION_NVECS = np.array(
    [2, 2, 2, 2],
    dtype=np.int64,
)


# ======================================================================
# UTILITY
# ======================================================================

def safe_float(value, default=0.0):

    try:

        value = float(value)

        if np.isfinite(value):
            return value

        return default

    except Exception:

        return default


def close_sumo():

    try:

        if traci.isLoaded():
            traci.close()

    except Exception:
        pass


# ======================================================================
# ACTION VALIDATION
# ======================================================================

def validate_action(action):

    action = np.asarray(
        action,
        dtype=np.int64,
    ).flatten()


    if action.shape != (4,):

        raise ValueError(
            "Invalid PPO-MARL action shape: "
            f"{action.shape}. Expected (4,)"
        )


    if np.any(action < 0) or np.any(action > 1):

        raise ValueError(
            "Invalid PPO-MARL action values: "
            f"{action}"
        )


    return action


# ======================================================================
# LIVE SUMO METRICS
# ======================================================================

def get_live_metrics():

    try:

        vehicle_ids = list(
            traci.vehicle.getIDList()
        )

    except Exception:

        return 0.0, 0.0, 0.0


    if not vehicle_ids:

        return 0.0, 0.0, 0.0


    speeds = []

    waiting_times = []

    queue_length = 0


    for vehicle_id in vehicle_ids:

        try:

            speed = safe_float(
                traci.vehicle.getSpeed(
                    vehicle_id
                )
            )

            waiting = safe_float(
                traci.vehicle.getAccumulatedWaitingTime(
                    vehicle_id
                )
            )


            speeds.append(
                speed
            )

            waiting_times.append(
                waiting
            )


            if speed <= 0.1:

                queue_length += 1


        except Exception:

            continue


    average_speed = (

        float(
            np.mean(speeds)
        )

        if speeds

        else 0.0
    )


    waiting_time = (

        float(
            np.sum(waiting_times)
        )

        if waiting_times

        else 0.0
    )


    return (
        average_speed,
        waiting_time,
        float(queue_length),
    )


# ======================================================================
# THROUGHPUT
# ======================================================================

def get_step_arrivals():

    """
    Return vehicles that arrived during the CURRENT
    SUMO simulation step.

    The episode-level throughput is accumulated by
    summing this value at every step.
    """

    try:

        return max(
            0,
            int(
                traci.simulation.getArrivedNumber()
            ),
        )

    except Exception:

        return 0


# ======================================================================
# EPISODE EVALUATION
# ======================================================================

def evaluate_episode(
    model,
    scenario,
    episode_number,
    seed,
):

    start_time = time.perf_counter()


    env = None


    result = {

        "policy":
            "ppo_marl_500k",

        "model":
            "ppo_marl_500k_final",

        "scenario":
            scenario,

        "episode":
            episode_number,

        "seed":
            seed,

        "observation_type":
            "Global-120D",

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

        "terminated":
            False,

        "truncated":
            False,

        "error_type":
            "",

        "error":
            "",

        "runtime_seconds":
            0.0,
    }


    reward_sum = 0.0

    waiting_sum = 0.0

    queue_sum = 0.0

    speed_sum = 0.0

    throughput_sum = 0


    try:

        # --------------------------------------------------------------
        # CREATE ENVIRONMENT
        # --------------------------------------------------------------

        env = TrafficEnvironment(
            scenario_name=scenario,
            render_mode=None,
        )


        # --------------------------------------------------------------
        # RESET
        # --------------------------------------------------------------

        observation, info = env.reset(
            seed=seed
        )


        observation = np.asarray(
            observation,
            dtype=np.float32,
        ).flatten()


        if observation.shape != (
            EXPECTED_OBSERVATION_DIM,
        ):

            raise ValueError(
                "Invalid reset observation shape: "
                f"{observation.shape}. "
                "Expected (120,)"
            )


        if not np.all(
            np.isfinite(observation)
        ):

            raise ValueError(
                "Reset observation contains "
                "NaN or infinite values."
            )


        # --------------------------------------------------------------
        # SIMULATION
        # --------------------------------------------------------------

        for step in range(
            MAX_STEPS
        ):


            # ----------------------------------------------------------
            # PPO-MARL ACTION
            # ----------------------------------------------------------

            action, _ = model.predict(
                observation,
                deterministic=True,
            )


            action = validate_action(
                action
            )


            # ----------------------------------------------------------
            # ENVIRONMENT STEP
            # ----------------------------------------------------------

            (
                next_observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )


            next_observation = np.asarray(
                next_observation,
                dtype=np.float32,
            ).flatten()


            if next_observation.shape != (
                EXPECTED_OBSERVATION_DIM,
            ):

                raise ValueError(
                    "Invalid step observation shape: "
                    f"{next_observation.shape}. "
                    "Expected (120,)"
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
                safe_float(reward)
            ):

                raise ValueError(
                    "Reward is not finite."
                )


            # ----------------------------------------------------------
            # REWARD
            # ----------------------------------------------------------

            reward_sum += safe_float(
                reward
            )


            # ----------------------------------------------------------
            # LIVE SUMO METRICS
            # ----------------------------------------------------------

            (
                average_speed,
                waiting_time,
                queue_length,
            ) = get_live_metrics()


            speed_sum += (
                average_speed
            )

            waiting_sum += (
                waiting_time
            )

            queue_sum += (
                queue_length
            )


            # ----------------------------------------------------------
            # THROUGHPUT
            # ----------------------------------------------------------

            throughput_sum += (
                get_step_arrivals()
            )


            # ----------------------------------------------------------
            # UPDATE
            # ----------------------------------------------------------

            observation = (
                next_observation
            )


            result["steps"] = (
                step + 1
            )


            result["terminated"] = bool(
                terminated
            )

            result["truncated"] = bool(
                truncated
            )


            # ----------------------------------------------------------
            # END
            # ----------------------------------------------------------

            if terminated or truncated:

                break


        # --------------------------------------------------------------
        # FINAL EPISODE METRICS
        # --------------------------------------------------------------

        steps = max(
            1,
            result["steps"],
        )


        result["reward"] = float(
            reward_sum
        )


        result["waiting_time"] = float(
            waiting_sum / steps
        )


        result["queue_length"] = float(
            queue_sum / steps
        )


        result["average_speed"] = float(
            speed_sum / steps
        )


        result["throughput"] = int(
            throughput_sum
        )


    except Exception as exc:

        result["error_type"] = (
            type(exc).__name__
        )

        result["error"] = str(
            exc
        )


    finally:

        try:

            if env is not None:

                env.close()

        except Exception:

            pass


        close_sumo()


    result["runtime_seconds"] = (
        time.perf_counter()
        - start_time
    )


    return result


# ======================================================================
# MODEL VALIDATION
# ======================================================================

def load_and_validate_model():

    print(
        "[CHECK] Loading PPO-MARL 500K..."
    )


    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "PPO-MARL checkpoint not found:\n"
            f"{MODEL_PATH}"
        )


    model = PPO.load(
        str(MODEL_PATH),
        device="cpu",
    )


    # --------------------------------------------------------------
    # OBSERVATION
    # --------------------------------------------------------------

    observation_shape = tuple(
        model.observation_space.shape
        or ()
    )


    print(
        "[CHECK] Observation space:",
        model.observation_space,
    )


    if observation_shape != (
        EXPECTED_OBSERVATION_DIM,
    ):

        raise ValueError(
            "Wrong PPO-MARL observation space.\n"
            f"Expected: (120,)\n"
            f"Found: {model.observation_space}"
        )


    # --------------------------------------------------------------
    # ACTION
    # --------------------------------------------------------------

    print(
        "[CHECK] Action space:",
        model.action_space,
    )


    if not hasattr(
        model.action_space,
        "nvec",
    ):

        raise ValueError(
            "PPO-MARL does not use "
            "MultiDiscrete action space."
        )


    actual_nvec = np.asarray(
        model.action_space.nvec,
        dtype=np.int64,
    )


    if not np.array_equal(
        actual_nvec,
        EXPECTED_ACTION_NVECS,
    ):

        raise ValueError(
            "Wrong PPO-MARL action space.\n"
            "Expected: MultiDiscrete([2,2,2,2])\n"
            f"Found: {model.action_space}"
        )


    print(
        "[PASS] PPO-MARL architecture validated."
    )

    print(
        "       Observation: 120-D"
    )

    print(
        "       Action: MultiDiscrete([2,2,2,2])"
    )


    return model


# ======================================================================
# ENVIRONMENT VALIDATION
# ======================================================================

def validate_environment():

    print()
    print(
        "[CHECK] Validating all scenarios..."
    )


    for scenario in SCENARIOS:

        env = None

        try:

            env = TrafficEnvironment(
                scenario_name=scenario,
                render_mode=None,
            )


            observation, _ = env.reset(
                seed=42
            )


            observation = np.asarray(
                observation,
                dtype=np.float32,
            ).flatten()


            if observation.shape != (
                EXPECTED_OBSERVATION_DIM,
            ):

                raise ValueError(
                    f"{scenario}: invalid "
                    f"observation shape "
                    f"{observation.shape}"
                )


            if not np.all(
                np.isfinite(observation)
            ):

                raise ValueError(
                    f"{scenario}: "
                    "observation contains NaN/Inf"
                )


            print(
                f"[PASS] {scenario:<22} "
                "120-D observation"
            )


        finally:

            try:

                if env is not None:

                    env.close()

            except Exception:

                pass


            close_sumo()


    print(
        "[PASS] All scenarios validated."
    )


# ======================================================================
# RESULT VALIDATION
# ======================================================================

def validate_results(results):

    expected_count = (
        len(SCENARIOS)
        * EPISODES_PER_SCENARIO
    )


    if len(results) != expected_count:

        raise RuntimeError(
            "Unexpected evaluation count.\n"
            f"Expected: {expected_count}\n"
            f"Found: {len(results)}"
        )


    errors = [
        result
        for result in results
        if result["error"]
    ]


    if errors:

        print()
        print(
            "[ERROR] Failed episodes:"
        )


        for result in errors:

            print(
                f"{result['scenario']} | "
                f"episode={result['episode']} | "
                f"seed={result['seed']} | "
                f"{result['error']}"
            )


        raise RuntimeError(
            f"{len(errors)} evaluation "
            "episodes failed."
        )


    for result in results:

        for metric in [
            "reward",
            "waiting_time",
            "queue_length",
            "average_speed",
            "throughput",
            "steps",
        ]:

            value = safe_float(
                result[metric]
            )


            if not np.isfinite(value):

                raise RuntimeError(
                    f"Non-finite metric: "
                    f"{metric}={value}"
                )


    print(
        f"[PASS] {len(results)} evaluation "
        "episodes completed."
    )


# ======================================================================
# OUTPUTS
# ======================================================================

def create_outputs(results):

    df = pd.DataFrame(
        results
    )


    # --------------------------------------------------------------
    # RAW
    # --------------------------------------------------------------

    raw_path = (
        OUTPUT_DIR
        / "ppo_marl_500k_raw.csv"
    )


    df.to_csv(
        raw_path,
        index=False,
    )


    # --------------------------------------------------------------
    # SCENARIO SUMMARY
    # --------------------------------------------------------------

    scenario_summary = (
        df
        .groupby(
            [
                "policy",
                "scenario",
            ],
            as_index=False,
        )
        .agg(
            episodes=(
                "episode",
                "count",
            ),

            reward_mean=(
                "reward",
                "mean",
            ),

            reward_std=(
                "reward",
                "std",
            ),

            waiting_time_mean=(
                "waiting_time",
                "mean",
            ),

            queue_length_mean=(
                "queue_length",
                "mean",
            ),

            average_speed_mean=(
                "average_speed",
                "mean",
            ),

            throughput_mean=(
                "throughput",
                "mean",
            ),

            steps_mean=(
                "steps",
                "mean",
            ),
        )
    )


    scenario_summary[
        "reward_std"
    ] = (
        scenario_summary[
            "reward_std"
        ]
        .fillna(0.0)
    )


    scenario_path = (
        OUTPUT_DIR
        / "ppo_marl_500k_scenario_summary.csv"
    )


    scenario_summary.to_csv(
        scenario_path,
        index=False,
    )


    # --------------------------------------------------------------
    # OVERALL
    # --------------------------------------------------------------

    overall = pd.DataFrame(
        [
            {

                "policy":
                    "ppo_marl_500k",

                "episodes":
                    int(len(df)),

                "reward_mean":
                    float(
                        df["reward"].mean()
                    ),

                "reward_std":
                    float(
                        df["reward"].std(
                            ddof=1
                        )
                    )
                    if len(df) > 1
                    else 0.0,

                "waiting_time_mean":
                    float(
                        df[
                            "waiting_time"
                        ].mean()
                    ),

                "queue_length_mean":
                    float(
                        df[
                            "queue_length"
                        ].mean()
                    ),

                "average_speed_mean":
                    float(
                        df[
                            "average_speed"
                        ].mean()
                    ),

                "throughput_mean":
                    float(
                        df[
                            "throughput"
                        ].mean()
                    ),

                "steps_mean":
                    float(
                        df[
                            "steps"
                        ].mean()
                    ),
            }
        ]
    )


    overall_path = (
        OUTPUT_DIR
        / "ppo_marl_500k_summary.csv"
    )


    overall.to_csv(
        overall_path,
        index=False,
    )


    # --------------------------------------------------------------
    # JSON
    # --------------------------------------------------------------

    report = {

        "project":
            "MAI-07 | Multi-Agent Reinforcement "
            "Learning for Network-Wide Adaptive "
            "Traffic Control",

        "evaluation":
            "PPO-MARL 500K Final Evaluation",

        "controller":
            "PPO-MARL",

        "role":
            "Proposed system",

        "performance_values_hardcoded":
            False,

        "data_source":
            "Live SUMO/TraCI",

        "algorithm":
            "PPO-MARL",

        "training_timesteps":
            500000,

        "environment_observation_dimension":
            120,

        "model_observation_dimension":
            120,

        "observation_type":
            "Global-120D graph-aware",

        "action_space":
            "MultiDiscrete([2,2,2,2])",

        "number_of_agents":
            4,

        "scenarios":
            SCENARIOS,

        "episodes_per_scenario":
            EPISODES_PER_SCENARIO,

        "total_episodes":
            len(results),

        "max_episode_steps":
            MAX_STEPS,

        "seeds":
            SEEDS,

        "deterministic_inference":
            True,

        "device":
            "cpu",

        "metrics": {

            "reward":
                "Cumulative environment reward",

            "waiting_time":
                "Mean per-step sum of vehicle "
                "accumulated waiting time from SUMO",

            "queue_length":
                "Mean per-step number of vehicles "
                "with speed <= 0.1 m/s",

            "average_speed":
                "Mean vehicle speed from SUMO",

            "throughput":
                "Total vehicles arriving during "
                "the episode",

            "steps":
                "Number of simulation steps",
        },

        "overall":
            overall.to_dict(
                orient="records"
            )[0],

        "scenario_summary":
            scenario_summary.to_dict(
                orient="records"
            ),

        "episode_results":
            results,

        "evaluation_runtime_seconds":
            float(
                sum(
                    safe_float(
                        x[
                            "runtime_seconds"
                        ]
                    )
                    for x in results
                )
            ),

        "timestamp":
            pd.Timestamp.now(
                tz="UTC"
            ).isoformat(),
    }


    report_path = (
        OUTPUT_DIR
        / "ppo_marl_500k_report.json"
    )


    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )


    print()
    print("=" * 80)
    print("PPO-MARL FINAL EVALUATION COMPLETE")
    print("=" * 80)

    print(
        f"Raw CSV      : {raw_path}"
    )

    print(
        f"Summary CSV  : {overall_path}"
    )

    print(
        f"Scenario CSV : {scenario_path}"
    )

    print(
        f"JSON report  : {report_path}"
    )

    print("=" * 80)


# ======================================================================
# MAIN
# ======================================================================

def main():

    print()
    print("=" * 80)
    print("MAI-07 | PPO-MARL 500K FINAL EVALUATION")
    print("=" * 80)

    print(
        "Controller     : PPO-MARL ONLY"
    )

    print(
        "Role           : Proposed system"
    )

    print(
        "Observation    : 120-D"
    )

    print(
        "Action         : MultiDiscrete([2,2,2,2])"
    )

    print(
        f"Scenarios      : {len(SCENARIOS)}"
    )

    print(
        f"Episodes/scenario: "
        f"{EPISODES_PER_SCENARIO}"
    )

    print(
        f"Seeds          : {SEEDS}"
    )

    print(
        f"Max steps      : {MAX_STEPS}"
    )

    print(
        "Performance     : LIVE SUMO/TraCI"
    )

    print(
        "Hardcoded       : NO"
    )

    print("=" * 80)


    # --------------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------------

    random.seed(42)

    np.random.seed(42)


    # --------------------------------------------------------------
    # Headless SUMO
    # --------------------------------------------------------------

    EnvironmentConfig.SUMO_BINARY = "sumo"


    # --------------------------------------------------------------
    # Validate model
    # --------------------------------------------------------------

    model = (
        load_and_validate_model()
    )


    # --------------------------------------------------------------
    # Validate environment
    # --------------------------------------------------------------

    validate_environment()


    # --------------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------------

    results = []


    total = (
        len(SCENARIOS)
        * EPISODES_PER_SCENARIO
    )


    completed = 0


    print()
    print("=" * 80)
    print("STARTING PPO-MARL EVALUATION")
    print("=" * 80)


    for scenario in SCENARIOS:

        print()
        print(
            f"----- {scenario.upper()} -----"
        )


        for episode, seed in zip(
            range(
                1,
                EPISODES_PER_SCENARIO + 1,
            ),
            SEEDS,
        ):

            completed += 1


            result = evaluate_episode(
                model=model,
                scenario=scenario,
                episode_number=episode,
                seed=seed,
            )


            results.append(
                result
            )


            if result["error"]:

                print(
                    f"[{completed}/{total}] "
                    f"{scenario} | "
                    f"episode={episode} | "
                    f"seed={seed} | ERROR | "
                    f"{result['error']}"
                )

            else:

                print(
                    f"[{completed}/{total}] "
                    f"{scenario} | "
                    f"episode={episode} | "
                    f"seed={seed} | "
                    f"reward="
                    f"{result['reward']:.2f} | "
                    f"wait="
                    f"{result['waiting_time']:.2f} | "
                    f"queue="
                    f"{result['queue_length']:.2f} | "
                    f"speed="
                    f"{result['average_speed']:.2f} | "
                    f"throughput="
                    f"{result['throughput']}"
                )


    # --------------------------------------------------------------
    # Validate
    # --------------------------------------------------------------

    validate_results(
        results
    )


    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    create_outputs(
        results
    )


    print()
    print(
        "PPO-MARL ONLY evaluation finished."
    )


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()