"""
MAI-07 | DQN 500K BASELINE EVALUATION

Purpose
-------
Evaluate ONLY the trained DQN 500K baseline under the same SUMO/TraCI
evaluation protocol used by the PPO-MARL proposed system.

Protocol
--------
- 10 SUMO traffic scenarios
- 3 episodes per scenario
- seeds 42, 43, 44
- deterministic DQN inference
- CPU inference
- live SUMO/TraCI metrics
- unified waiting-time and throughput definitions with PPO-MARL
- no hardcoded performance results

Model contract
--------------
- Observation: 120-D global observation
- DQN action: Discrete(16)
- Each discrete action is decoded into four binary intersection actions,
  matching the TrafficEnvironment's joint MultiDiscrete([2, 2, 2, 2]).

Outputs
-------
rl/results/dqn_500k/
    dqn_500k_raw.csv
    dqn_500k_summary.csv
    dqn_500k_scenario_summary.csv
    dqn_500k_report.json
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import traci
from stable_baselines3 import DQN


# ============================================================================
# PROJECT ROOT
# ============================================================================

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================================
# PROJECT IMPORTS
# ============================================================================

from rl.environments.config import EnvironmentConfig
from rl.environments.traffic_env import TrafficEnvironment


# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

MODEL_PATH = ROOT / "rl" / "checkpoints" / "dqn_500k" / "dqn_500k_final.zip"

OUTPUT_DIR = ROOT / "rl" / "results" / "dqn_500k"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = (
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
)

SEEDS = (42, 43, 44)
EPISODES_PER_SCENARIO = len(SEEDS)

# The environment remains the source of truth for the episode horizon.
MAX_STEPS = int(EnvironmentConfig.MAX_EPISODE_STEPS)

# DQN 500K model contract.
EXPECTED_OBSERVATION_DIM = 120
EXPECTED_ACTION_COUNT = 16
EXPECTED_AGENT_COUNT = 4
EXPECTED_BINARY_ACTIONS = 2


# ============================================================================
# GENERAL UTILITIES
# ============================================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to a finite float, otherwise return default."""
    try:
        converted = float(value)
        return converted if np.isfinite(converted) else default
    except (TypeError, ValueError):
        return default


def close_sumo() -> None:
    """Safely close the active TraCI connection."""
    try:
        if traci.isLoaded():
            traci.close()
    except Exception:
        pass


def validate_finite_array(values: np.ndarray, name: str) -> None:
    """Raise if an observation contains NaN or infinite values."""
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains NaN or infinite values.")


# ============================================================================
# DQN ACTION ADAPTER
# ============================================================================

def decode_joint_action(action_id: int) -> np.ndarray:
    """
    Decode DQN Discrete(16) into four binary intersection actions.

    Encoding:
        bit 0 -> Intersection A
        bit 1 -> Intersection B
        bit 2 -> Intersection C
        bit 3 -> Intersection D

    This creates the same 16 possible joint combinations as
    MultiDiscrete([2, 2, 2, 2]).
    """
    action_id = int(action_id)

    if not 0 <= action_id < EXPECTED_ACTION_COUNT:
        raise ValueError(
            f"DQN action must be in [0, {EXPECTED_ACTION_COUNT - 1}], "
            f"got {action_id}."
        )

    return np.asarray(
        [
            (action_id >> agent_index) & 1
            for agent_index in range(EXPECTED_AGENT_COUNT)
        ],
        dtype=np.int64,
    )


def validate_action_mapping() -> None:
    """Verify that all 16 DQN actions decode to valid four-agent actions."""
    decoded = [decode_joint_action(action_id) for action_id in range(EXPECTED_ACTION_COUNT)]

    if len(decoded) != EXPECTED_ACTION_COUNT:
        raise RuntimeError("DQN action mapping validation failed.")

    for action_id, action_vector in enumerate(decoded):
        if action_vector.shape != (EXPECTED_AGENT_COUNT,):
            raise RuntimeError(
                f"Action {action_id} decoded to shape {action_vector.shape}."
            )

        if not np.all(
            (action_vector >= 0)
            & (action_vector < EXPECTED_BINARY_ACTIONS)
        ):
            raise RuntimeError(
                f"Action {action_id} contains an invalid binary action."
            )

    unique_actions = {
        tuple(int(value) for value in action_vector)
        for action_vector in decoded
    }

    if len(unique_actions) != EXPECTED_ACTION_COUNT:
        raise RuntimeError("DQN action mapping is not one-to-one.")

    print("[PASS] DQN Discrete(16) -> four binary actions mapping validated.")


# ============================================================================
# LIVE SUMO METRICS
# ============================================================================

def get_live_metrics() -> tuple[float, float, float]:
    """
    Read live SUMO metrics after one simulation step.

    waiting_time:
        Sum of accumulated waiting time across vehicles currently in SUMO.

    queue_length:
        Number of currently present vehicles with speed <= 0.1 m/s.

    average_speed:
        Mean speed of currently present vehicles.
    """
    try:
        vehicle_ids = list(traci.vehicle.getIDList())
    except Exception:
        return 0.0, 0.0, 0.0

    if not vehicle_ids:
        return 0.0, 0.0, 0.0

    speeds: list[float] = []
    accumulated_waiting: list[float] = []
    queue_length = 0

    for vehicle_id in vehicle_ids:
        try:
            speed = safe_float(traci.vehicle.getSpeed(vehicle_id))
            waiting = safe_float(
                traci.vehicle.getAccumulatedWaitingTime(vehicle_id)
            )

            speeds.append(speed)
            accumulated_waiting.append(waiting)

            if speed <= 0.1:
                queue_length += 1

        except Exception:
            # A vehicle may disappear between getIDList() and the metric call.
            continue

    average_speed = float(np.mean(speeds)) if speeds else 0.0
    waiting_time = float(np.sum(accumulated_waiting)) if accumulated_waiting else 0.0

    return average_speed, waiting_time, float(queue_length)


# ============================================================================
# THROUGHPUT
# ============================================================================

def get_step_arrivals() -> int:
    """
    Return the number of vehicles that arrived during the CURRENT SUMO step.

    SUMO's getArrivedNumber() is a per-step value. Therefore episode
    throughput MUST be calculated by summing this value after every step.
    """
    try:
        return max(0, int(traci.simulation.getArrivedNumber()))
    except Exception:
        return 0


# ============================================================================
# MODEL VALIDATION
# ============================================================================

def load_and_validate_model() -> DQN:
    """Load DQN 500K and validate its observation/action spaces."""
    print("[CHECK] Loading DQN 500K...")
    print(f"[CHECK] Model path: {MODEL_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "DQN checkpoint not found:\n"
            f"{MODEL_PATH}"
        )

    model = DQN.load(
        str(MODEL_PATH),
        device="cpu",
    )

    observation_shape = tuple(model.observation_space.shape or ())
    print(f"[CHECK] Observation space: {model.observation_space}")

    if observation_shape != (EXPECTED_OBSERVATION_DIM,):
        raise ValueError(
            "Wrong DQN observation space.\n"
            f"Expected: ({EXPECTED_OBSERVATION_DIM},)\n"
            f"Found: {model.observation_space}"
        )

    print(f"[CHECK] Action space: {model.action_space}")

    if not isinstance(model.action_space, gym.spaces.Discrete):
        raise ValueError("DQN must use a Discrete action space.")

    if int(model.action_space.n) != EXPECTED_ACTION_COUNT:
        raise ValueError(
            "Wrong DQN action space.\n"
            f"Expected: Discrete({EXPECTED_ACTION_COUNT})\n"
            f"Found: {model.action_space}"
        )

    print("[PASS] DQN model contract validated.")
    print(f"       Observation: {EXPECTED_OBSERVATION_DIM}-D")
    print(f"       Action: Discrete({EXPECTED_ACTION_COUNT})")

    return model


# ============================================================================
# ENVIRONMENT VALIDATION
# ============================================================================

def validate_environment() -> None:
    """
    Validate every required scenario before the actual evaluation.

    This catches broken scenario configuration or observation-shape problems
    before producing result files.
    """
    print("\n[CHECK] Validating all evaluation scenarios...")

    for scenario in SCENARIOS:
        env = None

        try:
            env = TrafficEnvironment(
                scenario_name=scenario,
                render_mode=None,
            )

            observation, _ = env.reset(seed=SEEDS[0])
            observation = np.asarray(
                observation,
                dtype=np.float32,
            ).flatten()

            if observation.shape != (EXPECTED_OBSERVATION_DIM,):
                raise ValueError(
                    f"{scenario}: invalid observation shape "
                    f"{observation.shape}; expected "
                    f"({EXPECTED_OBSERVATION_DIM},)."
                )

            validate_finite_array(
                observation,
                f"{scenario} reset observation",
            )

            print(
                f"[PASS] {scenario:<22} "
                f"{EXPECTED_OBSERVATION_DIM}-D observation"
            )

        finally:
            try:
                if env is not None:
                    env.close()
            except Exception:
                pass

            close_sumo()

    print("[PASS] All scenarios validated.")


# ============================================================================
# EPISODE EVALUATION
# ============================================================================

def evaluate_episode(
    model: DQN,
    scenario: str,
    episode_number: int,
    seed: int,
) -> dict[str, Any]:
    """Evaluate one deterministic DQN episode in live SUMO."""
    start_time = time.perf_counter()
    env = None

    result: dict[str, Any] = {
        "policy": "dqn_500k",
        "model": "dqn_500k_final",
        "observation_type": "Global-120D",
        "action_space": "Discrete(16)",
        "underlying_action_space": "MultiDiscrete([2,2,2,2])",
        "scenario": scenario,
        "episode": int(episode_number),
        "seed": int(seed),
        "reward": 0.0,
        "waiting_time": 0.0,
        "queue_length": 0.0,
        "average_speed": 0.0,
        "throughput": 0,
        "steps": 0,
        "evaluation_time": 0.0,
        "terminated": False,
        "truncated": False,
        "error_type": "",
        "error": "",
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
        observation, _ = env.reset(seed=seed)

        observation = np.asarray(
            observation,
            dtype=np.float32,
        ).flatten()

        if observation.shape != (EXPECTED_OBSERVATION_DIM,):
            raise ValueError(
                f"Invalid reset observation shape: {observation.shape}. "
                f"Expected ({EXPECTED_OBSERVATION_DIM},)."
            )

        validate_finite_array(
            observation,
            "Reset observation",
        )

        # --------------------------------------------------------------
        # SIMULATION
        # --------------------------------------------------------------
        for step in range(MAX_STEPS):

            # DQN deterministic inference.
            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            action_array = np.asarray(action).reshape(-1)

            if action_array.size != 1:
                raise ValueError(
                    f"DQN returned unexpected action shape: {np.asarray(action).shape}"
                )

            action_id = int(action_array[0])

            if not 0 <= action_id < EXPECTED_ACTION_COUNT:
                raise ValueError(
                    f"DQN returned invalid action id: {action_id}"
                )

            # Convert DQN's single discrete action into the four-agent joint
            # action expected by TrafficEnvironment.
            joint_action = decode_joint_action(action_id)

            # ----------------------------------------------------------
            # ENVIRONMENT STEP
            # ----------------------------------------------------------
            (
                next_observation,
                reward,
                terminated,
                truncated,
                _,
            ) = env.step(joint_action)

            next_observation = np.asarray(
                next_observation,
                dtype=np.float32,
            ).flatten()

            if next_observation.shape != (EXPECTED_OBSERVATION_DIM,):
                raise ValueError(
                    f"Invalid step observation shape: {next_observation.shape}. "
                    f"Expected ({EXPECTED_OBSERVATION_DIM},)."
                )

            validate_finite_array(
                next_observation,
                "Step observation",
            )

            reward_value = safe_float(reward)

            if not np.isfinite(reward_value):
                raise ValueError("Reward is not finite.")

            # ----------------------------------------------------------
            # REWARD
            # ----------------------------------------------------------
            reward_sum += reward_value

            # ----------------------------------------------------------
            # LIVE SUMO METRICS
            # ----------------------------------------------------------
            (
                average_speed,
                waiting_time,
                queue_length,
            ) = get_live_metrics()

            speed_sum += average_speed
            waiting_sum += waiting_time
            queue_sum += queue_length

            # ----------------------------------------------------------
            # THROUGHPUT
            # ----------------------------------------------------------
            # IMPORTANT:
            # getArrivedNumber() is current-step arrivals, so accumulate it.
            throughput_sum += get_step_arrivals()

            # ----------------------------------------------------------
            # UPDATE
            # ----------------------------------------------------------
            observation = next_observation

            result["steps"] = step + 1
            result["terminated"] = bool(terminated)
            result["truncated"] = bool(truncated)

            if terminated or truncated:
                break

        # --------------------------------------------------------------
        # FINAL EPISODE METRICS
        # --------------------------------------------------------------
        steps = max(1, int(result["steps"]))

        result["reward"] = float(reward_sum)
        result["waiting_time"] = float(waiting_sum / steps)
        result["queue_length"] = float(queue_sum / steps)
        result["average_speed"] = float(speed_sum / steps)
        result["throughput"] = int(throughput_sum)

    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)

    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass

        close_sumo()

    result["evaluation_time"] = float(
        time.perf_counter() - start_time
    )

    result["timestamp"] = pd.Timestamp.now(tz="UTC").isoformat()

    return result


# ============================================================================
# RESULT VALIDATION
# ============================================================================

def validate_results(results: list[dict[str, Any]]) -> None:
    """Validate completeness and numerical integrity of all episodes."""
    expected_count = len(SCENARIOS) * EPISODES_PER_SCENARIO

    if len(results) != expected_count:
        raise RuntimeError(
            "Unexpected evaluation count.\n"
            f"Expected: {expected_count}\n"
            f"Found: {len(results)}"
        )

    errors = [
        result
        for result in results
        if result.get("error")
    ]

    if errors:
        print("\n[ERROR] Failed evaluation episodes:")

        for result in errors:
            print(
                f"{result['scenario']} | "
                f"episode={result['episode']} | "
                f"seed={result['seed']} | "
                f"{result['error_type']} | "
                f"{result['error']}"
            )

        raise RuntimeError(
            f"{len(errors)} evaluation episode(s) failed."
        )

    required_metrics = (
        "reward",
        "waiting_time",
        "queue_length",
        "average_speed",
        "throughput",
        "steps",
    )

    for result in results:
        if int(result["steps"]) <= 0:
            raise RuntimeError(
                f"Invalid step count for "
                f"{result['scenario']} episode {result['episode']}."
            )

        for metric in required_metrics:
            value = safe_float(result[metric])

            if not np.isfinite(value):
                raise RuntimeError(
                    f"Non-finite metric: {metric}={value}"
                )

        if result["throughput"] < 0:
            raise RuntimeError(
                f"Negative throughput in "
                f"{result['scenario']} episode {result['episode']}."
            )

    print(
        f"[PASS] {len(results)} evaluation episodes completed "
        "with valid metrics and no errors."
    )


# ============================================================================
# OUTPUT GENERATION
# ============================================================================

def create_outputs(results: list[dict[str, Any]]) -> None:
    """
    Create raw, scenario-level, overall and JSON evidence files.

    All performance values are calculated from the evaluated episodes.
    No performance result is hardcoded.
    """
    df = pd.DataFrame(results)

    # ------------------------------------------------------------------
    # RAW EPISODE RESULTS
    # ------------------------------------------------------------------
    raw_path = OUTPUT_DIR / "dqn_500k_raw.csv"
    df.to_csv(raw_path, index=False)

    # ------------------------------------------------------------------
    # SCENARIO SUMMARY
    # ------------------------------------------------------------------
    scenario_summary = (
        df
        .groupby(
            ["policy", "scenario"],
            as_index=False,
        )
        .agg(
            episodes=("episode", "count"),
            reward_mean=("reward", "mean"),
            reward_std=("reward", "std"),
            waiting_time_mean=("waiting_time", "mean"),
            queue_length_mean=("queue_length", "mean"),
            average_speed_mean=("average_speed", "mean"),
            throughput_mean=("throughput", "mean"),
            steps_mean=("steps", "mean"),
        )
    )

    scenario_summary["reward_std"] = (
        scenario_summary["reward_std"].fillna(0.0)
    )

    scenario_path = (
        OUTPUT_DIR / "dqn_500k_scenario_summary.csv"
    )

    scenario_summary.to_csv(
        scenario_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # OVERALL SUMMARY
    # ------------------------------------------------------------------
    overall = pd.DataFrame(
        [
            {
                "policy": "dqn_500k",
                "episodes": int(len(df)),
                "reward_mean": float(df["reward"].mean()),
                "reward_std": (
                    float(df["reward"].std(ddof=1))
                    if len(df) > 1
                    else 0.0
                ),
                "waiting_time_mean": float(
                    df["waiting_time"].mean()
                ),
                "queue_length_mean": float(
                    df["queue_length"].mean()
                ),
                "average_speed_mean": float(
                    df["average_speed"].mean()
                ),
                "throughput_mean": float(
                    df["throughput"].mean()
                ),
                "steps_mean": float(
                    df["steps"].mean()
                ),
            }
        ]
    )

    overall_path = OUTPUT_DIR / "dqn_500k_summary.csv"

    overall.to_csv(
        overall_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # JSON REPORT
    # ------------------------------------------------------------------
    report = {
        "project": (
            "MAI-07 | Multi-Agent Reinforcement Learning "
            "for Network-Wide Adaptive Traffic Control"
        ),
        "evaluation": "DQN 500K Baseline Evaluation",
        "controller": "DQN",
        "role": "Primary Baseline",
        "algorithm": "DQN",
        "training_timesteps": 500000,
        "observation_type": "Global-120D",
        "observation_dimension": EXPECTED_OBSERVATION_DIM,
        "action_space": "Discrete(16)",
        "underlying_action_space": (
            "MultiDiscrete([2,2,2,2])"
        ),
        "number_of_agents": EXPECTED_AGENT_COUNT,
        "scenarios": list(SCENARIOS),
        "episodes_per_scenario": EPISODES_PER_SCENARIO,
        "total_episodes": len(results),
        "seeds": list(SEEDS),
        "max_episode_steps": MAX_STEPS,
        "deterministic_inference": True,
        "device": "cpu",
        "data_source": "Live SUMO/TraCI simulation",
        "hardcoded_performance_results": False,
        "metrics": {
            "reward": "Cumulative environment reward",
            "waiting_time": (
                "Mean per-step sum of vehicle accumulated waiting time"
            ),
            "queue_length": (
                "Mean per-step number of vehicles "
                "with speed <= 0.1 m/s"
            ),
            "average_speed": (
                "Mean vehicle speed from SUMO"
            ),
            "throughput": (
                "Total vehicles arriving during the episode, "
                "calculated by summing SUMO current-step arrivals"
            ),
        },
        "overall": overall.to_dict(
            orient="records"
        )[0],
        "scenario_summary": scenario_summary.to_dict(
            orient="records"
        ),
        "episode_results": results,
        "evaluation_runtime_seconds": float(
            sum(
                safe_float(result["evaluation_time"])
                for result in results
            )
        ),
        "timestamp": pd.Timestamp.now(
            tz="UTC"
        ).isoformat(),
    }

    report_path = (
        OUTPUT_DIR / "dqn_500k_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # CONSOLE SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("DQN 500K FINAL EVALUATION COMPLETE")
    print("=" * 72)
    print(f"Raw CSV      : {raw_path}")
    print(f"Summary CSV  : {overall_path}")
    print(f"Scenario CSV : {scenario_path}")
    print(f"JSON report  : {report_path}")
    print("=" * 72)

    print("\nOVERALL RESULTS")
    print("-" * 72)

    row = overall.iloc[0]

    print(f"Reward       : {row['reward_mean']:.4f}")
    print(f"Waiting time : {row['waiting_time_mean']:.4f}")
    print(f"Queue length : {row['queue_length_mean']:.4f}")
    print(f"Avg speed    : {row['average_speed_mean']:.4f}")
    print(f"Throughput   : {row['throughput_mean']:.4f}")
    print(f"Steps        : {row['steps_mean']:.4f}")

    print("\nSCENARIO RESULTS")
    print("-" * 72)

    display_columns = [
        "scenario",
        "reward_mean",
        "waiting_time_mean",
        "queue_length_mean",
        "average_speed_mean",
        "throughput_mean",
    ]

    print(
        scenario_summary[display_columns].to_string(
            index=False
        )
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    print("\n" + "=" * 72)
    print("MAI-07 | DQN 500K BASELINE EVALUATION")
    print("=" * 72)
    print("Controller     : DQN ONLY")
    print("Role           : Primary Baseline")
    print("Observation    : 120-D Global")
    print("Action         : Discrete(16)")
    print("Action mapping : Four binary intersection actions")
    print(f"Scenarios      : {len(SCENARIOS)}")
    print(f"Episodes/scen  : {EPISODES_PER_SCENARIO}")
    print(f"Seeds          : {list(SEEDS)}")
    print(f"Max steps      : {MAX_STEPS}")
    print("Inference      : Deterministic / CPU")
    print("Metrics        : Live SUMO/TraCI")
    print("Waiting metric : Accumulated waiting time")
    print("Throughput     : Sum of per-step arrivals")
    print("Hardcoded      : No performance results")
    print("=" * 72)

    # Reproducibility for evaluator-side operations.
    random.seed(SEEDS[0])
    np.random.seed(SEEDS[0])

    # Use headless SUMO for evaluation.
    EnvironmentConfig.SUMO_BINARY = "sumo"

    # Validate before running the expensive evaluation.
    validate_action_mapping()
    model = load_and_validate_model()
    validate_environment()

    results: list[dict[str, Any]] = []

    total = len(SCENARIOS) * EPISODES_PER_SCENARIO
    completed = 0

    print("\n" + "=" * 72)
    print("STARTING DQN 500K EVALUATION")
    print("=" * 72)

    for scenario in SCENARIOS:
        print(f"\n----- {scenario.upper()} -----")

        for episode_number, seed in zip(
            range(1, EPISODES_PER_SCENARIO + 1),
            SEEDS,
        ):
            completed += 1

            result = evaluate_episode(
                model=model,
                scenario=scenario,
                episode_number=episode_number,
                seed=seed,
            )

            results.append(result)

            if result["error"]:
                print(
                    f"[{completed}/{total}] "
                    f"{scenario} | "
                    f"episode={episode_number} | "
                    f"seed={seed} | ERROR | "
                    f"{result['error_type']}: "
                    f"{result['error']}"
                )
            else:
                print(
                    f"[{completed}/{total}] "
                    f"{scenario} | "
                    f"ep={episode_number} | "
                    f"seed={seed} | "
                    f"reward={result['reward']:.2f} | "
                    f"wait={result['waiting_time']:.2f} | "
                    f"queue={result['queue_length']:.2f} | "
                    f"speed={result['average_speed']:.2f} | "
                    f"throughput={result['throughput']}"
                )

    # Never write final evidence if an episode failed.
    validate_results(results)
    create_outputs(results)

    print("\n[PASS] DQN 500K evaluation finished successfully.")


if __name__ == "__main__":
    main()
