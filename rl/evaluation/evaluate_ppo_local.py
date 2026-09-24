from pathlib import Path
import sys
import csv
import json
import statistics
import numpy as np

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from stable_baselines3 import PPO
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box

from rl.environments.traffic_env import TrafficEnvironment
from rl.environments.config import EnvironmentConfig


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "rl"
    / "checkpoints"
    / "ppo_marl_250000_steps.zip"
)

OUTPUT_DIR = PROJECT_ROOT / "rl" / "evaluation"

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

MAX_STEPS = EnvironmentConfig.MAX_EPISODE_STEPS

SEED = 42


# ============================================================
# LEGACY 120-D -> 60-D OBSERVATION WRAPPER
# ============================================================

class Legacy60ObservationWrapper(ObservationWrapper):
    """
    Converts the current 120-dimensional graph observation into
    the 60-dimensional local observation expected by the
    previously trained PPO model.

    Current agent observation:

        local 15 + neighbour 15 = 30

    Four agents:

        30 x 4 = 120

    Legacy PPO:

        local 15 x 4 = 60

    Therefore we keep only the first 15 values of each
    agent's 30-dimensional observation.
    """

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = Box(
            low=-100000.0,
            high=100000.0,
            shape=(60,),
            dtype=np.float32,
        )

    def observation(self, observation):
        observation = np.asarray(
            observation,
            dtype=np.float32
        ).reshape(-1)

        if observation.shape[0] != 120:
            raise ValueError(
                f"Expected current observation size 120, "
                f"received {observation.shape[0]}"
            )

        local_observations = []

        for agent_index in range(4):
            start = agent_index * 30
            end = start + 15

            local_observations.append(
                observation[start:end]
            )

        legacy_observation = np.concatenate(
            local_observations
        ).astype(np.float32)

        if legacy_observation.shape != (60,):
            raise ValueError(
                "Legacy observation conversion failed: "
                f"expected (60,), got {legacy_observation.shape}"
            )

        return legacy_observation


# ============================================================
# METRIC HELPERS
# ============================================================

def safe_mean(values):
    if not values:
        return 0.0
    return float(statistics.mean(values))


def safe_max(values):
    if not values:
        return 0.0
    return float(max(values))


# ============================================================
# SINGLE EPISODE
# ============================================================

def run_episode(model, scenario_name, episode_number):
    """
    Run one complete evaluation episode.
    """

    base_env = TrafficEnvironment(
        scenario_name=scenario_name
    )

    env = Legacy60ObservationWrapper(base_env)

    observation, info = env.reset(
        seed=SEED + episode_number
    )

    episode_reward = 0.0

    waiting_times = []
    queue_lengths = []
    average_speeds = []
    vehicle_counts = []

    throughput = 0

    steps = 0

    terminated = False
    truncated = False

    try:

        while not terminated and not truncated:
            action, _ = model.predict(
                observation,
                deterministic=True
            )

            action = np.asarray(action).reshape(-1)

            if action.shape != (4,):
                raise ValueError(
                    f"Expected action shape (4,), "
                    f"received {action.shape}"
                )

            action = action.astype(np.int64)

            observation, reward, terminated, truncated, info = env.step(
                action
            )

            episode_reward += float(reward)

            # ------------------------------------------------
            # Current observation is 60-D here.
            #
            # Each agent has 15 local features:
            #
            # 0 vehicle count
            # 1 average speed
            # 2 waiting time
            # 3 stopped vehicles
            # 4 phase
            # 5 north queue
            # 6 east queue
            # 7 south queue
            # 8 west queue
            # ------------------------------------------------

            obs = np.asarray(
                observation,
                dtype=np.float32
            )

            for agent_index in range(4):

                start = agent_index * 15

                local = obs[
                    start:start + 15
                ]

                vehicle_counts.append(
                    float(local[0])
                )

                average_speeds.append(
                    float(local[1])
                )

                waiting_times.append(
                    float(local[2])
                )

                queue = (
                    float(local[5])
                    + float(local[6])
                    + float(local[7])
                    + float(local[8])
                )

                queue_lengths.append(queue)

            # ------------------------------------------------
            # Throughput
            # ------------------------------------------------

            try:
                import traci

                arrived = traci.simulation.getArrivedNumber()

                if arrived is not None:
                    throughput += int(arrived)

            except Exception:
                pass

            steps += 1

            if steps >= MAX_STEPS:
                break

    finally:
        env.close()

    result = {
        "controller": "ppo_local",
        "scenario": scenario_name,
        "episode": episode_number,
        "total_reward": float(episode_reward),
        "average_reward": (
            float(episode_reward / steps)
            if steps > 0
            else 0.0
        ),
        "average_waiting_time": safe_mean(waiting_times),
        "maximum_waiting_time": safe_max(waiting_times),
        "average_queue_length": safe_mean(queue_lengths),
        "maximum_queue_length": safe_max(queue_lengths),
        "average_speed": safe_mean(average_speeds),
        "average_vehicle_count": safe_mean(vehicle_counts),
        "throughput": int(throughput),
        "episode_length": int(steps),
    }

    print(
        f"[PPO-LOCAL] "
        f"{scenario_name:<20} "
        f"Episode {episode_number}: "
        f"Reward={result['total_reward']:.2f}, "
        f"Wait={result['average_waiting_time']:.2f}, "
        f"Queue={result['average_queue_length']:.2f}, "
        f"Speed={result['average_speed']:.2f}, "
        f"Throughput={result['throughput']}"
    )

    return result


# ============================================================
# RUN ALL SCENARIOS
# ============================================================

def run_evaluation():
    print("=" * 70)
    print("PPO-LOCAL 250K EVALUATION")
    print("=" * 70)

    print(f"Model: {MODEL_PATH}")
    print(f"Observation: 60-D legacy")
    print(f"Current environment: 120-D")
    print(f"Scenarios: {len(SCENARIOS)}")
    print(f"Episodes per scenario: {EPISODES_PER_SCENARIO}")
    print(f"Maximum steps: {MAX_STEPS}")
    print("=" * 70)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"PPO model not found:\n{MODEL_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load model WITHOUT environment first.
    # --------------------------------------------------------

    model = PPO.load(
        str(MODEL_PATH),
        device="auto"
    )

    print("\nLoaded PPO model successfully.")

    print(
        f"Model observation space: "
        f"{model.observation_space}"
    )

    print(
        f"Model action space: "
        f"{model.action_space}"
    )

    if model.observation_space.shape != (60,):
        raise ValueError(
            "The selected PPO model is not a 60-D model."
        )

    if tuple(model.action_space.nvec) != (2, 2, 2, 2):
        raise ValueError(
            "Unexpected PPO action space. "
            f"Received: {model.action_space}"
        )

    # --------------------------------------------------------
    # Run evaluation
    # --------------------------------------------------------

    all_results = []

    for scenario in SCENARIOS:

        print("\n" + "-" * 70)
        print(f"SCENARIO: {scenario}")
        print("-" * 70)

        for episode in range(
            1,
            EPISODES_PER_SCENARIO + 1
        ):

            result = run_episode(
                model=model,
                scenario_name=scenario,
                episode_number=episode,
            )

            all_results.append(result)

    # ========================================================
    # SAVE RAW RESULTS
    # ========================================================

    raw_csv = (
        OUTPUT_DIR
        / "ppo_local_250k_results.csv"
    )

    fieldnames = list(
        all_results[0].keys()
    )

    with open(
        raw_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_results)

    # ========================================================
    # SCENARIO SUMMARY
    # ========================================================

    scenario_summary = []

    for scenario in SCENARIOS:

        rows = [
            r
            for r in all_results
            if r["scenario"] == scenario
        ]

        summary = {
            "controller": "ppo_local",
            "scenario": scenario,
            "episodes": len(rows),

            "mean_total_reward": safe_mean(
                [r["total_reward"] for r in rows]
            ),

            "mean_average_reward": safe_mean(
                [r["average_reward"] for r in rows]
            ),

            "mean_waiting_time": safe_mean(
                [r["average_waiting_time"] for r in rows]
            ),

            "mean_maximum_waiting_time": safe_mean(
                [r["maximum_waiting_time"] for r in rows]
            ),

            "mean_queue_length": safe_mean(
                [r["average_queue_length"] for r in rows]
            ),

            "mean_maximum_queue_length": safe_mean(
                [r["maximum_queue_length"] for r in rows]
            ),

            "mean_average_speed": safe_mean(
                [r["average_speed"] for r in rows]
            ),

            "mean_vehicle_count": safe_mean(
                [r["average_vehicle_count"] for r in rows]
            ),

            "mean_throughput": safe_mean(
                [r["throughput"] for r in rows]
            ),

            "mean_episode_length": safe_mean(
                [r["episode_length"] for r in rows]
            ),
        }

        scenario_summary.append(summary)

    summary_csv = (
        OUTPUT_DIR
        / "ppo_local_250k_summary.csv"
    )

    summary_fields = list(
        scenario_summary[0].keys()
    )

    with open(
        summary_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=summary_fields
        )

        writer.writeheader()
        writer.writerows(scenario_summary)

    # ========================================================
    # JSON
    # ========================================================

    json_path = (
        OUTPUT_DIR
        / "ppo_local_250k_results.json"
    )

    json_data = {
        "model": "ppo_marl_250000_steps",
        "controller": "ppo_local",
        "observation_dimension": 60,
        "current_environment_dimension": 120,
        "scenarios": SCENARIOS,
        "episodes_per_scenario": EPISODES_PER_SCENARIO,
        "results": all_results,
        "summary": scenario_summary,
    }

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            json_data,
            file,
            indent=2
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PPO-LOCAL EVALUATION COMPLETE")
    print("=" * 70)

    print(f"Raw results:")
    print(raw_csv)

    print(f"\nSummary:")
    print(summary_csv)

    print(f"\nJSON:")
    print(json_path)

    print("\nScenario Summary:")
    print("-" * 70)

    for row in scenario_summary:

        print(
            f"{row['scenario']:<20} "
            f"Reward={row['mean_total_reward']:>12.2f} | "
            f"Wait={row['mean_waiting_time']:>8.2f} | "
            f"Queue={row['mean_queue_length']:>8.2f} | "
            f"Speed={row['mean_average_speed']:>8.2f} | "
            f"Throughput={row['mean_throughput']:>8.2f}"
        )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_evaluation()