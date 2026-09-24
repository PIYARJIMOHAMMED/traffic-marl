from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.environments.traffic_env import TrafficEnvironment


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


def run_scenario(scenario):

    print("\n" + "=" * 70)
    print(f"SCENARIO: {scenario}")
    print("=" * 70)

    env = TrafficEnvironment(
        scenario_name=scenario
    )

    try:

        observation, info = env.reset(seed=42)

        rewards = []
        queues = []
        waiting = []
        speeds = []
        vehicle_counts = []

        action = np.array(
            [0, 0, 0, 0],
            dtype=np.int64
        )

        for step in range(300):

            observation, reward, terminated, truncated, info = env.step(
                action
            )

            rewards.append(float(reward))

            obs = np.asarray(
                observation,
                dtype=np.float32
            )

            # Four agents × 30 graph features.
            for agent in range(4):

                start = agent * 30

                local = obs[start:start + 15]

                vehicle_counts.append(
                    float(local[0])
                )

                speeds.append(
                    float(local[1])
                )

                waiting.append(
                    float(local[2])
                )

                queue = (
                    float(local[5])
                    + float(local[6])
                    + float(local[7])
                    + float(local[8])
                )

                queues.append(queue)

            if terminated or truncated:
                break

        print(f"Steps:              {len(rewards)}")
        print(f"Total reward:        {sum(rewards):.2f}")
        print(f"Mean step reward:    {np.mean(rewards):.4f}")
        print(f"Mean vehicles:       {np.mean(vehicle_counts):.2f}")
        print(f"Mean speed:          {np.mean(speeds):.2f}")
        print(f"Mean waiting:        {np.mean(waiting):.2f}")
        print(f"Mean queue:          {np.mean(queues):.2f}")
        print(f"Maximum queue:       {np.max(queues):.2f}")

        return {
            "scenario": scenario,
            "steps": len(rewards),
            "reward": float(sum(rewards)),
            "vehicles": float(np.mean(vehicle_counts)),
            "speed": float(np.mean(speeds)),
            "waiting": float(np.mean(waiting)),
            "queue": float(np.mean(queues)),
        }

    finally:
        env.close()


def main():

    results = []

    for scenario in SCENARIOS:
        results.append(
            run_scenario(scenario)
        )

    print("\n\n")
    print("=" * 100)
    print("SCENARIO DIFFERENTIATION SUMMARY")
    print("=" * 100)

    print(
        f"{'Scenario':<22}"
        f"{'Reward':>14}"
        f"{'Vehicles':>14}"
        f"{'Speed':>12}"
        f"{'Waiting':>14}"
        f"{'Queue':>12}"
    )

    print("-" * 100)

    for r in results:

        print(
            f"{r['scenario']:<22}"
            f"{r['reward']:>14.2f}"
            f"{r['vehicles']:>14.2f}"
            f"{r['speed']:>12.2f}"
            f"{r['waiting']:>14.2f}"
            f"{r['queue']:>12.2f}"
        )

    print("=" * 100)


if __name__ == "__main__":
    main()