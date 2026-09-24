from pathlib import Path
import argparse
import json
import math
import shutil
import time
import xml.etree.ElementTree as ET

import numpy as np
from stable_baselines3 import PPO

# ============================================================
# MAI-07 LEVEL 1 RESEARCH EVALUATION
# ============================================================
# Adds evidence for:
#   1. Queue balance
#   2. Uncertainty / 95% confidence intervals
#   3. PPO inference computation cost
#   4. Distribution-shift transfer/generalization
#   5. Safety/failure audit summary
#
# IMPORTANT:
# - This script DOES NOT retrain PPO.
# - It uses the completed PPO-MARL 500K model.
# - Transfer uses a temporary demand-shifted copy of the
#   baseline SUMO scenario and deletes it afterwards.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT
    / "rl"
    / "checkpoints"
    / "ppo_marl_500k"
    / "ppo_marl_500k_final.zip"
)

RESULTS_DIR = ROOT / "rl" / "results" / "level1"

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

INTERSECTIONS = ["A", "B", "C", "D"]
MAX_STEPS = 300
BASE_SEED = 42


# ============================================================
# STATISTICS
# ============================================================

def confidence_interval_95(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return 0.0, 0.0

    mean_value = float(np.mean(values))

    if len(values) == 1:
        return mean_value, mean_value

    std_value = float(np.std(values, ddof=1))
    standard_error = std_value / math.sqrt(len(values))
    margin = 1.96 * standard_error

    return (
        mean_value - margin,
        mean_value + margin,
    )


def safe_std(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) <= 1:
        return 0.0

    return float(np.std(values, ddof=1))


# ============================================================
# OBSERVATION HELPERS
# ============================================================

def get_local_blocks(observation):
    """
    Current MAI-07 observation:

        A = 30 values
        B = 30 values
        C = 30 values
        D = 30 values

    First 15 values are local features.

    Local indices:
        0 vehicle count
        1 average speed
        2 waiting time
        3 stopped vehicles
        4 phase
        5 north queue
        6 east queue
        7 south queue
        8 west queue
        9 lane density
        10 green timer
        11 yellow timer
        12 current action
        13 in yellow
        14 reserved
    """

    observation = np.asarray(
        observation,
        dtype=np.float32,
    ).reshape(-1)

    if observation.shape != (120,):
        raise ValueError(
            "Expected 120-D observation, "
            f"received {observation.shape}"
        )

    return [
        observation[index * 30:index * 30 + 15]
        for index in range(4)
    ]


def calculate_step_metrics(observation):
    blocks = get_local_blocks(observation)

    queues = []
    waiting = []
    speeds = []
    vehicles = []
    directional_imbalances = []

    for local in blocks:
        directional_queue = np.asarray(
            local[5:9],
            dtype=float,
        )

        queue_total = float(
            np.sum(directional_queue)
        )

        queue_imbalance = float(
            np.max(directional_queue)
            - np.min(directional_queue)
        )

        queues.append(queue_total)
        waiting.append(float(local[2]))
        speeds.append(float(local[1]))
        vehicles.append(float(local[0]))
        directional_imbalances.append(
            queue_imbalance
        )

    return {
        "queue": float(np.mean(queues)),
        "waiting": float(np.mean(waiting)),
        "speed": float(np.mean(speeds)),
        "vehicles": float(np.mean(vehicles)),
        "queue_imbalance": float(
            np.mean(directional_imbalances)
        ),
    }


# ============================================================
# MODEL CHECK
# ============================================================

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "\nPPO-MARL 500K model was not found:\n"
            f"{MODEL_PATH}\n\n"
            "Expected the completed model at:\n"
            "rl/checkpoints/ppo_marl_500k/"
            "ppo_marl_500k_final.zip"
        )

    model = PPO.load(
        str(MODEL_PATH),
        device="auto",
    )

    if model.observation_space.shape != (120,):
        raise ValueError(
            "Expected PPO-MARL observation "
            f"shape (120,), got "
            f"{model.observation_space.shape}"
        )

    if tuple(model.action_space.nvec) != (
        2,
        2,
        2,
        2,
    ):
        raise ValueError(
            "Unexpected PPO-MARL action space: "
            f"{model.action_space}"
        )

    return model


# ============================================================
# INFERENCE COMPUTATION COST
# ============================================================

def benchmark_inference(model, observation):
    """
    Measures PPO policy inference only.

    This is NOT SUMO runtime and NOT full episode latency.
    It is the computational cost of producing an action
    from a 120-D observation.
    """

    latencies_ms = []

    for _ in range(200):
        start = time.perf_counter()

        model.predict(
            observation,
            deterministic=True,
        )

        elapsed = (
            time.perf_counter() - start
        ) * 1000.0

        latencies_ms.append(
            elapsed
        )

    return {
        "samples": len(latencies_ms),
        "mean_ms": float(
            np.mean(latencies_ms)
        ),
        "median_ms": float(
            np.median(latencies_ms)
        ),
        "p95_ms": float(
            np.percentile(
                latencies_ms,
                95,
            )
        ),
        "max_ms": float(
            np.max(latencies_ms)
        ),
    }


# ============================================================
# ONE EPISODE
# ============================================================

def run_episode(
    model,
    scenario,
    episode_number,
):
    from rl.environments.traffic_env import (
        TrafficEnvironment,
    )

    print()
    print(
        f"SCENARIO: {scenario} | "
        f"EPISODE: {episode_number}"
    )

    env = TrafficEnvironment(
        scenario_name=scenario
    )

    observation, info = env.reset(
        seed=BASE_SEED + episode_number - 1
    )

    total_reward = 0.0
    step_metrics = []
    throughput = 0
    steps = 0

    evaluation_start = time.perf_counter()

    try:
        for _ in range(MAX_STEPS):

            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            total_reward += float(reward)
            steps += 1

            step_metrics.append(
                calculate_step_metrics(
                    observation
                )
            )

            try:
                import traci

                arrived = (
                    traci.simulation
                    .getArrivedNumber()
                )

                throughput += int(
                    arrived or 0
                )

            except Exception:
                pass

            if terminated or truncated:
                break

    finally:
        env.close()

    if not step_metrics:
        raise RuntimeError(
            "No metrics were collected."
        )

    return {
        "scenario": scenario,
        "episode": episode_number,
        "reward": float(total_reward),
        "waiting_time": float(
            np.mean([
                row["waiting"]
                for row in step_metrics
            ])
        ),
        "queue_length": float(
            np.mean([
                row["queue"]
                for row in step_metrics
            ])
        ),
        "average_speed": float(
            np.mean([
                row["speed"]
                for row in step_metrics
            ])
        ),
        "vehicle_count": float(
            np.mean([
                row["vehicles"]
                for row in step_metrics
            ])
        ),
        "queue_imbalance": float(
            np.mean([
                row["queue_imbalance"]
                for row in step_metrics
            ])
        ),
        "max_queue_imbalance": float(
            np.max([
                row["queue_imbalance"]
                for row in step_metrics
            ])
        ),
        "throughput": int(throughput),
        "steps": int(steps),
        "evaluation_time_s": float(
            time.perf_counter()
            - evaluation_start
        ),
    }


# ============================================================
# TEMPORARY TRANSFER SCENARIO
# ============================================================

def make_transfer_scenario():
    """
    Creates a temporary scenario derived from baseline.

    Demand shift:
      - flow probabilities increase by 35%
      - period-based flows become 35% more frequent

    The scenario is intentionally NOT part of the normal
    ten-scenario training list. It is used as a simple
    distribution-shift/generalization test.

    It is deleted automatically after the test.
    """

    source = (
        ROOT
        / "simulation"
        / "scenarios"
        / "baseline"
    )

    target = (
        ROOT
        / "simulation"
        / "scenarios"
        / "__level1_transfer_shift"
    )

    if not source.exists():
        raise FileNotFoundError(
            f"Baseline scenario not found: {source}"
        )

    if target.exists():
        shutil.rmtree(target)

    shutil.copytree(
        source,
        target,
    )

    route_file = (
        target / "routes.rou.xml"
    )

    tree = ET.parse(route_file)
    root = tree.getroot()

    probability_changes = 0
    period_changes = 0

    for flow in root.findall(".//flow"):

        if "probability" in flow.attrib:
            old_value = float(
                flow.attrib["probability"]
            )

            new_value = min(
                old_value * 1.35,
                0.95,
            )

            flow.set(
                "probability",
                f"{new_value:.3f}",
            )

            probability_changes += 1

        elif "period" in flow.attrib:
            old_value = float(
                flow.attrib["period"]
            )

            new_value = max(
                old_value / 1.35,
                5.0,
            )

            flow.set(
                "period",
                f"{new_value:.2f}",
            )

            period_changes += 1

    tree.write(
        route_file,
        encoding="utf-8",
        xml_declaration=True,
    )

    return {
        "scenario_name":
            "__level1_transfer_shift",
        "source":
            str(source),
        "target":
            str(target),
        "probability_changes":
            probability_changes,
        "period_changes":
            period_changes,
        "description":
            "Temporary 35% demand-intensity shift "
            "derived from baseline; no PPO retraining.",
    }


def run_transfer(
    model,
    episodes,
):
    from rl.environments.traffic_env import (
        TrafficEnvironment,
    )

    transfer_name = (
        "__level1_transfer_shift"
    )

    original_scenarios = list(
        TrafficEnvironment.SCENARIOS
    )

    TrafficEnvironment.SCENARIOS = list(
        dict.fromkeys(
            original_scenarios
            + [transfer_name]
        )
    )

    try:
        rows = []

        for episode in range(
            1,
            episodes + 1,
        ):
            rows.append(
                run_episode(
                    model,
                    transfer_name,
                    episode,
                )
            )

        return rows

    finally:
        TrafficEnvironment.SCENARIOS = (
            original_scenarios
        )

        shutil.rmtree(
            ROOT
            / "simulation"
            / "scenarios"
            / transfer_name,
            ignore_errors=True,
        )


# ============================================================
# SUMMARY
# ============================================================

def summarize(rows):
    import pandas as pd

    df = pd.DataFrame(rows)

    summary_rows = []

    metrics = [
        "reward",
        "waiting_time",
        "queue_length",
        "average_speed",
        "throughput",
        "queue_imbalance",
        "max_queue_imbalance",
        "evaluation_time_s",
    ]

    for scenario, group in df.groupby(
        "scenario"
    ):

        record = {
            "scenario": scenario,
            "episodes": int(len(group)),
            "steps_mean": float(
                group["steps"].mean()
            ),
        }

        for metric in metrics:

            values = (
                group[metric]
                .astype(float)
                .to_numpy()
            )

            low, high = (
                confidence_interval_95(
                    values
                )
            )

            record[
                f"{metric}_mean"
            ] = float(np.mean(values))

            record[
                f"{metric}_std"
            ] = safe_std(values)

            record[
                f"{metric}_ci95_low"
            ] = low

            record[
                f"{metric}_ci95_high"
            ] = high

        summary_rows.append(record)

    return pd.DataFrame(
        summary_rows
    )


# ============================================================
# SAFETY AUDIT
# ============================================================

def find_safety_audits():
    candidates = []

    for base in [
        ROOT / "rl" / "logs",
        ROOT / "rl" / "results",
    ]:
        if base.exists():
            candidates.extend(
                base.rglob("*audit*.csv")
            )

    # Prefer the existing safety evaluation audit.
    preferred = [
        p for p in candidates
        if (
            "ppo_marl_safe"
            in p.name.lower()
            or "safety"
            in str(p.parent).lower()
        )
    ]

    return list(
        dict.fromkeys(
            preferred + candidates
        )
    )


def safety_audit_summary():
    import pandas as pd

    files = find_safety_audits()

    if not files:
        return {
            "status": "NOT_FOUND",
            "message":
                "No safety audit CSV was found. "
                "Run the existing safety/governance "
                "evaluation before final submission.",
        }

    audit_file = files[0]

    df = pd.read_csv(
        audit_file
    )

    columns = {
        str(column).lower():
            column
        for column in df.columns
    }

    # The dashboard audit schema uses status/reason and does not
    # contain the older ``overridden`` / ``valid_action`` columns.
    # Parse the actual production audit schema directly.
    status_column = columns.get("status")
    reason_column = columns.get("reason")
    proposed_column = columns.get("proposed_action")
    approved_column = columns.get("safety_approved_action")
    final_column = columns.get("final_action")

    result = {
        "status": "FOUND",
        "file": str(audit_file),
        "records": int(len(df)),
    }

    if status_column:
        status_values = (
            df[status_column]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )

        result["status_counts"] = {
            str(key): int(value)
            for key, value in status_values.value_counts().items()
            if str(key).strip()
        }

        # SAFETY_OVERRIDE is an intervention, NOT a violation.
        result["overrides"] = int(
            (status_values == "SAFETY_OVERRIDE").sum()
        )

        result["human_rejected"] = int(
            (status_values == "HUMAN_REJECTED").sum()
        )
        result["human_approved"] = int(
            (status_values == "HUMAN_APPROVED").sum()
        )

        explicit_violation_statuses = {
            "VIOLATION",
            "SAFETY_VIOLATION",
            "INVALID_ACTION",
        }
        result["explicit_violation_records"] = int(
            status_values.isin(explicit_violation_statuses).sum()
        )
    else:
        result["status_counts"] = {}
        result["overrides"] = None
        result["human_rejected"] = None
        result["human_approved"] = None
        result["explicit_violation_records"] = None

    # Independently validate the numeric action fields.
    # Valid MAI-07 actions are only 0 (North/South) and 1 (East/West).
    invalid_action_records = 0
    for col in [proposed_column, approved_column, final_column]:
        if col:
            numeric = pd.to_numeric(df[col], errors="coerce")
            invalid_action_records += int(
                ((numeric.isna()) | (~numeric.isin([0, 1]))).sum()
            )

    result["invalid_action_records"] = invalid_action_records
    result["violations"] = int(
        (result.get("explicit_violation_records") or 0)
        + invalid_action_records
    )

    if reason_column:
        reasons = (
            df[reason_column]
            .fillna("")
            .astype(str)
            .str.strip()
            .value_counts()
            .head(10)
        )

        result["top_safety_reasons"] = {
            str(key): int(value)
            for key, value in reasons.items()
            if str(key).strip()
        }
    else:
        result["top_safety_reasons"] = {}

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "MAI-07 Level 1 research "
            "evaluation"
        )
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help=(
            "Episodes per normal scenario "
            "(default: 3)"
        ),
    )

    parser.add_argument(
        "--transfer",
        action="store_true",
        help=(
            "Also run temporary "
            "distribution-shift transfer test."
        ),
    )

    parser.add_argument(
        "--audit-only",
        action="store_true",
        help=(
            "Only summarize existing safety "
            "audit CSV files."
        ),
    )

    args = parser.parse_args()

    if args.episodes < 1:
        raise ValueError(
            "--episodes must be >= 1"
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Safety-only mode
    # --------------------------------------------------------

    if args.audit_only:

        result = safety_audit_summary()

        print(
            json.dumps(
                result,
                indent=2,
            )
        )

        return

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("=" * 78)
    print(
        "MAI-07 LEVEL 1 RESEARCH EVALUATION"
    )
    print("=" * 78)

    print(
        f"Model: {MODEL_PATH}"
    )

    model = load_model()

    print(
        "Model validation: PASS"
    )

    print(
        "Observation: 120-D"
    )

    print(
        "Action: MultiDiscrete([2,2,2,2])"
    )

    # --------------------------------------------------------
    # Inference cost
    # --------------------------------------------------------

    benchmark_observation = (
        np.zeros(
            120,
            dtype=np.float32,
        )
    )

    inference = benchmark_inference(
        model,
        benchmark_observation,
    )

    print()
    print(
        "PPO INFERENCE COMPUTATION COST"
    )
    print(
        json.dumps(
            inference,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # Normal scenarios
    # --------------------------------------------------------

    all_rows = []

    for scenario in SCENARIOS:

        for episode in range(
            1,
            args.episodes + 1,
        ):

            result = run_episode(
                model=model,
                scenario=scenario,
                episode_number=episode,
            )

            all_rows.append(
                result
            )

            print(
                f"Reward={result['reward']:.2f} | "
                f"Wait={result['waiting_time']:.2f} | "
                f"Queue={result['queue_length']:.2f} | "
                f"Queue imbalance="
                f"{result['queue_imbalance']:.2f} | "
                f"Speed={result['average_speed']:.2f} | "
                f"Throughput={result['throughput']}"
            )

    # --------------------------------------------------------
    # Transfer test
    # --------------------------------------------------------

    transfer_metadata = None

    if args.transfer:

        transfer_metadata = (
            make_transfer_scenario()
        )

        print()
        print(
            "TRANSFER / DISTRIBUTION-SHIFT TEST"
        )

        transfer_rows = run_transfer(
            model=model,
            episodes=args.episodes,
        )

        all_rows.extend(
            transfer_rows
        )

        for result in transfer_rows:
            print(
                f"Transfer Episode "
                f"{result['episode']}: "
                f"Reward={result['reward']:.2f} | "
                f"Wait={result['waiting_time']:.2f} | "
                f"Queue={result['queue_length']:.2f} | "
                f"Queue imbalance="
                f"{result['queue_imbalance']:.2f} | "
                f"Speed={result['average_speed']:.2f} | "
                f"Throughput={result['throughput']}"
            )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    import pandas as pd

    raw_df = pd.DataFrame(
        all_rows
    )

    raw_path = (
        RESULTS_DIR
        / "level1_raw.csv"
    )

    raw_df.to_csv(
        raw_path,
        index=False,
    )

    summary_df = summarize(
        all_rows
    )

    summary_path = (
        RESULTS_DIR
        / "level1_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    safety = (
        safety_audit_summary()
    )

    overall = {
        "project":
            "MAI-07 | Multi-Agent Reinforcement Learning for Network-Wide Adaptive Traffic Control",
        "model":
            str(MODEL_PATH),
        "model_training_timesteps":
            500000,
        "observation_dimension":
            120,
        "action_space":
            "[2,2,2,2]",
        "scenarios":
            SCENARIOS,
        "episodes_per_scenario":
            args.episodes,
        "queue_balance_definition":
            "mean across four agents of "
            "(maximum directional queue - "
            "minimum directional queue)",
        "uncertainty_method":
            "mean +/- 1.96 * standard_error",
        "transfer_test":
            transfer_metadata,
        "inference_latency":
            inference,
        "safety_audit":
            safety,
    }

    overall_path = (
        RESULTS_DIR
        / "level1_overall.json"
    )

    with open(
        overall_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            overall,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Final console output
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print(
        "LEVEL 1 COMPLETE"
    )
    print("=" * 78)

    print()
    print(
        "Raw results:"
    )
    print(raw_path)

    print()
    print(
        "Statistical summary:"
    )
    print(summary_path)

    print()
    print(
        "Overall evidence:"
    )
    print(overall_path)

    print()
    print(
        "Safety audit:"
    )
    print(
        json.dumps(
            safety,
            indent=2,
        )
    )

    print()
    print(
        "IMPORTANT: "
        "No PPO retraining was performed."
    )


if __name__ == "__main__":
    main()
