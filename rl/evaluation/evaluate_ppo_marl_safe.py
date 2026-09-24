"""
PPO-MARL SAFETY + GOVERNANCE EVALUATION

PROJECT:
    MAI-07 | Multi-Agent Reinforcement Learning
    for Network-Wide Adaptive Traffic Control

ACTUAL PROJECT APIs:

TrafficEnvironment:
    TrafficEnvironment(
        scenario_name="baseline",
        render_mode=None
    )

HumanApprovalGate:
    process(
        safety_decision: SafetyDecision,
        human_approved: bool = True
    ) -> ApprovalDecision

Evaluation flow:

    PPO-MARL
        |
        v
    Proposed Actions
        |
        v
    Safety Validator
        |
        v
    SafetyDecision
        |
        v
    Human Approval Gate
        |
        v
    ApprovalDecision
        |
        v
    Final Actions
        |
        v
    SUMO / TrafficEnvironment
        |
        v
    Metrics + Audit

IMPORTANT:
    - Does NOT modify PPO training.
    - Uses completed 120-D PPO-MARL model.
    - Uses actual TrafficEnvironment constructor.
    - Uses the actual SafetyDecision object with HumanApprovalGate.
    - Human governance is evaluated in automatic mode.
    - Every governance decision is written to the audit log.
"""

# ==============================================================
# IMPORTS
# ==============================================================

import os
import sys
import csv
import json
from datetime import datetime, timezone

import numpy as np
from stable_baselines3 import PPO


# ==============================================================
# PROJECT ROOT
# ==============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ==============================================================
# PROJECT IMPORTS
# ==============================================================

from rl.environments.traffic_env import TrafficEnvironment
from rl.safety.safety_validator import SafetyValidator
from rl.safety.human_approval import HumanApprovalGate
from rl.safety.audit_logger import AuditLogger


# ==============================================================
# CONFIGURATION
# ==============================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "rl",
    "checkpoints",
    "ppo_marl_120d_final.zip",
)


RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "rl",
    "results",
    "ppo_marl_safe",
)


AUDIT_DIR = os.path.join(
    PROJECT_ROOT,
    "rl",
    "logs",
    "safety",
)


RAW_FILE = os.path.join(
    RESULTS_DIR,
    "ppo_marl_safe_raw.csv",
)


SUMMARY_FILE = os.path.join(
    RESULTS_DIR,
    "ppo_marl_safe_summary.csv",
)


OVERALL_FILE = os.path.join(
    RESULTS_DIR,
    "ppo_marl_safe_overall.json",
)


AUDIT_FILE = os.path.join(
    AUDIT_DIR,
    "ppo_marl_safe_evaluation_audit.csv",
)


# ==============================================================
# SCENARIOS
# ==============================================================

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


# ==============================================================
# INTERSECTIONS
# ==============================================================

INTERSECTIONS = [
    "A",
    "B",
    "C",
    "D",
]


# ==============================================================
# EVALUATION SETTINGS
# ==============================================================

EPISODES_PER_SCENARIO = 3

MAX_STEPS = 300

BASE_SEED = 42


# ==============================================================
# CREATE DIRECTORIES
# ==============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True,
)

os.makedirs(
    AUDIT_DIR,
    exist_ok=True,
)


# ==============================================================
# BASIC HELPERS
# ==============================================================

def utc_now():
    """
    Return current UTC timestamp.
    """

    return datetime.now(
        timezone.utc
    ).isoformat()


# ==============================================================
# SAFE FLOAT
# ==============================================================

def to_float(
    value,
    default=0.0,
):
    """
    Safely convert value to float.
    """

    try:
        return float(value)

    except Exception:
        return default


# ==============================================================
# SAFE INTEGER
# ==============================================================

def to_int(
    value,
    default=0,
):
    """
    Safely convert value to integer.
    """

    try:
        return int(value)

    except Exception:
        return default


# ==============================================================
# NORMALIZE PPO ACTION
# ==============================================================

def normalize_action(
    action,
):
    """
    Convert PPO MultiDiscrete output into
    exactly four binary actions.

    Expected action:

        [A, B, C, D]

    Each action:

        0 or 1
    """

    arr = np.asarray(
        action
    ).reshape(-1)

    if arr.size != 4:

        raise ValueError(
            "PPO action dimension error. "
            f"Expected 4 actions, got {arr.size}."
        )

    actions = []

    for value in arr:

        action_value = int(
            np.clip(
                round(
                    float(value)
                ),
                0,
                1,
            )
        )

        actions.append(
            action_value
        )

    return actions


# ==============================================================
# VALIDATE OBSERVATION
# ==============================================================

def validate_observation(
    observation,
):
    """
    Validate the current 120-D
    graph-aware observation.
    """

    observation = np.asarray(
        observation,
        dtype=np.float32,
    )

    if observation.shape != (120,):

        raise ValueError(
            "PPO-MARL observation dimension error. "
            f"Expected (120,), got {observation.shape}."
        )

    if not np.all(
        np.isfinite(
            observation
        )
    ):

        raise ValueError(
            "Observation contains NaN or infinite values."
        )

    return observation


# ==============================================================
# BUILD SAFETY STATES
# ==============================================================

def build_safety_states(
    observation,
):
    """
    Convert the 120-D global observation
    into safety state information for
    intersections A, B, C and D.

    Current architecture:

        A = 30 values
        B = 30 values
        C = 30 values
        D = 30 values

    Each 30-D block contains:

        0  vehicle count
        1  average speed
        2  waiting time
        3  stopped vehicles
        4  phase
        5  north queue
        6  east queue
        7  south queue
        8  west queue
        9  lane density
        10 green timer
        11 yellow timer
        12 current action
        13 in yellow
        14 reserved

        Remaining 15 values:
            neighbour / graph information
    """

    observation = validate_observation(
        observation
    )

    states = {}

    for index, intersection in enumerate(
        INTERSECTIONS
    ):

        start = (
            index * 30
        )

        local = observation[
            start:start + 15
        ]

        states[
            intersection
        ] = {

            "green_timer":
                to_float(
                    local[10]
                ),

            "in_yellow":
                bool(
                    local[13] > 0.5
                ),
        }

    return states


# ==============================================================
# ENVIRONMENT METRICS
# ==============================================================

def get_environment_metrics(
    env,
):
    """
    Read traffic metrics from the environment.

    If get_metrics() is not available,
    safe zero defaults are returned.
    """

    metrics = {

        "waiting_time":
            0.0,

        "queue_length":
            0.0,

        "average_speed":
            0.0,
    }


    if not hasattr(
        env,
        "get_metrics",
    ):

        return metrics


    try:

        data = env.get_metrics()

    except Exception:

        return metrics


    if not isinstance(
        data,
        dict,
    ):

        return metrics


    for key in metrics:

        if key in data:

            metrics[key] = to_float(
                data[key]
            )


    return metrics


# ==============================================================
# PROCESS HUMAN APPROVAL
# ==============================================================

def process_approval(
    approval_gate,
    safety_decision,
):
    """
    IMPORTANT:

    This uses the ACTUAL project API:

        process(
            safety_decision,
            human_approved=True
        )

    The previous evaluator was incorrect because
    it passed individual integer values.

    The real HumanApprovalGate expects the
    complete SafetyDecision object.
    """

    if not hasattr(
        approval_gate,
        "process",
    ):

        raise RuntimeError(
            "HumanApprovalGate does not contain "
            "process()."
        )


    approval = approval_gate.process(
        safety_decision,
        human_approved=True,
    )


    if approval is None:

        raise RuntimeError(
            "HumanApprovalGate.process() "
            "returned None."
        )


    return approval


# ==============================================================
# EXTRACT FINAL ACTION
# ==============================================================

def extract_final_action(
    approval,
    fallback_action,
):
    """
    Extract final_action from ApprovalDecision.
    """

    if approval is None:

        return int(
            fallback_action
        )


    # ----------------------------------------------------------
    # Dictionary support
    # ----------------------------------------------------------

    if isinstance(
        approval,
        dict,
    ):

        if "final_action" in approval:

            return to_int(
                approval[
                    "final_action"
                ],
                fallback_action,
            )


        if "approved_action" in approval:

            return to_int(
                approval[
                    "approved_action"
                ],
                fallback_action,
            )


        if "action" in approval:

            return to_int(
                approval[
                    "action"
                ],
                fallback_action,
            )


    # ----------------------------------------------------------
    # Object support
    # ----------------------------------------------------------

    if hasattr(
        approval,
        "final_action",
    ):

        return to_int(
            approval.final_action,
            fallback_action,
        )


    if hasattr(
        approval,
        "approved_action",
    ):

        return to_int(
            approval.approved_action,
            fallback_action,
        )


    if hasattr(
        approval,
        "action",
    ):

        return to_int(
            approval.action,
            fallback_action,
        )


    # ----------------------------------------------------------
    # Integer support
    # ----------------------------------------------------------

    if isinstance(
        approval,
        (
            int,
            np.integer,
        ),
    ):

        return int(
            approval
        )


    return int(
        fallback_action
    )


# ==============================================================
# APPROVAL VALUE HELPER
# ==============================================================

def get_approval_value(
    approval,
    names,
    default,
):
    """
    Read a field from ApprovalDecision.
    """

    # ----------------------------------------------------------
    # Dictionary
    # ----------------------------------------------------------

    if isinstance(
        approval,
        dict,
    ):

        for name in names:

            if name in approval:

                return approval[name]


        return default


    # ----------------------------------------------------------
    # Object
    # ----------------------------------------------------------

    for name in names:

        if hasattr(
            approval,
            name,
        ):

            return getattr(
                approval,
                name,
            )


    return default


# ==============================================================
# WRITE AUDIT
# ==============================================================
# ==============================================================
# WRITE AUDIT
# ==============================================================

def write_audit(
    audit_logger,
    approval,
):
    """
    Write the actual ApprovalDecision object
    through the project's AuditLogger API.

    ACTUAL API confirmed from runtime traceback:

        AuditLogger.log(approval_decision)

    AuditLogger.log() internally calls:

        approval_decision.to_dict()

    Therefore we MUST pass the ApprovalDecision
    object directly.

    DO NOT convert it to a dictionary here.
    """

    if approval is None:

        raise RuntimeError(
            "HumanApprovalGate returned None. "
            "Cannot write audit record."
        )

    if not hasattr(
        approval,
        "to_dict",
    ):

        raise RuntimeError(
            "HumanApprovalGate did not return the expected "
            "ApprovalDecision object. "
            "The returned object does not have to_dict(). "
            f"Detected type: {type(approval)}"
        )

    audit_logger.log(
        approval
    )

    
# ==============================================================
# SINGLE EPISODE
# ==============================================================

def run_episode(
    model,
    scenario,
    episode_number,
    safety_validator,
    approval_gate,
    audit_logger,
):
    """
    Run one complete PPO-MARL episode.
    """

    seed = (
        BASE_SEED
        + episode_number
        - 1
    )


    print()
    print(
        f"Starting SUMO scenario: "
        f"{scenario}"
    )


    # ==========================================================
    # ACTUAL TRAFFIC ENVIRONMENT API
    # ==========================================================

    env = TrafficEnvironment(
        scenario_name=scenario
    )


    try:

        # ======================================================
        # RESET
        # ======================================================

        reset_result = env.reset(
            seed=seed
        )


        if not isinstance(
            reset_result,
            tuple,
        ):

            raise RuntimeError(
                "TrafficEnvironment.reset() "
                "did not return a tuple."
            )


        if len(
            reset_result
        ) != 2:

            raise RuntimeError(
                "TrafficEnvironment.reset() "
                "must return (observation, info)."
            )


        observation, info = (
            reset_result
        )


        observation = validate_observation(
            observation
        )


        # ======================================================
        # METRIC STORAGE
        # ======================================================

        total_reward = 0.0

        waiting_values = []

        queue_values = []

        speed_values = []


        throughput = 0


        safety_overrides = 0

        safety_violations = 0


        steps = 0


        # ======================================================
        # EPISODE LOOP
        # ======================================================

        for step in range(
            MAX_STEPS
        ):

            steps += 1


            # ==================================================
            # 1. PPO ACTION
            # ==================================================

            proposed_action, _ = model.predict(
                observation,
                deterministic=True,
            )


            proposed_actions = normalize_action(
                proposed_action
            )


            # ==================================================
            # 2. SAFETY STATE
            # ==================================================

            agent_states = build_safety_states(
                observation
            )


            # ==================================================
            # 3. SAFETY VALIDATION
            # ==================================================

            decisions = (
                safety_validator.validate_network_actions(
                    proposed_actions,
                    agent_states,
                )
            )


            if not isinstance(
                decisions,
                dict,
            ):

                raise RuntimeError(
                    "SafetyValidator.validate_network_actions() "
                    "must return a dictionary of SafetyDecision objects."
                )


            final_actions = []


            # ==================================================
            # 4. EACH INTERSECTION
            # ==================================================

            for intersection in INTERSECTIONS:

                if intersection not in decisions:

                    raise RuntimeError(
                        "Safety validator did not return "
                        f"a decision for intersection "
                        f"{intersection}."
                    )


                decision = decisions[
                    intersection
                ]


                # =================================================
                # SAFETY VALIDATION OBJECT CHECK
                # =================================================

                required_fields = [
                    "proposed_action",
                    "approved_action",
                    "valid_action",
                    "overridden",
                ]


                for field in required_fields:

                    if not hasattr(
                        decision,
                        field,
                    ):

                        raise RuntimeError(
                            "SafetyValidator returned an invalid "
                            "SafetyDecision object. Missing field: "
                            f"{field}"
                        )


                # =================================================
                # SAFETY COUNTERS
                # =================================================

                if decision.overridden:

                    safety_overrides += 1


                if not decision.valid_action:

                    safety_violations += 1


                # =================================================
                # 5. HUMAN GOVERNANCE
                # =================================================
                #
                # CRITICAL:
                #
                # HumanApprovalGate.process() expects:
                #
                #     SafetyDecision
                #
                # NOT:
                #
                #     integer action
                #
                # =================================================

                approval = process_approval(
                    approval_gate,
                    decision,
                )


                # =================================================
                # CHECK APPROVAL RESULT
                # =================================================

                if not hasattr(
                    approval,
                    "final_action",
                ):

                    raise RuntimeError(
                        "HumanApprovalGate returned an object "
                        "without final_action."
                    )


                # =================================================
                # 6. FINAL ACTION
                # =================================================

                final_action = extract_final_action(
                    approval,
                    decision.approved_action,
                )


                final_action = int(
                    np.clip(
                        final_action,
                        0,
                        1,
                    )
                )


                final_actions.append(
                    final_action
                )


                # =================================================
                # 7. AUDIT
                # =================================================

                write_audit(
                    audit_logger,
                    approval,
                )


            # ==================================================
            # FINAL NETWORK ACTION VALIDATION
            # ==================================================

            final_actions = np.asarray(
                final_actions,
                dtype=np.int64,
            )


            if final_actions.shape != (
                4,
            ):

                raise ValueError(
                    "Final network action dimension error. "
                    "Expected shape (4), "
                    f"got {final_actions.shape}."
                )


            if not np.all(
                np.isin(
                    final_actions,
                    [0, 1],
                )
            ):

                raise ValueError(
                    "Final network action contains "
                    "values outside {0, 1}."
                )


            # ==================================================
            # 8. ENVIRONMENT STEP
            # ==================================================

            step_result = env.step(
                final_actions
            )


            if not isinstance(
                step_result,
                tuple,
            ):

                raise RuntimeError(
                    "TrafficEnvironment.step() "
                    "did not return a tuple."
                )


            if len(
                step_result
            ) != 5:

                raise RuntimeError(
                    "TrafficEnvironment.step() "
                    "must return "
                    "(observation, reward, terminated, "
                    "truncated, info)."
                )


            (
                next_observation,
                reward,
                terminated,
                truncated,
                info,
            ) = step_result


            observation = validate_observation(
                next_observation
            )


            total_reward += to_float(
                reward
            )


            # ==================================================
            # 9. TRAFFIC METRICS
            # ==================================================

            metrics = get_environment_metrics(
                env
            )


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


            # ==================================================
            # 10. THROUGHPUT
            # ==================================================

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


            # ==================================================
            # 11. TERMINATION
            # ==================================================

            if (
                terminated
                or truncated
            ):

                break


        # ======================================================
        # EPISODE RESULT
        # ======================================================

        result = {

            "scenario":
                scenario,

            "episode":
                episode_number,

            "seed":
                seed,

            "reward":
                total_reward,

            "waiting_time":
                (
                    float(
                        np.mean(
                            waiting_values
                        )
                    )
                    if waiting_values
                    else 0.0
                ),

            "queue_length":
                (
                    float(
                        np.mean(
                            queue_values
                        )
                    )
                    if queue_values
                    else 0.0
                ),

            "average_speed":
                (
                    float(
                        np.mean(
                            speed_values
                        )
                    )
                    if speed_values
                    else 0.0
                ),

            "throughput":
                throughput,

            "episode_length":
                steps,

            "safety_overrides":
                safety_overrides,

            "safety_violations":
                safety_violations,

            "timestamp":
                utc_now(),
        }


        return result


    finally:

        try:

            env.close()

        except Exception:

            pass


# ==============================================================
# SAVE RAW CSV
# ==============================================================

def save_raw_results(
    results,
):
    """
    Save episode-level results.
    """

    if not results:

        raise RuntimeError(
            "No results available for raw CSV."
        )


    fields = list(
        results[0].keys()
    )


    with open(
        RAW_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            results
        )


# ==============================================================
# CREATE SCENARIO SUMMARY
# ==============================================================

def create_summary(
    results,
):
    """
    Create scenario-level averages.
    """

    summary = []


    for scenario in SCENARIOS:

        rows = [
            row
            for row in results
            if row["scenario"] == scenario
        ]


        if not rows:

            continue


        def average(
            key,
        ):

            return float(
                np.mean(
                    [
                        row[key]
                        for row in rows
                    ]
                )
            )


        summary.append(
            {

                "scenario":
                    scenario,

                "reward":
                    average(
                        "reward"
                    ),

                "waiting_time":
                    average(
                        "waiting_time"
                    ),

                "queue_length":
                    average(
                        "queue_length"
                    ),

                "average_speed":
                    average(
                        "average_speed"
                    ),

                "throughput":
                    average(
                        "throughput"
                    ),

                "episode_length":
                    average(
                        "episode_length"
                    ),

                "safety_overrides":
                    average(
                        "safety_overrides"
                    ),

                "safety_violations":
                    average(
                        "safety_violations"
                    ),
            }
        )


    return summary


# ==============================================================
# SAVE SUMMARY CSV
# ==============================================================

def save_summary(
    summary,
):
    """
    Save scenario-level summary.
    """

    if not summary:

        raise RuntimeError(
            "No scenario summary available."
        )


    fields = list(
        summary[0].keys()
    )


    with open(
        SUMMARY_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            summary
        )


# ==============================================================
# CREATE OVERALL SUMMARY
# ==============================================================

def create_overall(
    results,
    total_overrides,
    total_violations,
):
    """
    Create overall evaluation JSON.
    """

    if not results:

        raise RuntimeError(
            "Cannot create overall summary "
            "without results."
        )


    def average(
        key,
    ):

        return float(
            np.mean(
                [
                    row[key]
                    for row in results
                ]
            )
        )


    return {

        "model":
            MODEL_PATH,

        "evaluation_type":
            (
                "PPO-MARL with safety "
                "and human governance"
            ),

        "observation_dimension":
            120,

        "number_of_agents":
            4,

        "intersections":
            INTERSECTIONS,

        "scenarios":
            SCENARIOS,

        "episodes_per_scenario":
            EPISODES_PER_SCENARIO,

        "maximum_steps":
            MAX_STEPS,

        "base_seed":
            BASE_SEED,

        "total_episodes":
            len(results),

        "mean_reward":
            average(
                "reward"
            ),

        "mean_waiting_time":
            average(
                "waiting_time"
            ),

        "mean_queue_length":
            average(
                "queue_length"
            ),

        "mean_average_speed":
            average(
                "average_speed"
            ),

        "mean_throughput":
            average(
                "throughput"
            ),

        "mean_episode_length":
            average(
                "episode_length"
            ),

        "total_safety_overrides":
            int(
                total_overrides
            ),

        "total_safety_violations":
            int(
                total_violations
            ),

        "generated_at":
            utc_now(),
    }


# ==============================================================
# SAVE OVERALL JSON
# ==============================================================

def save_overall(
    overall,
):
    """
    Save overall JSON.
    """

    with open(
        OVERALL_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            overall,
            file,
            indent=4,
        )


# ==============================================================
# PRINT EPISODE RESULT
# ==============================================================

def print_episode_result(
    result,
):
    """
    Print readable episode result.
    """

    print()

    print(
        f"Reward           : "
        f"{result['reward']:.2f}"
    )

    print(
        f"Waiting Time     : "
        f"{result['waiting_time']:.2f}"
    )

    print(
        f"Queue Length     : "
        f"{result['queue_length']:.2f}"
    )

    print(
        f"Average Speed    : "
        f"{result['average_speed']:.2f}"
    )

    print(
        f"Throughput       : "
        f"{result['throughput']}"
    )

    print(
        f"Episode Length   : "
        f"{result['episode_length']}"
    )

    print(
        f"Safety Overrides : "
        f"{result['safety_overrides']}"
    )

    print(
        f"Safety Violations: "
        f"{result['safety_violations']}"
    )


# ==============================================================
# MAIN EVALUATION
# ==============================================================

def evaluate():
    """
    Run complete PPO-MARL safety evaluation.
    """

    print()

    print(
        "=" * 78
    )

    print(
        "PPO-MARL SAFETY + GOVERNANCE EVALUATION"
    )

    print(
        "=" * 78
    )


    # ==========================================================
    # MODEL
    # ==========================================================

    print()

    print(
        "Model:"
    )

    print(
        MODEL_PATH
    )


    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            "Final PPO-MARL model not found:\n"
            f"{MODEL_PATH}"
        )


    # ==========================================================
    # LOAD MODEL
    # ==========================================================

    print()

    print(
        "Loading PPO-MARL model..."
    )


    model = PPO.load(
        MODEL_PATH,
        device="cpu",
    )


    print(
        "PPO-MARL model loaded."
    )


    # ==========================================================
    # MODEL SPACE CHECK
    # ==========================================================

    print()

    print(
        "Checking PPO model spaces..."
    )


    model_observation_shape = (
        model.observation_space.shape
    )


    if model_observation_shape != (
        120,
    ):

        raise ValueError(
            "Loaded PPO model does not use "
            "the expected 120-D observation space. "
            f"Detected: {model_observation_shape}"
        )


    model_action_nvec = (
        model.action_space.nvec
    )


    expected_action_space = np.array(
        [2, 2, 2, 2]
    )


    if not np.array_equal(
        model_action_nvec,
        expected_action_space,
    ):

        raise ValueError(
            "Loaded PPO model does not use "
            "the expected MultiDiscrete([2,2,2,2]) "
            "action space. "
            f"Detected: {model_action_nvec}"
        )


    print(
        "Observation space : "
        f"{model_observation_shape}"
    )

    print(
        "Action space      : "
        f"MultiDiscrete({list(model_action_nvec)})"
    )


    # ==========================================================
    # SAFETY VALIDATOR
    # ==========================================================

    safety_validator = SafetyValidator(
        min_green_steps=10,
        yellow_protection=True,
    )


    # ==========================================================
    # HUMAN APPROVAL GATE
    # ==========================================================

    approval_gate = HumanApprovalGate(
        mode="automatic"
    )


    # ==========================================================
    # VERIFY ACTUAL HUMAN APPROVAL API
    # ==========================================================

    approval_process = getattr(
        approval_gate,
        "process",
        None,
    )


    if approval_process is None:

        raise RuntimeError(
            "HumanApprovalGate.process() "
            "does not exist."
        )


    import inspect


    print()

    print(
        "HumanApprovalGate.process signature:"
    )

    print(
        inspect.signature(
            approval_process
        )
    )


    # ==========================================================
    # IMPORTANT API VALIDATION
    # ==========================================================

    parameters = list(
        inspect.signature(
            approval_process
        ).parameters.values()
    )


    if not parameters:

        raise RuntimeError(
            "HumanApprovalGate.process() "
            "has no parameters. "
            "Expected SafetyDecision parameter."
        )


    first_parameter = parameters[0]


    print()

    print(
        "Approval input parameter:"
    )

    print(
        f"    {first_parameter.name}"
    )


    # ==========================================================
    # AUDIT LOGGER
    # ==========================================================

    audit_logger = AuditLogger(
        AUDIT_FILE
    )


    # ==========================================================
    # RESULTS
    # ==========================================================

    results = []


    total_overrides = 0

    total_violations = 0


    # ==========================================================
    # SCENARIO LOOP
    # ==========================================================

    for scenario_number, scenario in enumerate(
        SCENARIOS,
        start=1,
    ):

        print()

        print(
            "=" * 78
        )

        print(
            f"SCENARIO "
            f"{scenario_number}/"
            f"{len(SCENARIOS)}: "
            f"{scenario}"
        )

        print(
            "=" * 78
        )


        # ======================================================
        # EPISODE LOOP
        # ======================================================

        for episode in range(
            1,
            EPISODES_PER_SCENARIO + 1,
        ):

            print()

            print(
                f"Episode "
                f"{episode}/"
                f"{EPISODES_PER_SCENARIO}"
            )


            result = run_episode(
                model=model,
                scenario=scenario,
                episode_number=episode,
                safety_validator=safety_validator,
                approval_gate=approval_gate,
                audit_logger=audit_logger,
            )


            results.append(
                result
            )


            total_overrides += (
                result[
                    "safety_overrides"
                ]
            )


            total_violations += (
                result[
                    "safety_violations"
                ]
            )


            print_episode_result(
                result
            )


    # ==========================================================
    # VALIDATE RESULT COUNT
    # ==========================================================

    expected_results = (
        len(SCENARIOS)
        * EPISODES_PER_SCENARIO
    )


    if len(results) != expected_results:

        raise RuntimeError(
            "Unexpected number of evaluation results. "
            f"Expected {expected_results}, "
            f"got {len(results)}."
        )


    # ==========================================================
    # SAVE RAW
    # ==========================================================

    save_raw_results(
        results
    )


    # ==========================================================
    # CREATE + SAVE SUMMARY
    # ==========================================================

    summary = create_summary(
        results
    )


    save_summary(
        summary
    )


    # ==========================================================
    # CREATE + SAVE OVERALL
    # ==========================================================

    overall = create_overall(
        results,
        total_overrides,
        total_violations,
    )


    save_overall(
        overall
    )


    # ==========================================================
    # FINAL OUTPUT
    # ==========================================================

    print()

    print(
        "=" * 78
    )

    print(
        "PPO-MARL SAFETY + GOVERNANCE "
        "EVALUATION COMPLETE"
    )

    print(
        "=" * 78
    )


    print()

    print(
        f"Total episodes       : "
        f"{len(results)}"
    )


    print(
        f"Safety overrides     : "
        f"{total_overrides}"
    )


    print(
        f"Safety violations    : "
        f"{total_violations}"
    )


    print()

    print(
        "Mean Reward          : "
        f"{overall['mean_reward']:.2f}"
    )


    print(
        "Mean Waiting Time    : "
        f"{overall['mean_waiting_time']:.2f}"
    )


    print(
        "Mean Queue Length    : "
        f"{overall['mean_queue_length']:.2f}"
    )


    print(
        "Mean Average Speed   : "
        f"{overall['mean_average_speed']:.2f}"
    )


    print(
        "Mean Throughput      : "
        f"{overall['mean_throughput']:.2f}"
    )


    print()

    print(
        "Generated files:"
    )


    print()

    print(
        "RAW:"
    )

    print(
        RAW_FILE
    )


    print()

    print(
        "SUMMARY:"
    )

    print(
        SUMMARY_FILE
    )


    print()

    print(
        "OVERALL:"
    )

    print(
        OVERALL_FILE
    )


    print()

    print(
        "AUDIT:"
    )

    print(
        AUDIT_FILE
    )


    print()

    print(
        "=" * 78
    )


# ==============================================================
# ENTRY POINT
# ==============================================================

if __name__ == "__main__":

    evaluate()