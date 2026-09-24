"""
=========================================================
Traffic MARL Project

Module:
MARL Component Validation

Purpose:
Validate the complete MARL architecture before PPO
training.

Tests:

1. Environment creation
2. MARL controller creation
3. Four traffic-light agents
4. Neighbour graph
5. Local observations
6. Graph observations
7. Global observation
8. Action synchronization
9. Reward synchronization
10. Shared reward
11. Network statistics

IMPORTANT:
No PPO training is performed.

=========================================================
"""

from __future__ import annotations

import numpy as np

from rl.environments.config import EnvironmentConfig
from rl.environments.traffic_env import TrafficEnvironment
from rl.controllers.marl_controller import MARLController


# =========================================================
# EXPECTED PROJECT STRUCTURE
# =========================================================

AGENTS = list(
    EnvironmentConfig.TRAFFIC_LIGHT_IDS
)

LOCAL_SIZE = int(
    EnvironmentConfig.OBSERVATION_SIZE
)

EXPECTED_GRAPH_SIZE = (
    LOCAL_SIZE * 2
)

EXPECTED_GLOBAL_SIZE = (
    EXPECTED_GRAPH_SIZE
    * EnvironmentConfig.NUMBER_OF_AGENTS
)


# =========================================================
# PRINT HEADER
# =========================================================

def print_header():

    print()
    print("=" * 80)
    print("TRAFFIC MARL COMPONENT VALIDATION")
    print("=" * 80)

    print()

    print(
        "Project:",
        EnvironmentConfig.PROJECT_NAME
    )

    print(
        "Version:",
        EnvironmentConfig.VERSION
    )

    print()

    print(
        "Agents:",
        AGENTS
    )

    print(
        "Number of Agents:",
        EnvironmentConfig.NUMBER_OF_AGENTS
    )

    print(
        "Configured Local Observation:",
        LOCAL_SIZE
    )

    print(
        "Expected Graph Observation:",
        EXPECTED_GRAPH_SIZE
    )

    print(
        "Expected Global Observation:",
        EXPECTED_GLOBAL_SIZE
    )

    print(
        "Actions:",
        EnvironmentConfig.ACTIONS
    )

    print()


# =========================================================
# TEST AGENT CREATION
# =========================================================

def test_agents(
    marl: MARLController
):

    print("-" * 80)
    print("1. AGENT VALIDATION")
    print("-" * 80)

    actual_agents = list(
        marl.get_agents().keys()
    )

    print(
        "Configured agents:",
        actual_agents
    )

    if actual_agents != AGENTS:

        raise AssertionError(
            "Agent IDs do not match "
            "EnvironmentConfig.TRAFFIC_LIGHT_IDS."
        )

    if len(actual_agents) != 4:

        raise AssertionError(
            f"Expected 4 agents, "
            f"found {len(actual_agents)}."
        )

    for agent_id in AGENTS:

        agent = marl.get_agent(
            agent_id
        )

        print(
            f"{agent_id}: "
            f"neighbours={agent.get_neighbours()}"
        )

    print(
        "[PASS] Four MARL agents created."
    )

    print()


# =========================================================
# TEST NEIGHBOUR GRAPH
# =========================================================

def test_neighbour_graph(
    marl: MARLController
):

    print("-" * 80)
    print("2. NEIGHBOUR GRAPH VALIDATION")
    print("-" * 80)

    for agent_id in AGENTS:

        expected = set(
            EnvironmentConfig
            .NEIGHBOURS
            .get(
                agent_id,
                []
            )
        )

        actual = set(
            marl.get_agent(
                agent_id
            ).get_neighbours()
        )

        print(
            f"{agent_id}: "
            f"expected={sorted(expected)}, "
            f"actual={sorted(actual)}"
        )

        if expected != actual:

            raise AssertionError(
                f"Neighbour mismatch for "
                f"agent {agent_id}."
            )

        if agent_id in actual:

            raise AssertionError(
                f"Agent {agent_id} "
                f"cannot be its own neighbour."
            )

    print(
        "[PASS] Neighbour graph is consistent."
    )

    print()


# =========================================================
# TEST ENVIRONMENT RESET
# =========================================================

def test_environment_reset():

    print("-" * 80)
    print("3. ENVIRONMENT RESET")
    print("-" * 80)

    env = TrafficEnvironment(
        scenario_name="baseline"
    )

    try:

        observation, info = env.reset(
            seed=EnvironmentConfig.RANDOM_SEED
        )

        observation = np.asarray(
            observation,
            dtype=np.float32
        )

        print(
            "Scenario:",
            info.get(
                "scenario"
            )
        )

        print(
            "Observation shape:",
            observation.shape
        )

        print(
            "Observation space:",
            env.observation_space.shape
        )

        print(
            "Actual observation size:",
            observation.size
        )

        print(
            "Expected global size:",
            EXPECTED_GLOBAL_SIZE
        )

        # -------------------------------------------------
        # IMPORTANT DIAGNOSTIC
        # -------------------------------------------------

        if observation.size != EXPECTED_GLOBAL_SIZE:

            print()
            print(
                "[WARNING] Observation dimension "
                "does not match the MARL design."
            )

            print(
                f"Expected: {EXPECTED_GLOBAL_SIZE}"
            )

            print(
                f"Received: {observation.size}"
            )

            print()
            print(
                "DO NOT START PPO TRAINING YET."
            )

            return False

        print(
            "[PASS] Environment observation "
            "dimension is correct."
        )

        return True

    finally:

        env.close()


# =========================================================
# TEST MARL STATE SYNCHRONIZATION
# =========================================================

def test_state_synchronization(
    marl: MARLController
):

    print("-" * 80)
    print("4. STATE SYNCHRONIZATION")
    print("-" * 80)

    observations = {}

    for index, agent_id in enumerate(
        AGENTS
    ):

        values = np.full(
            LOCAL_SIZE,
            float(index + 1),
            dtype=np.float32
        )

        observations[
            agent_id
        ] = values

    marl.update_states(
        observations
    )

    states = marl.get_states()

    for agent_id in AGENTS:

        state = states[
            agent_id
        ]

        if state is None:

            raise AssertionError(
                f"State for {agent_id} "
                "is None."
            )

        state = np.asarray(
            state
        )

        if state.size != LOCAL_SIZE:

            raise AssertionError(
                f"State dimension mismatch "
                f"for {agent_id}: "
                f"{state.size}"
            )

        print(
            f"{agent_id}: "
            f"state_shape={state.shape}"
        )

    print(
        "[PASS] Local states synchronized."
    )

    print()


# =========================================================
# TEST GRAPH OBSERVATIONS
# =========================================================

def test_graph_observations(
    marl: MARLController
):

    print("-" * 80)
    print("5. GRAPH OBSERVATION VALIDATION")
    print("-" * 80)

    graph_observations = (
        marl.graph_observations()
    )

    for agent_id in AGENTS:

        observation = np.asarray(
            graph_observations[
                agent_id
            ],
            dtype=np.float32
        ).flatten()

        print(
            f"{agent_id}: "
            f"graph_shape={observation.shape}"
        )

        if observation.size != EXPECTED_GRAPH_SIZE:

            raise AssertionError(
                f"Graph observation mismatch "
                f"for {agent_id}. "
                f"Expected "
                f"{EXPECTED_GRAPH_SIZE}, "
                f"received "
                f"{observation.size}."
            )

    print(
        "[PASS] Graph observations are "
        "correctly constructed."
    )

    print()


# =========================================================
# TEST GLOBAL STATE
# =========================================================

def test_global_state(
    marl: MARLController
):

    print("-" * 80)
    print("6. GLOBAL STATE VALIDATION")
    print("-" * 80)

    global_state = (
        marl.global_state()
    )

    global_state = np.asarray(
        global_state,
        dtype=np.float32
    ).flatten()

    print(
        "Global state shape:",
        global_state.shape
    )

    print(
        "Expected from current states:",
        LOCAL_SIZE
        * EnvironmentConfig.NUMBER_OF_AGENTS
    )

    # -----------------------------------------------------
    # Current global_state() uses the stored agent states.
    # At this point those are local 15-dimensional states.
    # -----------------------------------------------------

    expected_state_size = (
        LOCAL_SIZE
        * EnvironmentConfig.NUMBER_OF_AGENTS
    )

    if global_state.size != expected_state_size:

        raise AssertionError(
            "MARLController.global_state() "
            "dimension mismatch. "
            f"Expected "
            f"{expected_state_size}, "
            f"received "
            f"{global_state.size}."
        )

    print(
        "[PASS] Global local-state "
        "concatenation works."
    )

    print()


# =========================================================
# TEST ACTION SYNCHRONIZATION
# =========================================================

def test_actions(
    marl: MARLController
):

    print("-" * 80)
    print("7. ACTION SYNCHRONIZATION")
    print("-" * 80)

    actions = {

        "A":
            EnvironmentConfig.ACTION_NORTH_SOUTH,

        "B":
            EnvironmentConfig.ACTION_EAST_WEST,

        "C":
            EnvironmentConfig.ACTION_NORTH_SOUTH,

        "D":
            EnvironmentConfig.ACTION_EAST_WEST,

    }

    marl.apply_actions(
        actions
    )

    stored = (
        marl.get_actions()
    )

    print(
        "Requested:",
        actions
    )

    print(
        "Stored:",
        stored
    )

    if stored != actions:

        raise AssertionError(
            "Action synchronization failed."
        )

    print(
        "[PASS] Actions synchronized."
    )

    print()


# =========================================================
# TEST REWARD SYNCHRONIZATION
# =========================================================

def test_rewards(
    marl: MARLController
):

    print("-" * 80)
    print("8. REWARD SYNCHRONIZATION")
    print("-" * 80)

    rewards = {

        "A": 10.0,

        "B": 20.0,

        "C": 30.0,

        "D": 40.0,

    }

    marl.update_rewards(
        rewards
    )

    stored = (
        marl.get_rewards()
    )

    print(
        "Requested:",
        rewards
    )

    print(
        "Stored:",
        stored
    )

    for agent_id in AGENTS:

        if not np.isclose(
            stored[agent_id],
            rewards[agent_id]
        ):

            raise AssertionError(
                f"Reward mismatch "
                f"for {agent_id}."
            )

    print(
        "[PASS] Rewards synchronized."
    )

    print()


# =========================================================
# TEST SHARED REWARDS
# =========================================================

def test_shared_rewards(
    marl: MARLController
):

    print("-" * 80)
    print("9. SHARED REWARD VALIDATION")
    print("-" * 80)

    shared = (
        marl.get_shared_rewards()
    )

    for agent_id in AGENTS:

        value = float(
            shared[agent_id]
        )

        print(
            f"{agent_id}: "
            f"shared_reward={value:.4f}"
        )

        if not np.isfinite(
            value
        ):

            raise AssertionError(
                f"Invalid shared reward "
                f"for {agent_id}."
            )

    print(
        "[PASS] Shared rewards are valid."
    )

    print()


# =========================================================
# TEST NETWORK STATISTICS
# =========================================================

def test_network_statistics(
    marl: MARLController
):

    print("-" * 80)
    print("10. NETWORK STATISTICS")
    print("-" * 80)

    statistics = (
        marl.network_statistics()
    )

    print(
        "Network statistics:",
        statistics
    )

    required = [

        "waiting_time",

        "queue_length",

        "average_speed",

    ]

    for key in required:

        if key not in statistics:

            raise AssertionError(
                f"Missing network statistic: "
                f"{key}"
            )

        if not np.isfinite(
            float(
                statistics[key]
            )
        ):

            raise AssertionError(
                f"Invalid network statistic: "
                f"{key}"
            )

    print(
        "[PASS] Network statistics valid."
    )

    print()


# =========================================================
# MAIN
# =========================================================

def main():

    print_header()

    # -----------------------------------------------------
    # Environment dimension check
    # -----------------------------------------------------

    environment_ok = (
        test_environment_reset()
    )

    if not environment_ok:

        print()
        print("=" * 80)
        print(
            "[STOP] Observation architecture "
            "must be fixed before PPO training."
        )
        print("=" * 80)
        print()

        return False

    # -----------------------------------------------------
    # Create controller
    # -----------------------------------------------------

    marl = MARLController()

    try:

        test_agents(
            marl
        )

        test_neighbour_graph(
            marl
        )

        test_state_synchronization(
            marl
        )

        test_graph_observations(
            marl
        )

        test_global_state(
            marl
        )

        test_actions(
            marl
        )

        test_rewards(
            marl
        )

        test_shared_rewards(
            marl
        )

        test_network_statistics(
            marl
        )

    finally:

        marl.terminate()

    print()
    print("=" * 80)
    print(
        "[PASS] MARL COMPONENT VALIDATION COMPLETE"
    )
    print("=" * 80)

    print()
    print(
        "The MARL components are ready for "
        "the PPO smoke test."
    )

    print()

    return True


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    success = main()

    if not success:

        raise SystemExit(
            "MARL component validation failed."
        )