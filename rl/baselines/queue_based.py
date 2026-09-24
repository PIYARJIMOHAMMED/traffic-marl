"""
=========================================================
Traffic MARL Project

Module:
Queue-Based Adaptive Baseline Policy

Description:
Non-RL adaptive traffic signal controller.

The policy examines the queue lengths at an
intersection and selects an action based on
the largest directional queue.

Purpose:

    Provide a meaningful traditional adaptive
    traffic-control baseline for comparison
    against Multi-Agent PPO.

Decision principle:

    largest queue
          ↓
    corresponding action
          ↓
    traffic signal

This policy does NOT learn.

Week:
6

Task:
6.1

Author:
Mohammed Piyarji
=========================================================
"""

from typing import Dict, Iterable, Optional

import numpy as np

from rl.environments.config import EnvironmentConfig


class QueueBasedPolicy:
    """
    Adaptive queue-based traffic signal policy.

    The policy uses directional queue information
    from each traffic-light observation.

    Expected local observation layout:

        0 = vehicle count
        1 = average speed
        2 = waiting time
        3 = stopped vehicles
        4 = traffic-light phase
        5 = north queue
        6 = east queue
        7 = south queue
        8 = west queue

    Neighbour features may follow these values.

    The policy therefore only uses the first
    directional queue values.
    """

    def __init__(
        self,
        switch_threshold: float = 0.0,
        seed: Optional[int] = 42,
    ):
        """
        Parameters
        ----------
        switch_threshold:
            Minimum queue advantage required before
            switching direction.

        seed:
            Seed used when tie-breaking is required.
        """

        self.switch_threshold = float(
            switch_threshold
        )

        self.seed = seed

        self.rng = np.random.default_rng(
            seed
        )

        self.step_count = 0

    # =====================================================
    # RESET
    # =====================================================

    def reset(
        self,
        seed: Optional[int] = None,
    ) -> None:
        """
        Reset runtime state.
        """

        if seed is not None:

            self.seed = seed

            self.rng = np.random.default_rng(
                seed
            )

        self.step_count = 0

    # =====================================================
    # EXTRACT QUEUES
    # =====================================================

    @staticmethod
    def extract_directional_queues(
        observation: Iterable[float],
    ) -> np.ndarray:
        """
        Extract north/east/south/west queue lengths.

        Parameters
        ----------
        observation:
            Local or graph observation.

        Returns
        -------
        np.ndarray
            Four directional queue values.
        """

        observation = np.asarray(
            observation,
            dtype=np.float32
        ).flatten()

        if observation.shape[0] < 9:

            raise ValueError(
                "Observation must contain at least "
                "9 values to use QueueBasedPolicy."
            )

        queues = observation[
            5:9
        ]

        return np.asarray(
            queues,
            dtype=np.float32
        )

    # =====================================================
    # SELECT DIRECTION
    # =====================================================

    def select_direction(
        self,
        observation: Iterable[float],
    ) -> int:
        """
        Select the direction with the largest queue.

        Returns:

            0 = north
            1 = east
            2 = south
            3 = west
        """

        queues = (
            self.extract_directional_queues(
                observation
            )
        )

        maximum_queue = float(
            np.max(queues)
        )

        candidates = np.flatnonzero(
            np.isclose(
                queues,
                maximum_queue
            )
        )

        if len(candidates) == 0:

            return 0

        if len(candidates) == 1:

            return int(
                candidates[0]
            )

        # -------------------------------------------------
        # Deterministic tie-breaking
        # -------------------------------------------------

        return int(
            candidates[0]
        )

    # =====================================================
    # ACTION FOR ONE AGENT
    # =====================================================

    def action_for_agent(
        self,
        agent_id: str,
        observation: Iterable[float],
    ) -> int:
        """
        Select an action for one traffic-light agent.

        IMPORTANT:

        The mapping between directional priority
        and TrafficController action IDs is assumed
        to follow:

            0 = north
            1 = east
            2 = south
            3 = west

        If your TrafficController uses a different
        action mapping, this mapping should be changed
        after inspecting TrafficController.
        """

        if agent_id not in (
            EnvironmentConfig.TRAFFIC_LIGHT_IDS
        ):
            raise ValueError(
                f"Unknown traffic-light agent: {agent_id}"
            )

        direction = self.select_direction(
            observation
        )

        number_of_actions = (
            EnvironmentConfig.NUMBER_OF_ACTIONS
        )

        if number_of_actions <= 0:

            raise ValueError(
                "NUMBER_OF_ACTIONS must be greater than 0."
            )

        action = (
            direction
            % number_of_actions
        )

        return int(
            action
        )

    # =====================================================
    # ALL ACTIONS
    # =====================================================

    def actions(
        self,
        observations: Dict[str, Iterable[float]],
    ) -> Dict[str, int]:
        """
        Generate adaptive actions for all agents.

        Parameters
        ----------
        observations:
            Dictionary:

                {
                    "A": observation_A,
                    "B": observation_B,
                    "C": observation_C,
                    "D": observation_D
                }
        """

        actions = {}

        for agent_id in (
            EnvironmentConfig.TRAFFIC_LIGHT_IDS
        ):

            if agent_id not in observations:

                raise KeyError(
                    f"Missing observation for "
                    f"agent {agent_id}."
                )

            actions[agent_id] = (
                self.action_for_agent(
                    agent_id,
                    observations[agent_id]
                )
            )

        return actions

    # =====================================================
    # NUMPY ACTION ARRAY
    # =====================================================

    def action_array(
        self,
        observations: Dict[str, Iterable[float]],
    ) -> np.ndarray:
        """
        Return adaptive actions as a NumPy array.
        """

        actions = self.actions(
            observations
        )

        return np.asarray(
            [
                actions[agent_id]
                for agent_id in (
                    EnvironmentConfig.TRAFFIC_LIGHT_IDS
                )
            ],
            dtype=np.int32
        )

    # =====================================================
    # STEP
    # =====================================================

    def step(
        self,
        observations: Dict[str, Iterable[float]],
    ) -> Dict[str, int]:
        """
        Generate actions for one environment step.
        """

        actions = self.actions(
            observations
        )

        self.step_count += 1

        return actions

    # =====================================================
    # STEP ARRAY
    # =====================================================

    def step_array(
        self,
        observations: Dict[str, Iterable[float]],
    ) -> np.ndarray:
        """
        Generate actions as a NumPy array and
        advance the policy clock.
        """

        actions = self.action_array(
            observations
        )

        self.step_count += 1

        return actions

    # =====================================================
    # INFORMATION
    # =====================================================

    def info(self) -> Dict:
        """
        Return policy configuration.
        """

        return {
            "policy": "queue_based",
            "switch_threshold": (
                self.switch_threshold
            ),
            "seed": self.seed,
            "step_count": self.step_count,
            "number_of_agents": (
                EnvironmentConfig.NUMBER_OF_AGENTS
            ),
            "traffic_lights": (
                EnvironmentConfig.TRAFFIC_LIGHT_IDS
            ),
            "number_of_actions": (
                EnvironmentConfig.NUMBER_OF_ACTIONS
            ),
            "queue_indices": {
                "north": 5,
                "east": 6,
                "south": 7,
                "west": 8,
            },
        }


# =========================================================
# SIMPLE TEST
# =========================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("Queue-Based Baseline Policy Test")
    print("=" * 70)
    print()

    policy = QueueBasedPolicy()

    print(
        "Policy:",
        policy.info()["policy"]
    )

    print(
        "Agents:",
        EnvironmentConfig.TRAFFIC_LIGHT_IDS
    )

    print(
        "Actions:",
        EnvironmentConfig.NUMBER_OF_ACTIONS
    )

    print()

    # -----------------------------------------------------
    # Artificial test observations
    #
    # First 9 values:
    #
    # vehicle count
    # speed
    # waiting
    # stopped
    # phase
    # north
    # east
    # south
    # west
    # -----------------------------------------------------

    observations = {}

    for index, agent_id in enumerate(
        EnvironmentConfig.TRAFFIC_LIGHT_IDS
    ):

        observation = np.zeros(
            EnvironmentConfig.OBSERVATION_SIZE,
            dtype=np.float32
        )

        observation[5] = 10 + index
        observation[6] = 20 + index
        observation[7] = 30 + index
        observation[8] = 40 + index

        observations[agent_id] = (
            observation
        )

    actions = policy.step(
        observations
    )

    print(
        "Generated Actions:"
    )

    for agent_id, action in actions.items():

        print(
            f"  {agent_id}: {action}"
        )

    print()

    print(
        "Policy Info:"
    )

    print(
        policy.info()
    )

    print()
    print("=" * 70)
    print("Queue-Based Policy Test Complete")
    print("=" * 70)
    print()