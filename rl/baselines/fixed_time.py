"""
=========================================================
Traffic MARL Project

Module:
Fixed-Time Baseline Policy

Description:
Traditional fixed-time traffic signal controller.

The policy does not use reinforcement learning.

Each traffic-light agent follows a deterministic
cyclic sequence of signal actions.

Purpose:

    Research baseline for comparison with PPO MARL.

The controller is:

    deterministic
    reproducible
    computationally inexpensive
    independent of learned parameters

Week:
6

Task:
6.1

Author:
Mohammed Piyarji
=========================================================
"""

from typing import Dict

import numpy as np

from rl.environments.config import EnvironmentConfig


class FixedTimePolicy:
    """
    Deterministic fixed-time traffic signal policy.

    Each intersection follows a cyclic signal sequence.

    Example:

        Action 0
           ↓
        Action 1
           ↓
        Action 2
           ↓
        Action 3
           ↓
        repeat

    The exact meaning of each action is defined by
    TrafficController.
    """

    def __init__(
        self,
        cycle_length: int = 10,
    ):
        """
        Parameters
        ----------
        cycle_length:
            Number of environment steps between
            signal changes.
        """

        if cycle_length <= 0:
            raise ValueError(
                "cycle_length must be greater than 0."
            )

        self.cycle_length = int(
            cycle_length
        )

        self.step_count = 0

        self.agent_offsets: Dict[str, int] = {}

        for index, agent_id in enumerate(
            EnvironmentConfig.TRAFFIC_LIGHT_IDS
        ):
            self.agent_offsets[agent_id] = index

    # =====================================================
    # RESET
    # =====================================================

    def reset(self) -> None:
        """
        Reset the policy state.
        """

        self.step_count = 0

    # =====================================================
    # ACTION FOR ONE AGENT
    # =====================================================

    def action_for_agent(
        self,
        agent_id: str,
    ) -> int:
        """
        Return the deterministic action for one
        traffic-light agent.
        """

        if agent_id not in (
            EnvironmentConfig.TRAFFIC_LIGHT_IDS
        ):
            raise ValueError(
                f"Unknown traffic-light agent: {agent_id}"
            )

        number_of_actions = (
            EnvironmentConfig.NUMBER_OF_ACTIONS
        )

        if number_of_actions <= 0:
            raise ValueError(
                "NUMBER_OF_ACTIONS must be greater than 0."
            )

        offset = self.agent_offsets.get(
            agent_id,
            0
        )

        phase_index = (
            (
                self.step_count
                // self.cycle_length
            )
            + offset
        ) % number_of_actions

        return int(
            phase_index
        )

    # =====================================================
    # ACTION DICTIONARY
    # =====================================================

    def actions(self) -> Dict[str, int]:
        """
        Return actions for all traffic-light agents.

        Returns
        -------
        Dict[str, int]
            Example:

            {
                "A": 0,
                "B": 1,
                "C": 2,
                "D": 3
            }
        """

        return {
            agent_id: self.action_for_agent(
                agent_id
            )
            for agent_id in (
                EnvironmentConfig.TRAFFIC_LIGHT_IDS
            )
        }

    # =====================================================
    # NUMPY ACTION
    # =====================================================

    def action_array(self) -> np.ndarray:
        """
        Return actions in the same ordering used by
        TrafficEnvironment.action_space.

        Returns
        -------
        np.ndarray
        """

        return np.asarray(
            [
                self.action_for_agent(agent_id)
                for agent_id in (
                    EnvironmentConfig.TRAFFIC_LIGHT_IDS
                )
            ],
            dtype=np.int32
        )

    # =====================================================
    # STEP
    # =====================================================

    def step(self) -> Dict[str, int]:
        """
        Generate actions for the current step and
        advance the policy clock.

        Returns
        -------
        Dict[str, int]
            Actions for all agents.
        """

        current_actions = self.actions()

        self.step_count += 1

        return current_actions

    # =====================================================
    # STEP WITH ARRAY
    # =====================================================

    def step_array(self) -> np.ndarray:
        """
        Generate an action array and advance the
        policy clock.

        Useful directly with:

            env.step(actions)
        """

        current_actions = self.action_array()

        self.step_count += 1

        return current_actions

    # =====================================================
    # INFORMATION
    # =====================================================

    def info(self) -> Dict:
        """
        Return policy configuration and runtime state.
        """

        return {
            "policy": "fixed_time",
            "cycle_length": self.cycle_length,
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
        }


# =========================================================
# SIMPLE TEST
# =========================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("Fixed-Time Baseline Policy Test")
    print("=" * 70)
    print()

    policy = FixedTimePolicy(
        cycle_length=10
    )

    print(
        "Policy:",
        policy.info()["policy"]
    )

    print(
        "Cycle Length:",
        policy.cycle_length
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

    for step in range(20):

        actions = policy.step()

        print(
            f"Step {step + 1:02d}:",
            actions
        )

    print()
    print("=" * 70)
    print("Fixed-Time Policy Test Complete")
    print("=" * 70)
    print()