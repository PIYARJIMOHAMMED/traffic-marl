"""
=========================================================
Traffic MARL Project

Module:
Random Baseline Policy

Description:
Random traffic-light control policy used as a
weak reference baseline.

The policy selects valid actions uniformly at random.

Purpose:

    1. Verify action-space compatibility.
    2. Provide a lower-quality reference.
    3. Compare learned PPO behaviour against
       non-intelligent control.

For reproducibility, a NumPy random seed can
be supplied.

Week:
6

Task:
6.1

Author:
Mohammed Piyarji
=========================================================
"""

from typing import Dict, Optional

import numpy as np

from rl.environments.config import EnvironmentConfig


class RandomPolicy:
    """
    Random baseline traffic-light policy.
    """

    def __init__(
        self,
        seed: Optional[int] = 42,
    ):
        """
        Parameters
        ----------
        seed:
            Random seed for reproducibility.
        """

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
        Reset the policy.

        If a seed is supplied, the random generator
        is recreated using that seed.
        """

        if seed is not None:

            self.seed = seed

            self.rng = np.random.default_rng(
                seed
            )

        self.step_count = 0

    # =====================================================
    # SINGLE ACTION
    # =====================================================

    def action_for_agent(
        self,
        agent_id: str,
    ) -> int:
        """
        Generate a random valid action for one agent.
        """

        if agent_id not in (
            EnvironmentConfig.TRAFFIC_LIGHT_IDS
        ):
            raise ValueError(
                f"Unknown traffic-light agent: {agent_id}"
            )

        return int(
            self.rng.integers(
                0,
                EnvironmentConfig.NUMBER_OF_ACTIONS
            )
        )

    # =====================================================
    # ALL ACTIONS
    # =====================================================

    def actions(self) -> Dict[str, int]:
        """
        Generate random actions for every agent.
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
        Return random actions as a NumPy array.

        Ordering matches:

            EnvironmentConfig.TRAFFIC_LIGHT_IDS
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
        Generate actions and advance the step counter.
        """

        current_actions = self.actions()

        self.step_count += 1

        return current_actions

    # =====================================================
    # STEP ARRAY
    # =====================================================

    def step_array(self) -> np.ndarray:
        """
        Generate random actions as an array.
        """

        current_actions = self.action_array()

        self.step_count += 1

        return current_actions

    # =====================================================
    # INFORMATION
    # =====================================================

    def info(self) -> Dict:
        """
        Return policy metadata.
        """

        return {
            "policy": "random",
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
        }


# =========================================================
# SIMPLE TEST
# =========================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("Random Baseline Policy Test")
    print("=" * 70)
    print()

    policy = RandomPolicy(
        seed=42
    )

    print(
        "Policy:",
        policy.info()["policy"]
    )

    print(
        "Seed:",
        policy.seed
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

    for step in range(10):

        actions = policy.step()

        print(
            f"Step {step + 1:02d}:",
            actions
        )

    print()
    print("=" * 70)
    print("Random Policy Test Complete")
    print("=" * 70)
    print()