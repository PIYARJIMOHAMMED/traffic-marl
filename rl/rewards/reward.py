"""
=========================================================
Traffic MARL Project

Module:
    Cooperative Reward Function

Project:
    MAI-07 | Multi-Agent Reinforcement Learning for
    Network-Wide Adaptive Traffic Control

Description:
    Cooperative reward function for a 4-intersection
    SUMO traffic network.

Reward components:
    1. Traffic speed
    2. Waiting time
    3. Stopped vehicles
    4. Total queue
    5. Directional queue imbalance
    6. Directional starvation
    7. Neighbour cooperation
    8. Network-level cooperation

Observation architecture:

    Local observation = 15 features

        0  vehicle count
        1  average speed
        2  waiting time
        3  stopped vehicles
        4  traffic-light phase
        5  north queue
        6  east queue
        7  south queue
        8  west queue
        9  lane density
        10 green timer
        11 yellow timer
        12 current action
        13 in yellow
        14 safety/reserved feature

    Graph observation = 30 features

        0-14   local observation
        15-29  neighbour observation

Action semantics:

    0 = North/South
    1 = East/West

IMPORTANT:
    This module preserves the calculate_reward()
    interface required by TrafficEnvironment.
=========================================================
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np


class RewardFunction:
    """
    Cooperative reward function for multi-agent traffic control.

    The reward is deliberately separated into:

        Local traffic quality
                +
        Neighbour cooperation
                +
        Network cooperation

    This keeps the reward compatible with the existing
    TrafficEnvironment while adding directional fairness
    and starvation awareness.
    """

    # =========================================================
    # REWARD WEIGHTS
    # =========================================================

    # Positive traffic-flow term.
    speed_weight = 1.50

    # Waiting-time penalty.
    wait_weight = 0.20

    # Stopped-vehicle penalty.
    stop_weight = 0.15

    # Total queue penalty.
    queue_weight = 0.25

    # Directional imbalance penalty.
    imbalance_weight = 0.35

    # Severe directional starvation penalty.
    starvation_weight = 0.40

    # Cooperative neighbour contribution.
    neighbour_weight = 0.10

    # Network/global contribution.
    global_weight = 0.10

    # =========================================================
    # NORMALISATION SCALES
    # =========================================================

    speed_scale = 10.0
    wait_scale = 100.0
    queue_scale = 20.0
    stop_scale = 20.0

    # =========================================================
    # STARVATION THRESHOLDS
    # =========================================================

    starvation_high_queue = 8.0
    starvation_low_queue = 2.0

    # =========================================================
    # REWARD CLIPPING
    # =========================================================

    reward_clip = 10.0

    # =========================================================
    # GRAPH OBSERVATION CONSTANTS
    # =========================================================

    LOCAL_SIZE = 15
    GRAPH_SIZE = 30

    # =========================================================
    # UTILITY METHODS
    # =========================================================

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0,
    ) -> float:
        """
        Safely convert a value to float.
        """

        try:

            number = float(value)

            if not np.isfinite(number):
                return default

            return number

        except (
            TypeError,
            ValueError,
        ):

            return default

    @staticmethod
    def _clip01(
        value: float,
    ) -> float:
        """
        Clip a value into [0, 1].
        """

        return float(
            np.clip(
                value,
                0.0,
                1.0,
            )
        )

    # =========================================================
    # GRAPH OBSERVATION HANDLING
    # =========================================================

    def _extract_local_observation(
        self,
        observation: Any,
    ) -> np.ndarray:
        """
        Extract the 15-dimensional local portion from an
        observation.

        Existing environment passes a 30-D graph observation:

            [ local 15 | neighbour 15 ]

        Therefore only the first 15 values are used for
        local traffic reward calculation.
        """

        # -----------------------------------------------------
        # Dictionary observation
        # -----------------------------------------------------

        if isinstance(
            observation,
            Mapping,
        ):

            local = np.zeros(
                self.LOCAL_SIZE,
                dtype=np.float32,
            )

            local[0] = self._safe_float(
                observation.get(
                    "vehicle_count",
                    0.0,
                )
            )

            local[1] = self._safe_float(
                observation.get(
                    "average_speed",
                    0.0,
                )
            )

            local[2] = self._safe_float(
                observation.get(
                    "waiting_time",
                    0.0,
                )
            )

            local[3] = self._safe_float(
                observation.get(
                    "stopped_vehicles",
                    0.0,
                )
            )

            local[4] = self._safe_float(
                observation.get(
                    "phase",
                    0.0,
                )
            )

            local[5] = self._safe_float(
                observation.get(
                    "north_queue",
                    0.0,
                )
            )

            local[6] = self._safe_float(
                observation.get(
                    "east_queue",
                    0.0,
                )
            )

            local[7] = self._safe_float(
                observation.get(
                    "south_queue",
                    0.0,
                )
            )

            local[8] = self._safe_float(
                observation.get(
                    "west_queue",
                    0.0,
                )
            )

            local[9] = self._safe_float(
                observation.get(
                    "lane_density",
                    0.0,
                )
            )

            local[10] = self._safe_float(
                observation.get(
                    "green_timer",
                    0.0,
                )
            )

            local[11] = self._safe_float(
                observation.get(
                    "yellow_timer",
                    0.0,
                )
            )

            local[12] = self._safe_float(
                observation.get(
                    "current_action",
                    0.0,
                )
            )

            local[13] = self._safe_float(
                observation.get(
                    "in_yellow",
                    0.0,
                )
            )

            local[14] = self._safe_float(
                observation.get(
                    "safety",
                    observation.get(
                        "reserved",
                        0.0,
                    ),
                )
            )

            return local

        # -----------------------------------------------------
        # Array observation
        # -----------------------------------------------------

        try:

            array = np.asarray(
                observation,
                dtype=np.float32,
            ).flatten()

        except (
            TypeError,
            ValueError,
        ):

            array = np.zeros(
                self.LOCAL_SIZE,
                dtype=np.float32,
            )

        # -----------------------------------------------------
        # 30-D graph observation
        # -----------------------------------------------------

        if array.size >= self.GRAPH_SIZE:

            local = array[
                :self.LOCAL_SIZE
            ]

        # -----------------------------------------------------
        # 15-D local observation
        # -----------------------------------------------------

        elif array.size >= self.LOCAL_SIZE:

            local = array[
                :self.LOCAL_SIZE
            ]

        # -----------------------------------------------------
        # Short observation
        # -----------------------------------------------------

        else:

            local = np.zeros(
                self.LOCAL_SIZE,
                dtype=np.float32,
            )

            local[
                :array.size
            ] = array

        return np.asarray(
            local,
            dtype=np.float32,
        )

    # =========================================================
    # METRIC EXTRACTION
    # =========================================================

    def _extract_metrics(
        self,
        observation: Any,
    ) -> Dict[str, float]:
        """
        Extract local traffic metrics.

        The environment's graph observation stores the
        complete 15-D local state first, followed by the
        15-D neighbour state.
        """

        local = self._extract_local_observation(
            observation
        )

        return {

            "vehicle_count":
                max(
                    0.0,
                    self._safe_float(
                        local[0]
                    ),
                ),

            "average_speed":
                max(
                    0.0,
                    self._safe_float(
                        local[1]
                    ),
                ),

            "waiting_time":
                max(
                    0.0,
                    self._safe_float(
                        local[2]
                    ),
                ),

            "stopped_vehicles":
                max(
                    0.0,
                    self._safe_float(
                        local[3]
                    ),
                ),

            "phase":
                self._safe_float(
                    local[4]
                ),

            "north_queue":
                max(
                    0.0,
                    self._safe_float(
                        local[5]
                    ),
                ),

            "east_queue":
                max(
                    0.0,
                    self._safe_float(
                        local[6]
                    ),
                ),

            "south_queue":
                max(
                    0.0,
                    self._safe_float(
                        local[7]
                    ),
                ),

            "west_queue":
                max(
                    0.0,
                    self._safe_float(
                        local[8]
                    ),
                ),

            "lane_density":
                max(
                    0.0,
                    self._safe_float(
                        local[9]
                    ),
                ),

            "green_timer":
                max(
                    0.0,
                    self._safe_float(
                        local[10]
                    ),
                ),

            "yellow_timer":
                max(
                    0.0,
                    self._safe_float(
                        local[11]
                    ),
                ),

            "current_action":
                self._safe_float(
                    local[12]
                ),

            "in_yellow":
                self._safe_float(
                    local[13]
                ),

            "safety":
                self._safe_float(
                    local[14]
                ),
        }

    # =========================================================
    # DIRECTIONAL QUEUE ANALYSIS
    # =========================================================

    def _directional_metrics(
        self,
        metrics: Mapping[str, float],
    ) -> Dict[str, float]:
        """
        Calculate NS and EW queue statistics.
        """

        north = self._safe_float(
            metrics.get(
                "north_queue",
                0.0,
            )
        )

        south = self._safe_float(
            metrics.get(
                "south_queue",
                0.0,
            )
        )

        east = self._safe_float(
            metrics.get(
                "east_queue",
                0.0,
            )
        )

        west = self._safe_float(
            metrics.get(
                "west_queue",
                0.0,
            )
        )

        ns_queue = max(
            0.0,
            north + south,
        )

        ew_queue = max(
            0.0,
            east + west,
        )

        total_queue = (
            ns_queue
            + ew_queue
        )

        imbalance = abs(
            ns_queue
            - ew_queue
        )

        if total_queue > 0.0:

            imbalance_ratio = (
                imbalance
                / total_queue
            )

        else:

            imbalance_ratio = 0.0

        return {

            "ns_queue":
                ns_queue,

            "ew_queue":
                ew_queue,

            "total_queue":
                total_queue,

            "imbalance":
                imbalance,

            "imbalance_ratio":
                self._clip01(
                    imbalance_ratio
                ),
        }

    # =========================================================
    # STARVATION PENALTY
    # =========================================================

    def _starvation_penalty(
        self,
        ns_queue: float,
        ew_queue: float,
    ) -> float:
        """
        Detect severe directional starvation.

        A penalty is applied when:

            one direction has a high queue

        while:

            the opposite direction has a very small queue.

        This is a reward-shaping term.

        It is NOT the safety constraint.

        Safety remains handled by the controller and
        safety validator.
        """

        high = float(
            self.starvation_high_queue
        )

        low = float(
            self.starvation_low_queue
        )

        # -----------------------------------------------------
        # NS heavily queued, EW nearly empty
        # -----------------------------------------------------

        if (
            ns_queue >= high
            and ew_queue <= low
        ):

            severity = (
                (ns_queue - high + 1.0)
                / max(
                    high,
                    1.0,
                )
            )

            return self._clip01(
                severity
            )

        # -----------------------------------------------------
        # EW heavily queued, NS nearly empty
        # -----------------------------------------------------

        if (
            ew_queue >= high
            and ns_queue <= low
        ):

            severity = (
                (ew_queue - high + 1.0)
                / max(
                    high,
                    1.0,
                )
            )

            return self._clip01(
                severity
            )

        return 0.0

    # =========================================================
    # LOCAL REWARD
    # =========================================================

    def local_reward(
        self,
        observation: Any,
    ) -> float:
        """
        Calculate the local traffic-control reward.

        Reward:

            + speed
            - waiting
            - stopped vehicles
            - total queue
            - directional imbalance
            - starvation
        """

        metrics = self._extract_metrics(
            observation
        )

        # -----------------------------------------------------
        # Basic traffic metrics
        # -----------------------------------------------------

        average_speed = max(
            0.0,
            metrics[
                "average_speed"
            ],
        )

        waiting_time = max(
            0.0,
            metrics[
                "waiting_time"
            ],
        )

        stopped_vehicles = max(
            0.0,
            metrics[
                "stopped_vehicles"
            ],
        )

        # -----------------------------------------------------
        # Directional queues
        # -----------------------------------------------------

        directional = (
            self._directional_metrics(
                metrics
            )
        )

        ns_queue = directional[
            "ns_queue"
        ]

        ew_queue = directional[
            "ew_queue"
        ]

        total_queue = directional[
            "total_queue"
        ]

        imbalance_ratio = directional[
            "imbalance_ratio"
        ]

        # -----------------------------------------------------
        # Normalised speed
        # -----------------------------------------------------

        speed_score = self._clip01(
            average_speed
            / self.speed_scale
        )

        # -----------------------------------------------------
        # Normalised waiting
        # -----------------------------------------------------

        waiting_penalty = self._clip01(
            waiting_time
            / self.wait_scale
        )

        # -----------------------------------------------------
        # Normalised stopped vehicles
        # -----------------------------------------------------

        stopped_penalty = self._clip01(
            stopped_vehicles
            / self.stop_scale
        )

        # -----------------------------------------------------
        # Normalised total queue
        # -----------------------------------------------------

        queue_penalty = self._clip01(
            total_queue
            / self.queue_scale
        )

        # -----------------------------------------------------
        # Directional starvation
        # -----------------------------------------------------

        starvation_penalty = (
            self._starvation_penalty(
                ns_queue,
                ew_queue,
            )
        )

        # -----------------------------------------------------
        # Compose reward
        # -----------------------------------------------------

        reward = 0.0

        # Traffic-flow quality.
        reward += (
            self.speed_weight
            * speed_score
        )

        # Waiting-time penalty.
        reward -= (
            self.wait_weight
            * waiting_penalty
        )

        # Stopped-vehicle penalty.
        reward -= (
            self.stop_weight
            * stopped_penalty
        )

        # Total queue penalty.
        reward -= (
            self.queue_weight
            * queue_penalty
        )

        # Directional balance penalty.
        reward -= (
            self.imbalance_weight
            * imbalance_ratio
        )

        # Severe starvation penalty.
        reward -= (
            self.starvation_weight
            * starvation_penalty
        )

        # -----------------------------------------------------
        # PPO-safe clipping
        # -----------------------------------------------------

        reward = float(
            np.clip(
                reward,
                -self.reward_clip,
                self.reward_clip,
            )
        )

        return reward

    # =========================================================
    # NEIGHBOUR REWARD
    # =========================================================

    def neighbour_reward(
        self,
        agent_id: str | None,
        local_rewards: Mapping[str, float] | None,
        neighbours: Any,
    ) -> float:
        """
        Calculate the average reward of neighbouring agents.

        Existing TrafficEnvironment supplies:

            neighbour_information

        obtained from:

            self.marl.communication.neighbour_status(...)

        Depending on the implementation, this can be:

            - list of neighbour IDs
            - tuple of neighbour IDs
            - set of neighbour IDs
            - dictionary keyed by neighbour IDs
            - list/dictionary containing neighbour records

        This method handles the common forms defensively.
        """

        if not local_rewards:
            return 0.0

        # -----------------------------------------------------
        # Convert neighbour information into IDs
        # -----------------------------------------------------

        neighbour_ids = []

        if neighbours is None:

            neighbour_ids = []

        elif isinstance(
            neighbours,
            Mapping,
        ):

            neighbour_ids = list(
                neighbours.keys()
            )

        elif isinstance(
            neighbours,
            (list, tuple, set),
        ):

            for item in neighbours:

                if isinstance(
                    item,
                    str,
                ):

                    neighbour_ids.append(
                        item
                    )

                elif isinstance(
                    item,
                    Mapping,
                ):

                    # Try common ID keys.
                    neighbour_id = (
                        item.get("id")
                        or item.get(
                            "agent_id"
                        )
                        or item.get(
                            "traffic_light"
                        )
                    )

                    if neighbour_id is not None:

                        neighbour_ids.append(
                            str(
                                neighbour_id
                            )
                        )

        # -----------------------------------------------------
        # Remove self from neighbour list
        # -----------------------------------------------------

        if agent_id is not None:

            neighbour_ids = [
                neighbour_id
                for neighbour_id in neighbour_ids
                if str(neighbour_id)
                != str(agent_id)
            ]

        # -----------------------------------------------------
        # Collect known local rewards
        # -----------------------------------------------------

        values = []

        for neighbour_id in neighbour_ids:

            # Direct key match.
            if neighbour_id in local_rewards:

                values.append(
                    self._safe_float(
                        local_rewards[
                            neighbour_id
                        ]
                    )
                )

                continue

            # String-normalised key match.
            for key, value in (
                local_rewards.items()
            ):

                if (
                    str(key)
                    == str(neighbour_id)
                ):

                    values.append(
                        self._safe_float(
                            value
                        )
                    )

                    break

        # -----------------------------------------------------
        # No matching neighbour rewards
        # -----------------------------------------------------

        if not values:

            return 0.0

        return float(
            np.mean(values)
        )

    # =========================================================
    # GLOBAL REWARD
    # =========================================================

    def global_reward(
        self,
        local_rewards: Mapping[str, float] | None,
    ) -> float:
        """
        Calculate network-level cooperative reward.

        The global reward is the mean of all intersection
        local rewards.
        """

        if not local_rewards:

            return 0.0

        values = []

        for value in (
            local_rewards.values()
        ):

            values.append(
                self._safe_float(
                    value
                )
            )

        if not values:

            return 0.0

        return float(
            np.mean(values)
        )

    # =========================================================
    # MAIN ENVIRONMENT-COMPATIBLE REWARD API
    # =========================================================

    def calculate_reward(
        self,
        graph_observation: Any,
        neighbour_information: Any = None,
        local_rewards: Mapping[str, float] | None = None,
    ) -> float:
        """
        Calculate the final cooperative reward for ONE agent.

        This is the exact API expected by the existing
        TrafficEnvironment.

        Existing call:

            self.reward_function.calculate_reward(
                graph_observations[traffic_light],
                neighbour_information,
                local_rewards
            )

        Parameters
        ----------
        graph_observation:
            30-dimensional graph observation:

                [ local 15 | neighbour 15 ]

        neighbour_information:
            Neighbour information returned by the MARL
            communication module.

        local_rewards:
            Dictionary containing local rewards for all
            traffic-light agents.

        Returns
        -------
        float
            Final cooperative reward for this agent.
        """

        # -----------------------------------------------------
        # Local reward
        # -----------------------------------------------------

        local_reward = self.local_reward(
            graph_observation
        )

        # -----------------------------------------------------
        # IMPORTANT:
        #
        # The existing TrafficEnvironment does not pass the
        # traffic-light ID into this method.
        #
        # Therefore we cannot reliably identify the current
        # agent from local_rewards alone.
        #
        # We therefore use neighbour_information directly
        # where possible and calculate the cooperative term
        # from the supplied local reward values.
        # -----------------------------------------------------

        neighbour_reward = 0.0

        if local_rewards:

            # Extract IDs from neighbour information.
            neighbour_ids = []

            if isinstance(
                neighbour_information,
                Mapping,
            ):

                neighbour_ids = list(
                    neighbour_information.keys()
                )

            elif isinstance(
                neighbour_information,
                (list, tuple, set),
            ):

                for item in (
                    neighbour_information
                ):

                    if isinstance(
                        item,
                        str,
                    ):

                        neighbour_ids.append(
                            item
                        )

                    elif isinstance(
                        item,
                        Mapping,
                    ):

                        neighbour_id = (
                            item.get("id")
                            or item.get(
                                "agent_id"
                            )
                            or item.get(
                                "traffic_light"
                            )
                        )

                        if neighbour_id is not None:

                            neighbour_ids.append(
                                str(
                                    neighbour_id
                                )
                            )

            # Collect rewards for known neighbours.
            neighbour_values = []

            for neighbour_id in (
                neighbour_ids
            ):

                if (
                    neighbour_id
                    in local_rewards
                ):

                    neighbour_values.append(
                        self._safe_float(
                            local_rewards[
                                neighbour_id
                            ]
                        )
                    )

                    continue

                # Try string-equivalent keys.
                for key, value in (
                    local_rewards.items()
                ):

                    if (
                        str(key)
                        == str(neighbour_id)
                    ):

                        neighbour_values.append(
                            self._safe_float(
                                value
                            )
                        )

                        break

            if neighbour_values:

                neighbour_reward = float(
                    np.mean(
                        neighbour_values
                    )
                )

        # -----------------------------------------------------
        # Network reward
        # -----------------------------------------------------

        network_reward = (
            self.global_reward(
                local_rewards
            )
        )

        # -----------------------------------------------------
        # Final cooperative reward
        # -----------------------------------------------------

        final_reward = (
            local_reward
            + (
                self.neighbour_weight
                * neighbour_reward
            )
            + (
                self.global_weight
                * network_reward
            )
        )

        # -----------------------------------------------------
        # PPO-safe clipping
        # -----------------------------------------------------

        return float(
            np.clip(
                final_reward,
                -self.reward_clip,
                self.reward_clip,
            )
        )

    # =========================================================
    # MULTI-AGENT CALCULATION
    # =========================================================

    def calculate(
        self,
        observations: Mapping[str, Any],
        neighbours: Mapping[str, Any] | None = None,
    ) -> Dict[str, float]:
        """
        Calculate cooperative rewards for all agents.

        This method is useful for standalone evaluation and
        testing.

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

        neighbours:
            Network topology:

                {
                    "A": ["B", "C"],
                    "B": ["A", "D"],
                    "C": ["A", "D"],
                    "D": ["B", "C"]
                }
        """

        if not observations:

            return {}

        if neighbours is None:

            neighbours = {}

        # -----------------------------------------------------
        # Local rewards
        # -----------------------------------------------------

        local_rewards = {}

        for agent_id, observation in (
            observations.items()
        ):

            local_rewards[
                agent_id
            ] = self.local_reward(
                observation
            )

        # -----------------------------------------------------
        # Global reward
        # -----------------------------------------------------

        network_reward = (
            self.global_reward(
                local_rewards
            )
        )

        # -----------------------------------------------------
        # Cooperative rewards
        # -----------------------------------------------------

        final_rewards = {}

        for agent_id, observation in (
            observations.items()
        ):

            local = (
                local_rewards[
                    agent_id
                ]
            )

            neighbour = (
                self.neighbour_reward(
                    agent_id,
                    local_rewards,
                    neighbours.get(
                        agent_id,
                        [],
                    ),
                )
            )

            final = (
                local
                + (
                    self.neighbour_weight
                    * neighbour
                )
                + (
                    self.global_weight
                    * network_reward
                )
            )

            final_rewards[
                agent_id
            ] = float(
                np.clip(
                    final,
                    -self.reward_clip,
                    self.reward_clip,
                )
            )

        return final_rewards

    # =========================================================
    # COMPATIBILITY ALIAS
    # =========================================================

    def compute(
        self,
        observations: Mapping[str, Any],
        neighbours: Mapping[str, Any] | None = None,
    ) -> Dict[str, float]:
        """
        Compatibility alias for calculate().
        """

        return self.calculate(
            observations,
            neighbours,
        )

    # =========================================================
    # SINGLE-AGENT COMPATIBILITY
    # =========================================================

    def get_reward(
        self,
        observation: Any,
    ) -> float:
        """
        Calculate a local reward for one observation.
        """

        return self.local_reward(
            observation
        )


# =============================================================
# STANDALONE TEST
# =============================================================

if __name__ == "__main__":

    print()
    print("=" * 65)
    print("COOPERATIVE MARL REWARD TEST")
    print("=" * 65)

    reward_function = RewardFunction()

    # ---------------------------------------------------------
    # 30-D graph observations
    #
    # First 15 = local
    # Last 15 = neighbour
    # ---------------------------------------------------------

    test_observations = {

        "A": np.array(
            [
                # Local 15
                20.0,       # vehicle count
                7.5,        # average speed
                20.0,       # waiting time
                3.0,        # stopped vehicles
                0.0,        # phase
                5.0,        # north queue
                2.0,        # east queue
                4.0,        # south queue
                2.0,        # west queue
                0.20,       # lane density
                12.0,       # green timer
                0.0,        # yellow timer
                0.0,        # current action
                0.0,        # in yellow
                0.0,        # safety

                # Neighbour 15
                4.0,
                7.0,
                15.0,
                2.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        ),

        "B": np.array(
            [
                18.0,
                7.0,
                15.0,
                2.0,
                0.0,
                4.0,
                3.0,
                4.0,
                3.0,
                0.18,
                15.0,
                0.0,
                0.0,
                0.0,
                0.0,

                4.0,
                7.0,
                15.0,
                2.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        ),

        "C": np.array(
            [
                25.0,
                6.5,
                30.0,
                5.0,
                2.0,
                10.0,
                1.0,
                8.0,
                1.0,
                0.25,
                20.0,
                0.0,
                1.0,
                0.0,
                0.0,

                4.0,
                6.0,
                20.0,
                3.0,
                2.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        ),

        "D": np.array(
            [
                15.0,
                8.0,
                10.0,
                1.0,
                2.0,
                3.0,
                4.0,
                3.0,
                4.0,
                0.15,
                25.0,
                0.0,
                1.0,
                0.0,
                0.0,

                3.0,
                7.0,
                10.0,
                2.0,
                2.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        ),
    }

    # ---------------------------------------------------------
    # Network topology
    # ---------------------------------------------------------

    network = {

        "A": [
            "B",
            "C",
        ],

        "B": [
            "A",
            "D",
        ],

        "C": [
            "A",
            "D",
        ],

        "D": [
            "B",
            "C",
        ],
    }

    # ---------------------------------------------------------
    # Calculate cooperative rewards
    # ---------------------------------------------------------

    rewards = reward_function.calculate(
        test_observations,
        network,
    )

    print()
    print("Agent rewards")
    print("-" * 65)

    for agent_id, reward in (
        rewards.items()
    ):

        print(
            f"{agent_id}: "
            f"{reward:.4f}"
        )

    # ---------------------------------------------------------
    # Test exact TrafficEnvironment API
    # ---------------------------------------------------------

    print()
    print("TrafficEnvironment API test")
    print("-" * 65)

    local_rewards = {}

    for agent_id, observation in (
        test_observations.items()
    ):

        local_rewards[
            agent_id
        ] = reward_function.local_reward(
            observation
        )

    for agent_id in (
        ["A", "B", "C", "D"]
    ):

        cooperative_reward = (
            reward_function.calculate_reward(
                test_observations[
                    agent_id
                ],
                network[
                    agent_id
                ],
                local_rewards,
            )
        )

        print(
            f"{agent_id}: "
            f"{cooperative_reward:.4f}"
        )

    # ---------------------------------------------------------
    # Starvation test
    # ---------------------------------------------------------

    starvation_observation = np.array(
        [
            20.0,
            6.0,
            50.0,
            8.0,
            0.0,

            # NS = 18
            10.0,
            1.0,
            8.0,
            0.0,

            0.20,
            15.0,
            0.0,
            0.0,
            0.0,
            0.0,

            # Neighbour part
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )

    balanced_observation = np.array(
        [
            20.0,
            6.0,
            50.0,
            8.0,
            0.0,

            # NS = 10
            5.0,
            4.0,
            5.0,
            4.0,

            0.20,
            15.0,
            0.0,
            0.0,
            0.0,
            0.0,

            # Neighbour part
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )

    starvation_reward = (
        reward_function.get_reward(
            starvation_observation
        )
    )

    balanced_reward = (
        reward_function.get_reward(
            balanced_observation
        )
    )

    print()
    print("Directional fairness test")
    print("-" * 65)

    print(
        f"Starvation case reward : "
        f"{starvation_reward:.4f}"
    )

    print(
        f"Balanced case reward   : "
        f"{balanced_reward:.4f}"
    )

    # ---------------------------------------------------------
    # Assertions
    # ---------------------------------------------------------

    assert np.isfinite(
        starvation_reward
    )

    assert np.isfinite(
        balanced_reward
    )

    assert (
        balanced_reward
        > starvation_reward
    )

    # Check 30-D observation handling.
    for observation in (
        test_observations.values()
    ):

        assert (
            reward_function
            ._extract_local_observation(
                observation
            )
            .shape
            == (15,)
        )

    print()
    print("=" * 65)
    print("ALL REWARD TESTS PASSED")
    print("=" * 65)