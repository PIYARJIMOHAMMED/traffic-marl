"""
=========================================================
Traffic MARL Project

Module:
Neighbour Communication Manager

Description:
Handles communication between neighbouring
traffic-light agents.

Each simulation step the manager gathers
information from neighbouring intersections.

Week:
5

Task:
5.4

Author:
Mohammed Piyarji
=========================================================
"""

from typing import Dict
from typing import List

import numpy as np

from rl.environments.config import EnvironmentConfig


class NeighbourManager:

    def __init__(

        self,

        marl_controller

    ):

        self.controller = marl_controller

        self.graph = EnvironmentConfig.NEIGHBOURS
        
    # =====================================================
    # LOCAL STATUS
    # =====================================================

    def local_status(

        self,

        agent_id: str

    ) -> Dict:

        """
        Returns the local status of one agent.
        """

        return (

            self.controller

            .get_agent(agent_id)

            .export_status()

        )

    # =====================================================
    # NEIGHBOUR IDS
    # =====================================================

    def neighbours(

        self,

        agent_id: str

    ) -> List[str]:

        return self.graph.get(

            agent_id,

            []

        )

    # =====================================================
    # NEIGHBOUR STATUS
    # =====================================================

    def neighbour_status(

        self,

        agent_id: str

    ) -> Dict:

        """
        Returns

        {

            "B": {...},

            "C": {...}

        }

        """

        information = {}

        for neighbour in self.neighbours(

            agent_id

        ):

            information[neighbour] = (

                self.local_status(

                    neighbour

                )

            )

        return information

    # =====================================================
    # NETWORK STATUS
    # =====================================================

    def network_status(self):

        """
        Returns every agent status.
        """

        status = {}

        for agent in (

            EnvironmentConfig

            .TRAFFIC_LIGHT_IDS

        ):

            status[agent] = (

                self.local_status(

                    agent

                )

            )

        return status

    # =====================================================
    # SHARED REWARD
    # =====================================================

    def neighbour_reward(

        self,

        agent_id: str

    ) -> float:

        """
        Average reward of neighbouring agents.
        """

        rewards = []

        for neighbour in self.neighbours(

            agent_id

        ):

            rewards.append(

                self.controller

                .get_agent(neighbour)

                .get_reward()

            )

        if len(rewards) == 0:

            return 0.0

        return float(

            np.mean(rewards)

        )

    # =====================================================
    # SHARED REWARD
    # =====================================================

    def shared_reward(
        self,
        agent_id: str
    ) -> float:
        """
        Shared reward used for cooperative MARL.

        Formula

        70% Local Reward

        30% Neighbour Reward
        """

        local_reward = (

            self.controller

            .get_agent(agent_id)

            .get_reward()

        )

        neighbour_reward = (

            self.neighbour_reward(

                agent_id

            )

        )

        return (

            0.70 * local_reward

            +

            0.30 * neighbour_reward

        )

    # =====================================================
    # NETWORK WAITING TIME
    # =====================================================

    def network_waiting_time(self) -> float:

        waiting = []

        for agent in self.controller.get_agents().values():

            waiting.append(

                agent.statistics.waiting_time

            )

        if len(waiting) == 0:

            return 0.0

        return float(

            np.mean(waiting)

        )

    # =====================================================
    # NETWORK QUEUE LENGTH
    # =====================================================

    def network_queue_length(self) -> float:

        queues = []

        for agent in self.controller.get_agents().values():

            queues.append(

                agent.statistics.queue_length

            )

        if len(queues) == 0:

            return 0.0

        return float(

            np.mean(queues)

        )

    # =====================================================
    # NETWORK SPEED
    # =====================================================

    def network_average_speed(self) -> float:

        speeds = []

        for agent in self.controller.get_agents().values():

            speeds.append(

                agent.statistics.average_speed

            )

        if len(speeds) == 0:

            return 0.0

        return float(

            np.mean(speeds)

        )

    # =====================================================
    # GRAPH OBSERVATION
    # =====================================================

    def graph_observation(
        self,
        agent_id: str
    ) -> np.ndarray:
        """
        Returns one graph observation vector.

        Local State

        +

        Average Neighbour State
        """

        agent = self.controller.get_agent(

            agent_id

        )

        local_state = agent.get_state()

        if local_state is None:

            local_state = np.zeros(

                EnvironmentConfig.OBSERVATION_SIZE,

                dtype=np.float32

            )

        neighbour_states = []

        for neighbour in self.neighbours(

            agent_id

        ):

            state = (

                self.controller

                .get_agent(neighbour)

                .get_state()

            )

            if state is not None:

                neighbour_states.append(

                    state

                )

        if len(neighbour_states) == 0:

            neighbour_average = np.zeros_like(

                local_state

            )

        else:

            neighbour_average = np.mean(

                neighbour_states,

                axis=0

            )

        return np.concatenate(

            [

                local_state,

                neighbour_average

            ]

        ).astype(np.float32)

    # =====================================================
    # COMMUNICATION SUMMARY
    # =====================================================

    def communication_summary(self) -> Dict:

        summary = {}

        for agent in (

            EnvironmentConfig

            .TRAFFIC_LIGHT_IDS

        ):

            summary[agent] = {

                "neighbours": self.neighbours(

                    agent

                ),

                "shared_reward": round(

                    self.shared_reward(

                        agent

                    ),

                    3

                ),

                "network_waiting": round(

                    self.network_waiting_time(),

                    3

                ),

                "network_queue": round(

                    self.network_queue_length(),

                    3

                ),

                "network_speed": round(

                    self.network_average_speed(),

                    3

                )

            }

        return summary


    # =====================================================
    # NETWORK SUMMARY
    # =====================================================

    def network_summary(self) -> Dict:
        """
        Returns the communication summary for the complete network.

        This method provides the network-level interface expected
        by TrafficEnvironment while keeping communication_summary()
        available for existing code.
        """
        return self.communication_summary()
    