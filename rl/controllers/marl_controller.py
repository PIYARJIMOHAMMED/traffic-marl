"""
=========================================================
Traffic MARL Project

Module:
MARL Controller

Description:
Central controller for the complete Multi-Agent
Traffic Reinforcement Learning system.

Responsibilities:

• Agent management
• State synchronization
• Action execution
• Reward synchronization
• Neighbour communication
• Shared reward propagation
• Graph observations
• Network statistics

Architecture:

    TrafficAgent
          ↓
    MARLController
          ↓
    NeighbourManager
          ↓
    Traffic Environment

Agents:

    A
    B
    C
    D

=========================================================
"""

from typing import Dict

import numpy as np

from rl.agents.traffic_agent import TrafficAgent
from rl.communication.neighbour_manager import NeighbourManager
from rl.environments.config import EnvironmentConfig


class MARLController:

    """
    Central controller for all traffic-light agents.
    """

    # =====================================================
    # INITIALIZATION
    # =====================================================

    def __init__(self):

        self.agents: Dict[str, TrafficAgent] = {}

        # -------------------------------------------------
        # Create one agent per traffic light
        # -------------------------------------------------

        for agent_id in EnvironmentConfig.TRAFFIC_LIGHT_IDS:

            neighbours = EnvironmentConfig.NEIGHBOURS.get(
                agent_id,
                []
            )

            self.agents[agent_id] = TrafficAgent(

                agent_id=agent_id,

                neighbours=neighbours

            )

        # -------------------------------------------------
        # Communication manager
        # -------------------------------------------------

        self.communication = NeighbourManager(
            self
        )

    # =====================================================
    # RESET
    # =====================================================

    def reset(self):
        """
        Reset every MARL agent.
        """

        for agent in self.agents.values():

            agent.reset()

    # =====================================================
    # AGENT ACCESS
    # =====================================================

    def get_agent(
        self,
        agent_id: str
    ):

        if agent_id not in self.agents:

            raise KeyError(
                f"Unknown agent: {agent_id}"
            )

        return self.agents[agent_id]

    # -----------------------------------------------------

    def get_agents(self):

        return self.agents

    # =====================================================
    # STATE MANAGEMENT
    # =====================================================

    def update_states(
        self,
        observations: Dict[str, np.ndarray]
    ):
        """
        Update the current state of each agent.
        """

        for agent_id, observation in observations.items():

            if agent_id not in self.agents:

                raise KeyError(
                    f"Unknown agent: {agent_id}"
                )

            observation = np.asarray(
                observation,
                dtype=np.float32
            ).flatten()

            if observation.size == 0:

                raise ValueError(
                    f"Empty observation received "
                    f"for agent {agent_id}."
                )

            self.agents[agent_id].set_state(
                observation
            )

    # -----------------------------------------------------

    def get_states(self):

        return {

            agent_id:
                agent.get_state()

            for agent_id, agent
            in self.agents.items()

        }

    # =====================================================
    # ACTION MANAGEMENT
    # =====================================================

    def apply_actions(
        self,
        actions: Dict[str, int]
    ):
        """
        Store the latest action selected
        for each traffic-light agent.
        """

        for agent_id, action in actions.items():

            if agent_id not in self.agents:

                raise KeyError(
                    f"Unknown agent: {agent_id}"
                )

            action = int(action)

            if action not in EnvironmentConfig.ACTIONS:

                raise ValueError(
                    f"Invalid action {action} "
                    f"for agent {agent_id}. "
                    f"Valid actions: "
                    f"{EnvironmentConfig.ACTIONS}"
                )

            self.agents[agent_id].set_action(
                action
            )

    # -----------------------------------------------------

    def get_actions(self):

        return {

            agent_id:
                agent.get_action()

            for agent_id, agent
            in self.agents.items()

        }

    # =====================================================
    # REWARD MANAGEMENT
    # =====================================================

    def update_rewards(
        self,
        rewards: Dict[str, float]
    ):
        """
        Synchronize rewards between agents
        and update cooperative/shared rewards.
        """

        for agent_id, reward in rewards.items():

            if agent_id not in self.agents:

                raise KeyError(
                    f"Unknown agent: {agent_id}"
                )

            self.agents[agent_id].set_reward(
                float(reward)
            )

        self.update_shared_rewards()

    # -----------------------------------------------------

    def get_rewards(self):

        return {

            agent_id:
                agent.get_reward()

            for agent_id, agent
            in self.agents.items()

        }

    # =====================================================
    # SHARED REWARDS
    # =====================================================

    def update_shared_rewards(self):
        """
        Calculate and store cooperative
        shared reward for every agent.
        """

        for agent_id in EnvironmentConfig.TRAFFIC_LIGHT_IDS:

            reward = (
                self.communication
                .shared_reward(
                    agent_id
                )
            )

            self.agents[agent_id].set_shared_reward(
                float(reward)
            )

    # -----------------------------------------------------

    def get_shared_rewards(self):

        return {

            agent_id:
                self.agents[agent_id]
                .get_shared_reward()

            for agent_id
            in EnvironmentConfig.TRAFFIC_LIGHT_IDS

        }

    # =====================================================
    # NEIGHBOUR COMMUNICATION
    # =====================================================

    def synchronize_communication(self):
        """
        Update every agent with the latest
        information from its neighbours.
        """

        for agent_id in EnvironmentConfig.TRAFFIC_LIGHT_IDS:

            information = (
                self.communication
                .neighbour_status(
                    agent_id
                )
            )

            self.agents[agent_id].update_neighbour_information(
                information
            )

    # -----------------------------------------------------

    def broadcast(self):

        return (
            self.communication
            .network_status()
        )

    # =====================================================
    # GRAPH OBSERVATION
    # =====================================================

    def graph_observation(
        self,
        agent_id: str
    ) -> np.ndarray:
        """
        Return graph-aware observation for one agent.

        Graph observation contains:

            local state
                    +
            neighbour information
        """

        if agent_id not in self.agents:

            raise KeyError(
                f"Unknown agent: {agent_id}"
            )

        observation = (
            self.communication
            .graph_observation(
                agent_id
            )
        )

        return np.asarray(
            observation,
            dtype=np.float32
        ).flatten()

    # -----------------------------------------------------

    def graph_observations(self):

        observations = {}

        for agent_id in EnvironmentConfig.TRAFFIC_LIGHT_IDS:

            observations[agent_id] = (
                self.graph_observation(
                    agent_id
                )
            )

        return observations

    # =====================================================
    # GLOBAL STATE
    # =====================================================

    def global_state(self) -> np.ndarray:
        """
        Concatenate the current state of all agents.

        For four agents:

            A + B + C + D

        The actual dimension depends on the
        current state stored by each agent.
        """

        state = []

        for agent_id in EnvironmentConfig.TRAFFIC_LIGHT_IDS:

            observation = (
                self.agents[agent_id]
                .get_state()
            )

            if observation is None:

                observation = np.zeros(
                    EnvironmentConfig.OBSERVATION_SIZE,
                    dtype=np.float32
                )

            observation = np.asarray(
                observation,
                dtype=np.float32
            ).flatten()

            state.extend(
                observation.tolist()
            )

        return np.asarray(
            state,
            dtype=np.float32
        )

    # =====================================================
    # NETWORK STATISTICS
    # =====================================================

    def network_statistics(self):

        return {

            "waiting_time":
                float(
                    self.communication
                    .network_waiting_time()
                ),

            "queue_length":
                float(
                    self.communication
                    .network_queue_length()
                ),

            "average_speed":
                float(
                    self.communication
                    .network_average_speed()
                )

        }

    # =====================================================
    # NETWORK SUMMARY
    # =====================================================

    def network_summary(self):

        summary = {}

        for agent_id in EnvironmentConfig.TRAFFIC_LIGHT_IDS:

            summary[agent_id] = (
                self.agents[agent_id]
                .summary()
            )

        summary["network"] = (
            self.network_statistics()
        )

        return summary

    # =====================================================
    # COMMUNICATION SUMMARY
    # =====================================================

    def communication_summary(self):

        return (
            self.communication
            .communication_summary()
        )

    # =====================================================
    # TERMINATION
    # =====================================================

    def terminate(self):
        """
        Terminate all agents.
        """

        for agent in self.agents.values():

            agent.terminate()

    # =====================================================
    # REPRESENTATION
    # =====================================================

    def __repr__(self):

        return (

            "MARLController("
            f"agents={list(self.agents.keys())}, "
            f"number_of_agents={len(self.agents)})"

        )