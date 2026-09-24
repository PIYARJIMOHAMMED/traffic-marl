"""
=========================================================
Traffic MARL Project

Module:
Multi-Agent Manager

Description:
Manages all traffic-light agents inside the
four-intersection SUMO network.

Each intersection has

• Observation Builder

• Traffic Controller

The environment communicates only with this
manager instead of directly controlling every
intersection.

Week:
5

Task:
5.2

Author:
Mohammed Piyarji
=========================================================
"""

from typing import Dict
from typing import List

import numpy as np

from rl.environments.config import EnvironmentConfig
from rl.environments.observation import ObservationBuilder
from rl.environments.traffic_controller import TrafficController


class MultiAgentManager:
    """
    Multi-agent manager.

    Owns one controller and one observation
    builder for every traffic light.

    A

    B

    C

    D
    """

    def __init__(self):

        self.agent_ids = EnvironmentConfig.TRAFFIC_LIGHT_IDS

        self.controllers = {}

        self.observation_builders = {}

        self._create_agents()

    # =====================================================
    # Create Agents
    # =====================================================

    def _create_agents(self):

        """
        Creates one controller and one
        observation builder per traffic light.
        """

        for traffic_light in self.agent_ids:

            self.controllers[traffic_light] = (

                TrafficController(

                    traffic_light

                )

            )

            self.observation_builders[traffic_light] = (

                ObservationBuilder(

                    traffic_light

                )

            )

    # =====================================================
    # Reset
    # =====================================================

    def reset(self):

        """
        Reset every controller.
        """

        self.controllers.clear()

        self.observation_builders.clear()

        self._create_agents()

    # =====================================================
    # Controller
    # =====================================================

    def controller(

        self,

        traffic_light_id: str

    ) -> TrafficController:

        return self.controllers[traffic_light_id]

    # =====================================================
    # Observation Builder
    # =====================================================

    def observation_builder(

        self,

        traffic_light_id: str

    ) -> ObservationBuilder:

        return self.observation_builders[traffic_light_id]

    # =====================================================
    # Observations
    # =====================================================

    def get_observations(self) -> Dict[str, np.ndarray]:

        """
        Returns

        {

            "A": observation,

            "B": observation,

            "C": observation,

            "D": observation

        }
        """

        observations = {}

        for traffic_light in self.agent_ids:

            observations[traffic_light] = (

                self.observation_builders[

                    traffic_light

                ].get_observation()

            )

        return observations

    # =====================================================
    # Apply Actions
    # =====================================================

    def apply_actions(

        self,

        actions: Dict[str, int]

    ):

        """
        Applies one action for each agent.

        Example

        {

            "A":0,

            "B":1,

            "C":0,

            "D":1

        }
        """

        for traffic_light, action in actions.items():

            self.controllers[

                traffic_light

            ].apply_action(action)

    # =====================================================
    # Update Timers
    # =====================================================

    def update(self):

        """
        Updates every controller once
        per SUMO step.
        """

        for controller in self.controllers.values():

            controller.update()

    # =====================================================
    # Statistics
    # =====================================================

    def get_statistics(self):

        statistics = {}

        for traffic_light in self.agent_ids:

            statistics[traffic_light] = (

                self.controllers[

                    traffic_light

                ].statistics

            )

        return statistics

    # =====================================================
    # Agent IDs
    # =====================================================

    def get_agent_ids(self) -> List[str]:

        return self.agent_ids

    # =====================================================
    # Number of Agents
    # =====================================================

    def number_of_agents(self) -> int:

        return len(self.agent_ids)