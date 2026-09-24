"""
=========================================================
Traffic MARL Project

Module:
Traffic Agent

Description:
Represents one intelligent traffic-light agent
inside the Multi-Agent Reinforcement Learning
system.

Each intersection owns one autonomous agent.

Observation architecture:

    Local Observation       = 15
    Neighbour Observation   = 15
    Graph Observation       = 30
    Global Observation      = 120

Week:
5

Task:
5.4

Author:
Mohammed Piyarji
=========================================================
"""

from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Optional

import numpy as np


# =========================================================
# AGENT STATISTICS
# =========================================================

@dataclass
class AgentStatistics:
    """
    Runtime statistics for one traffic-light agent.
    """

    # -----------------------------------------------------
    # Reward
    # -----------------------------------------------------

    total_reward: float = 0.0

    episode_reward: float = 0.0

    episode_steps: int = 0

    # -----------------------------------------------------
    # Decisions
    # -----------------------------------------------------

    decisions: int = 0

    phase_changes: int = 0

    violations: int = 0

    # -----------------------------------------------------
    # Traffic metrics
    # -----------------------------------------------------

    vehicle_count: float = 0.0

    waiting_time: float = 0.0

    queue_length: float = 0.0

    average_speed: float = 0.0

    stopped_vehicles: float = 0.0

    # -----------------------------------------------------
    # Directional queues
    # -----------------------------------------------------

    north_queue: float = 0.0

    east_queue: float = 0.0

    south_queue: float = 0.0

    west_queue: float = 0.0

    # -----------------------------------------------------
    # Additional traffic information
    # -----------------------------------------------------

    lane_density: float = 0.0

    phase: float = 0.0

    stopped_ratio: float = 0.0

    # -----------------------------------------------------
    # Cooperative reward
    # -----------------------------------------------------

    shared_reward: float = 0.0


# =========================================================
# TRAFFIC AGENT
# =========================================================

class TrafficAgent:

    """
    One intelligent traffic-light agent.

    Example:

        Agent A
        Agent B
        Agent C
        Agent D
    """

    # =====================================================
    # INITIALIZATION
    # =====================================================

    def __init__(
        self,
        agent_id: str,
        neighbours: List[str]
    ):

        self.agent_id = agent_id

        # Make a copy so external modifications
        # do not unexpectedly change the agent.

        self.neighbours = list(
            neighbours
        )

        self.statistics = (
            AgentStatistics()
        )

        # -------------------------------------------------
        # State
        # -------------------------------------------------

        self.current_state = None

        self.previous_state = None

        # -------------------------------------------------
        # Action
        # -------------------------------------------------

        self.current_action = 0

        self.previous_action = 0

        # -------------------------------------------------
        # Reward
        # -------------------------------------------------

        self.current_reward = 0.0

        self.previous_reward = 0.0

        self.shared_reward = 0.0

        # -------------------------------------------------
        # Neighbour information
        # -------------------------------------------------

        self.neighbour_information = {}

        self.communication_log = []

        # -------------------------------------------------
        # Episode
        # -------------------------------------------------

        self.done = False

        self.info = {}

    # =====================================================
    # RESET
    # =====================================================

    def reset(self):
        """
        Reset the agent for a new episode.
        """

        self.current_state = None

        self.previous_state = None

        self.current_action = 0

        self.previous_action = 0

        self.current_reward = 0.0

        self.previous_reward = 0.0

        self.shared_reward = 0.0

        self.neighbour_information = {}

        self.communication_log = []

        self.done = False

        self.info = {}

        self.statistics = (
            AgentStatistics()
        )

    # =====================================================
    # STATE
    # =====================================================

    def set_state(
        self,
        observation: np.ndarray
    ):
        """
        Store the latest observation.

        The observation can be:

            Local observation = 15

        or:

            Graph observation = 30
        """

        if observation is None:

            self.previous_state = (
                self.current_state
            )

            self.current_state = None

            return

        observation = np.asarray(
            observation,
            dtype=np.float32
        ).flatten()

        # -------------------------------------------------
        # NaN / Inf protection
        # -------------------------------------------------

        observation = np.nan_to_num(

            observation,

            nan=0.0,

            posinf=0.0,

            neginf=0.0

        )

        self.previous_state = (
            self.current_state
        )

        self.current_state = (
            observation
        )

    def get_state(self):
        """
        Return the latest state.
        """

        return self.current_state

    # =====================================================
    # ACTION
    # =====================================================

    def set_action(
        self,
        action: int
    ):
        """
        Store the latest action.
        """

        self.previous_action = (
            self.current_action
        )

        self.current_action = int(
            action
        )

        self.statistics.decisions += 1

    def get_action(self):
        """
        Return the current action.
        """

        return self.current_action

    # =====================================================
    # REWARD
    # =====================================================

    def set_reward(
        self,
        reward: float
    ):
        """
        Store the latest local reward.
        """

        self.previous_reward = (
            self.current_reward
        )

        self.current_reward = float(
            reward
        )

        self.statistics.total_reward += (
            float(reward)
        )

        self.statistics.episode_reward += (
            float(reward)
        )

        self.statistics.episode_steps += 1

    def get_reward(self):
        """
        Return current local reward.
        """

        return self.current_reward

    # =====================================================
    # SHARED REWARD
    # =====================================================

    def set_shared_reward(
        self,
        reward: float
    ):
        """
        Store cooperative/shared reward.
        """

        self.shared_reward = float(
            reward
        )

        self.statistics.shared_reward = float(
            reward
        )

    def get_shared_reward(self):
        """
        Return cooperative reward.
        """

        return self.shared_reward

    # =====================================================
    # NEIGHBOURS
    # =====================================================

    def get_neighbours(self):
        """
        Return neighbour IDs.
        """

        return self.neighbours

    def add_neighbour(
        self,
        neighbour: str
    ):
        """
        Add a neighbour if it does not already exist.
        """

        if neighbour not in self.neighbours:

            self.neighbours.append(
                neighbour
            )

    # =====================================================
    # COMMUNICATION
    # =====================================================

    def update_neighbour_information(
        self,
        information: Dict
    ):
        """
        Store the latest neighbour communication.
        """

        if information is None:

            information = {}

        self.neighbour_information = (
            information
        )

        self.communication_log.append(
            information
        )

    def get_neighbour_information(self):
        """
        Return latest neighbour information.
        """

        return self.neighbour_information

    def communication_history(self):
        """
        Return communication history.
        """

        return self.communication_log

    # =====================================================
    # METRICS
    # =====================================================

    def update_metrics(
        self,
        waiting_time: float,
        queue_length: float,
        average_speed: float,
        vehicle_count: Optional[float] = None,
        stopped_vehicles: Optional[float] = None,
        north_queue: Optional[float] = None,
        east_queue: Optional[float] = None,
        south_queue: Optional[float] = None,
        west_queue: Optional[float] = None,
        lane_density: Optional[float] = None,
        phase: Optional[float] = None,
        stopped_ratio: Optional[float] = None
    ):
        """
        Update traffic metrics.

        The first three parameters are kept compatible
        with the existing TrafficEnvironment.

        Additional parameters support the complete
        15-feature local observation.
        """

        self.statistics.waiting_time = float(
            waiting_time
        )

        self.statistics.queue_length = float(
            queue_length
        )

        self.statistics.average_speed = float(
            average_speed
        )

        # -------------------------------------------------
        # Optional traffic metrics
        # -------------------------------------------------

        if vehicle_count is not None:

            self.statistics.vehicle_count = float(
                vehicle_count
            )

        if stopped_vehicles is not None:

            self.statistics.stopped_vehicles = float(
                stopped_vehicles
            )

        if north_queue is not None:

            self.statistics.north_queue = float(
                north_queue
            )

        if east_queue is not None:

            self.statistics.east_queue = float(
                east_queue
            )

        if south_queue is not None:

            self.statistics.south_queue = float(
                south_queue
            )

        if west_queue is not None:

            self.statistics.west_queue = float(
                west_queue
            )

        if lane_density is not None:

            self.statistics.lane_density = float(
                lane_density
            )

        if phase is not None:

            self.statistics.phase = float(
                phase
            )

        if stopped_ratio is not None:

            self.statistics.stopped_ratio = float(
                stopped_ratio
            )

    # =====================================================
    # STATE → METRICS
    # =====================================================

    def update_metrics_from_state(
        self,
        observation
    ):
        """
        Extract the 15 local traffic metrics from
        the first 15 values of the local state.

        Local observation layout:

        [0]  vehicle_count
        [1]  average_speed
        [2]  waiting_time
        [3]  stopped_vehicles
        [4]  phase
        [5]  north_queue
        [6]  east_queue
        [7]  south_queue
        [8]  west_queue
        [9]  lane_density
        [10] total_queue
        [11] max_queue
        [12] min_queue
        [13] controlled_lane_count
        [14] stopped_ratio
        """

        if observation is None:

            return

        observation = np.asarray(
            observation,
            dtype=np.float32
        ).flatten()

        if observation.size < 15:

            return

        observation = np.nan_to_num(

            observation,

            nan=0.0,

            posinf=0.0,

            neginf=0.0

        )

        self.statistics.vehicle_count = float(
            observation[0]
        )

        self.statistics.average_speed = float(
            observation[1]
        )

        self.statistics.waiting_time = float(
            observation[2]
        )

        self.statistics.stopped_vehicles = float(
            observation[3]
        )

        self.statistics.phase = float(
            observation[4]
        )

        self.statistics.north_queue = float(
            observation[5]
        )

        self.statistics.east_queue = float(
            observation[6]
        )

        self.statistics.south_queue = float(
            observation[7]
        )

        self.statistics.west_queue = float(
            observation[8]
        )

        self.statistics.lane_density = float(
            observation[9]
        )

        self.statistics.queue_length = float(

            observation[10]

        )

        self.statistics.stopped_ratio = float(
            observation[14]
        )

    # =====================================================
    # PHASE CHANGE
    # =====================================================

    def register_phase_change(self):
        """
        Register a traffic-light phase change.
        """

        self.statistics.phase_changes += 1

    # =====================================================
    # VIOLATIONS
    # =====================================================

    def register_violation(self):
        """
        Register a safety/control violation.
        """

        self.statistics.violations += 1

    # =====================================================
    # SHARED INFORMATION
    # =====================================================

    def export_status(self) -> Dict:
        """
        Export information used by neighbouring agents.

        This is the communication payload.

        It contains:

            reward
            shared reward
            action
            traffic metrics
            directional queues
            phase changes

        This allows ObservationBuilder to create
        the 15-dimensional neighbour observation.
        """

        # -------------------------------------------------
        # If a local state exists, synchronize metrics
        # -------------------------------------------------

        if self.current_state is not None:

            state = np.asarray(
                self.current_state,
                dtype=np.float32
            ).flatten()

            # A graph observation has local features
            # in its first 15 positions.

            if state.size >= 15:

                self.update_metrics_from_state(
                    state[:15]
                )

        # -------------------------------------------------
        # Export complete status
        # -------------------------------------------------

        return {

            "agent":
                self.agent_id,

            # ---------------------------------------------
            # Rewards
            # ---------------------------------------------

            "reward":
                float(
                    self.current_reward
                ),

            "shared_reward":
                float(
                    self.shared_reward
                ),

            # ---------------------------------------------
            # Action
            # ---------------------------------------------

            "action":
                int(
                    self.current_action
                ),

            # ---------------------------------------------
            # Traffic metrics
            # ---------------------------------------------

            "vehicle_count":
                float(
                    self.statistics.vehicle_count
                ),

            "waiting_time":
                float(
                    self.statistics.waiting_time
                ),

            "queue_length":
                float(
                    self.statistics.queue_length
                ),

            "average_speed":
                float(
                    self.statistics.average_speed
                ),

            "stopped_vehicles":
                float(
                    self.statistics.stopped_vehicles
                ),

            # ---------------------------------------------
            # Directional queues
            # ---------------------------------------------

            "north_queue":
                float(
                    self.statistics.north_queue
                ),

            "east_queue":
                float(
                    self.statistics.east_queue
                ),

            "south_queue":
                float(
                    self.statistics.south_queue
                ),

            "west_queue":
                float(
                    self.statistics.west_queue
                ),

            # ---------------------------------------------
            # Additional information
            # ---------------------------------------------

            "lane_density":
                float(
                    self.statistics.lane_density
                ),

            "phase":
                float(
                    self.statistics.phase
                ),

            "stopped_ratio":
                float(
                    self.statistics.stopped_ratio
                ),

            "phase_changes":
                int(
                    self.statistics.phase_changes
                ),

            "violations":
                int(
                    self.statistics.violations
                ),

        }

    # =====================================================
    # TERMINATION
    # =====================================================

    def terminate(self):
        """
        Mark agent as terminated.
        """

        self.done = True

    def is_done(self):
        """
        Check whether agent is terminated.
        """

        return self.done

    # =====================================================
    # SUMMARY
    # =====================================================

    def summary(self):
        """
        Return complete agent summary.
        """

        return {

            "agent_id":
                self.agent_id,

            "episode_reward":
                float(
                    self.statistics.episode_reward
                ),

            "total_reward":
                float(
                    self.statistics.total_reward
                ),

            "shared_reward":
                float(
                    self.statistics.shared_reward
                ),

            "steps":
                int(
                    self.statistics.episode_steps
                ),

            "decisions":
                int(
                    self.statistics.decisions
                ),

            # ---------------------------------------------
            # Traffic
            # ---------------------------------------------

            "vehicle_count":
                float(
                    self.statistics.vehicle_count
                ),

            "waiting_time":
                float(
                    self.statistics.waiting_time
                ),

            "queue_length":
                float(
                    self.statistics.queue_length
                ),

            "average_speed":
                float(
                    self.statistics.average_speed
                ),

            "stopped_vehicles":
                float(
                    self.statistics.stopped_vehicles
                ),

            # ---------------------------------------------
            # Directional queues
            # ---------------------------------------------

            "north_queue":
                float(
                    self.statistics.north_queue
                ),

            "east_queue":
                float(
                    self.statistics.east_queue
                ),

            "south_queue":
                float(
                    self.statistics.south_queue
                ),

            "west_queue":
                float(
                    self.statistics.west_queue
                ),

            "lane_density":
                float(
                    self.statistics.lane_density
                ),

            "phase":
                float(
                    self.statistics.phase
                ),

            "stopped_ratio":
                float(
                    self.statistics.stopped_ratio
                ),

            # ---------------------------------------------
            # Control
            # ---------------------------------------------

            "phase_changes":
                int(
                    self.statistics.phase_changes
                ),

            "violations":
                int(
                    self.statistics.violations
                ),

            # ---------------------------------------------
            # Network
            # ---------------------------------------------

            "neighbours":
                list(
                    self.neighbours
                ),

            "done":
                bool(
                    self.done
                )

        }