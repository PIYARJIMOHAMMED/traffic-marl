"""
=========================================================
Traffic MARL Project

Module:
Neighbour-aware Observation Builder

Description:
Builds observations for each traffic-light agent.

Architecture:

    Local Observation
        15 features

    Neighbour Observation
        15 features

    Graph Observation
        30 features

    Global Observation
        4 x 30 = 120 features

The important rule is:

    LOCAL -> 15
    NEIGHBOUR -> 15
    GRAPH -> 30

A graph observation must NEVER be passed back into
get_local_features().

Week:
5

Task:
5.4

Author:
Mohammed Piyarji
=========================================================
"""

from typing import Dict, List

import numpy as np
import traci

from rl.environments.config import EnvironmentConfig


class ObservationBuilder:

    """
    Builds the observation for ONE traffic-light agent.

    Architecture:

        Local       = 15
        Neighbour   = 15
        Graph       = 30
    """

    # =====================================================
    # FIXED DIMENSIONS
    # =====================================================

    LOCAL_SIZE = 15

    NEIGHBOUR_SIZE = 15

    GRAPH_SIZE = (
        LOCAL_SIZE
        + NEIGHBOUR_SIZE
    )

    # =====================================================
    # INITIALIZATION
    # =====================================================

    def __init__(
        self,
        traffic_light_id: str
    ):

        self.traffic_light_id = (
            traffic_light_id
        )

        if (
            EnvironmentConfig.OBSERVATION_SIZE
            != self.LOCAL_SIZE
        ):

            raise ValueError(
                "Observation dimension configuration "
                "is inconsistent.\n"
                f"EnvironmentConfig.OBSERVATION_SIZE="
                f"{EnvironmentConfig.OBSERVATION_SIZE}\n"
                f"ObservationBuilder.LOCAL_SIZE="
                f"{self.LOCAL_SIZE}"
            )

    # =====================================================
    # CONTROLLED LANES
    # =====================================================

    def get_controlled_lanes(self) -> List[str]:

        try:

            lanes = (
                traci.trafficlight
                .getControlledLanes(
                    self.traffic_light_id
                )
            )

        except traci.TraCIException:

            return []

        # Remove duplicate lanes while
        # preserving order.

        return list(
            dict.fromkeys(lanes)
        )

    # =====================================================
    # CONTROLLED VEHICLES
    # =====================================================

    def get_controlled_vehicles(self) -> List[str]:

        vehicles = []

        for lane in (
            self.get_controlled_lanes()
        ):

            try:

                vehicle_ids = (
                    traci.lane
                    .getLastStepVehicleIDs(
                        lane
                    )
                )

                vehicles.extend(
                    vehicle_ids
                )

            except traci.TraCIException:

                continue

        # Remove duplicates.

        return list(
            dict.fromkeys(
                vehicles
            )
        )

    # =====================================================
    # VEHICLE COUNT
    # =====================================================

    def get_vehicle_count(self) -> float:

        return float(
            len(
                self.get_controlled_vehicles()
            )
        )

    # =====================================================
    # AVERAGE SPEED
    # =====================================================

    def get_average_speed(self) -> float:

        vehicles = (
            self.get_controlled_vehicles()
        )

        if not vehicles:

            return 0.0

        speeds = []

        for vehicle in vehicles:

            try:

                speeds.append(
                    float(
                        traci.vehicle.getSpeed(
                            vehicle
                        )
                    )
                )

            except traci.TraCIException:

                continue

        if not speeds:

            return 0.0

        return float(
            np.mean(
                speeds
            )
        )

    # =====================================================
    # WAITING TIME
    # =====================================================

    def get_waiting_time(self) -> float:

        vehicles = (
            self.get_controlled_vehicles()
        )

        if not vehicles:

            return 0.0

        waiting = []

        for vehicle in vehicles:

            try:

                waiting.append(
                    float(
                        traci.vehicle.getWaitingTime(
                            vehicle
                        )
                    )
                )

            except traci.TraCIException:

                continue

        if not waiting:

            return 0.0

        return float(
            np.sum(
                waiting
            )
        )

    # =====================================================
    # STOPPED VEHICLES
    # =====================================================

    def get_stopped_vehicles(self) -> float:

        stopped = 0

        vehicles = (
            self.get_controlled_vehicles()
        )

        for vehicle in vehicles:

            try:

                speed = float(
                    traci.vehicle.getSpeed(
                        vehicle
                    )
                )

                if speed < 0.1:

                    stopped += 1

            except traci.TraCIException:

                continue

        return float(
            stopped
        )

    # =====================================================
    # PHASE
    # =====================================================

    def get_phase(self) -> float:

        try:

            return float(
                traci.trafficlight.getPhase(
                    self.traffic_light_id
                )
            )

        except traci.TraCIException:

            return 0.0

    # =====================================================
    # QUEUE LENGTHS
    # =====================================================

    def get_queue_lengths(self):

        north = 0.0
        east = 0.0
        south = 0.0
        west = 0.0

        for lane in (
            self.get_controlled_lanes()
        ):

            try:

                halted = float(
                    traci.lane
                    .getLastStepHaltingNumber(
                        lane
                    )
                )

            except traci.TraCIException:

                continue

            lane_upper = (
                str(lane).upper()
            )

            # -------------------------------------------------
            # IMPORTANT
            #
            # These prefixes follow your existing project
            # lane convention.
            # -------------------------------------------------

            if lane_upper.startswith(
                ("N", "A")
            ):

                north += halted

            elif lane_upper.startswith(
                ("E", "B")
            ):

                east += halted

            elif lane_upper.startswith(
                ("S", "C")
            ):

                south += halted

            elif lane_upper.startswith(
                ("W", "D")
            ):

                west += halted

        return (
            float(north),
            float(east),
            float(south),
            float(west)
        )

    # =====================================================
    # LANE DENSITY
    # =====================================================

    def get_lane_density(self) -> float:

        lanes = (
            self.get_controlled_lanes()
        )

        if not lanes:

            return 0.0

        density = []

        for lane in lanes:

            try:

                density.append(
                    float(
                        traci.lane
                        .getLastStepOccupancy(
                            lane
                        )
                    )
                )

            except traci.TraCIException:

                continue

        if not density:

            return 0.0

        return float(
            np.mean(
                density
            )
        )

    # =====================================================
    # TRAFFIC LIGHT TIMER
    # =====================================================

    def _get_controller_features(self):

        """
        Obtain controller-related features.

        These features are deliberately kept independent
        of TrafficController so ObservationBuilder does
        not create circular dependencies.

        Returns:

            green_timer
            yellow_timer
            current_action
            in_yellow
            safety/reserved
        """

        # -------------------------------------------------
        # Default values
        # -------------------------------------------------

        green_timer = 0.0
        yellow_timer = 0.0
        current_action = 0.0
        in_yellow = 0.0
        safety_feature = 0.0

        # -------------------------------------------------
        # Read actual SUMO phase where possible.
        # -------------------------------------------------

        try:

            phase = int(
                traci.trafficlight.getPhase(
                    self.traffic_light_id
                )
            )

            if phase in (
                EnvironmentConfig.GREEN_PHASE_NS,
                EnvironmentConfig.YELLOW_PHASE_NS,
            ):

                current_action = float(
                    EnvironmentConfig.ACTION_NORTH_SOUTH
                )

            elif phase in (
                EnvironmentConfig.GREEN_PHASE_EW,
                EnvironmentConfig.YELLOW_PHASE_EW,
            ):

                current_action = float(
                    EnvironmentConfig.ACTION_EAST_WEST
                )

            if phase in (
                EnvironmentConfig.YELLOW_PHASE_NS,
                EnvironmentConfig.YELLOW_PHASE_EW,
            ):

                in_yellow = 1.0

        except traci.TraCIException:

            pass

        # -------------------------------------------------
        # Safety feature
        #
        # Reserved for future controller state.
        # Kept at zero for now so the observation
        # dimension remains stable.
        # -------------------------------------------------

        safety_feature = 0.0

        return (
            green_timer,
            yellow_timer,
            current_action,
            in_yellow,
            safety_feature
        )

    # =====================================================
    # LOCAL FEATURES
    # =====================================================

    def get_local_features(self) -> np.ndarray:

        """
        Build EXACTLY 15 local features.

        Feature layout:

        0   vehicle_count
        1   average_speed
        2   waiting_time
        3   stopped_vehicles
        4   phase
        5   north_queue
        6   east_queue
        7   south_queue
        8   west_queue
        9   lane_density
        10  green_timer
        11  yellow_timer
        12  current_action
        13  in_yellow
        14  safety/reserved
        """

        (
            north,
            east,
            south,
            west
        ) = self.get_queue_lengths()

        (
            green_timer,
            yellow_timer,
            current_action,
            in_yellow,
            safety_feature
        ) = self._get_controller_features()

        features = np.array(

            [

                self.get_vehicle_count(),

                self.get_average_speed(),

                self.get_waiting_time(),

                self.get_stopped_vehicles(),

                self.get_phase(),

                north,

                east,

                south,

                west,

                self.get_lane_density(),

                green_timer,

                yellow_timer,

                current_action,

                in_yellow,

                safety_feature,

            ],

            dtype=np.float32

        )

        # -------------------------------------------------
        # HARD VALIDATION
        # -------------------------------------------------

        if features.size != self.LOCAL_SIZE:

            raise ValueError(

                "\n"
                "LOCAL OBSERVATION DIMENSION ERROR\n"
                f"Agent: {self.traffic_light_id}\n"
                f"Expected: {self.LOCAL_SIZE}\n"
                f"Received: {features.size}\n"

            )

        # -------------------------------------------------
        # Numerical safety
        # -------------------------------------------------

        features = np.nan_to_num(

            features,

            nan=0.0,

            posinf=100000.0,

            neginf=-100000.0

        ).astype(
            np.float32
        )

        return features

    # =====================================================
    # NEIGHBOUR FEATURES
    # =====================================================

    def get_neighbour_features(

        self,

        neighbour_information: Dict

    ) -> np.ndarray:

        """
        Build EXACTLY 15 aggregated neighbour features.

        Features:

        0   mean queue
        1   mean speed
        2   mean waiting
        3   mean total queue
        4   mean reward
        5   mean action
        6   mean phase changes
        7   neighbour count
        8   max queue
        9   min speed
        10  max waiting
        11  mean shared reward
        12  neighbour count
        13  NS action ratio
        14  EW action ratio

        IMPORTANT:

        This function consumes neighbour STATUS
        dictionaries.

        It does NOT consume graph observations.
        """

        if not neighbour_information:

            return np.zeros(

                self.NEIGHBOUR_SIZE,

                dtype=np.float32

            )

        queue_values = []
        speed_values = []
        waiting_values = []
        reward_values = []
        action_values = []
        phase_change_values = []
        shared_reward_values = []

        for info in (
            neighbour_information.values()
        ):

            if not isinstance(
                info,
                dict
            ):

                continue

            # ---------------------------------------------
            # Queue
            # ---------------------------------------------

            queue_value = float(
                info.get(
                    "queue_length",
                    0.0
                )
            )

            queue_values.append(
                queue_value
            )

            # ---------------------------------------------
            # Speed
            # ---------------------------------------------

            speed_value = float(
                info.get(
                    "average_speed",
                    0.0
                )
            )

            speed_values.append(
                speed_value
            )

            # ---------------------------------------------
            # Waiting
            # ---------------------------------------------

            waiting_value = float(
                info.get(
                    "waiting_time",
                    0.0
                )
            )

            waiting_values.append(
                waiting_value
            )

            # ---------------------------------------------
            # Reward
            # ---------------------------------------------

            reward_values.append(
                float(
                    info.get(
                        "reward",
                        0.0
                    )
                )
            )

            # ---------------------------------------------
            # Action
            # ---------------------------------------------

            action_values.append(
                float(
                    info.get(
                        "action",
                        0
                    )
                )
            )

            # ---------------------------------------------
            # Phase changes
            # ---------------------------------------------

            phase_change_values.append(
                float(
                    info.get(
                        "phase_changes",
                        0
                    )
                )
            )

            # ---------------------------------------------
            # Shared reward
            # ---------------------------------------------

            shared_reward_values.append(
                float(
                    info.get(
                        "shared_reward",
                        0.0
                    )
                )
            )

        # -------------------------------------------------
        # No usable neighbours
        # -------------------------------------------------

        if not queue_values:

            return np.zeros(

                self.NEIGHBOUR_SIZE,

                dtype=np.float32

            )

        # -------------------------------------------------
        # Arrays
        # -------------------------------------------------

        queues = np.asarray(
            queue_values,
            dtype=np.float32
        )

        speeds = np.asarray(
            speed_values,
            dtype=np.float32
        )

        waiting = np.asarray(
            waiting_values,
            dtype=np.float32
        )

        rewards = np.asarray(
            reward_values,
            dtype=np.float32
        )

        actions = np.asarray(
            action_values,
            dtype=np.float32
        )

        phase_changes = np.asarray(
            phase_change_values,
            dtype=np.float32
        )

        shared_rewards = np.asarray(
            shared_reward_values,
            dtype=np.float32
        )

        neighbour_count = float(
            len(queues)
        )

        # -------------------------------------------------
        # Action ratios
        # -------------------------------------------------

        ns_ratio = float(
            np.mean(
                actions == EnvironmentConfig.ACTION_NORTH_SOUTH
            )
        )

        ew_ratio = float(
            np.mean(
                actions == EnvironmentConfig.ACTION_EAST_WEST
            )
        )

        # -------------------------------------------------
        # 15-dimensional neighbour vector
        # -------------------------------------------------

        features = np.array(

            [

                np.mean(queues),

                np.mean(speeds),

                np.mean(waiting),

                np.mean(queues),

                np.mean(rewards),

                np.mean(actions),

                np.mean(phase_changes),

                neighbour_count,

                np.max(queues),

                np.min(speeds),

                np.max(waiting),

                np.mean(shared_rewards),

                neighbour_count,

                ns_ratio,

                ew_ratio,

            ],

            dtype=np.float32

        )

        # -------------------------------------------------
        # Numerical safety
        # -------------------------------------------------

        features = np.nan_to_num(

            features,

            nan=0.0,

            posinf=100000.0,

            neginf=-100000.0

        ).astype(
            np.float32
        )

        # -------------------------------------------------
        # HARD VALIDATION
        # -------------------------------------------------

        if features.size != self.NEIGHBOUR_SIZE:

            raise ValueError(

                "\n"
                "NEIGHBOUR OBSERVATION DIMENSION ERROR\n"
                f"Agent: {self.traffic_light_id}\n"
                f"Expected: {self.NEIGHBOUR_SIZE}\n"
                f"Received: {features.size}\n"

            )

        return features

    # =====================================================
    # GRAPH OBSERVATION
    # =====================================================

    def get_graph_observation(

        self,

        neighbour_information: Dict

    ) -> np.ndarray:

        """
        Build EXACTLY 30 graph features.

        Local:
            15

        Neighbour:
            15

        Graph:
            30
        """

        # -------------------------------------------------
        # Build LOCAL only from SUMO
        # -------------------------------------------------

        local = np.asarray(

            self.get_local_features(),

            dtype=np.float32

        ).flatten()

        # -------------------------------------------------
        # Build NEIGHBOUR only from status dictionaries
        # -------------------------------------------------

        neighbours = np.asarray(

            self.get_neighbour_features(

                neighbour_information

            ),

            dtype=np.float32

        ).flatten()

        # -------------------------------------------------
        # Validate local
        # -------------------------------------------------

        if local.size != self.LOCAL_SIZE:

            raise ValueError(

                "\n"
                "LOCAL COMPONENT ERROR\n"
                f"Agent: {self.traffic_light_id}\n"
                f"Expected: {self.LOCAL_SIZE}\n"
                f"Received: {local.size}\n"

            )

        # -------------------------------------------------
        # Validate neighbour
        # -------------------------------------------------

        if neighbours.size != self.NEIGHBOUR_SIZE:

            raise ValueError(

                "\n"
                "NEIGHBOUR COMPONENT ERROR\n"
                f"Agent: {self.traffic_light_id}\n"
                f"Expected: {self.NEIGHBOUR_SIZE}\n"
                f"Received: {neighbours.size}\n"

            )

        # -------------------------------------------------
        # Combine
        # -------------------------------------------------

        graph = np.concatenate(

            [
                local,
                neighbours
            ],

            axis=0

        ).astype(
            np.float32
        )

        # -------------------------------------------------
        # Final validation
        # -------------------------------------------------

        if graph.size != self.GRAPH_SIZE:

            raise ValueError(

                "\n"
                "GRAPH OBSERVATION DIMENSION ERROR\n"
                f"Agent: {self.traffic_light_id}\n"
                f"Expected: {self.GRAPH_SIZE}\n"
                f"Received: {graph.size}\n"

            )

        # -------------------------------------------------
        # Numerical safety
        # -------------------------------------------------

        graph = np.nan_to_num(

            graph,

            nan=0.0,

            posinf=100000.0,

            neginf=-100000.0

        ).astype(
            np.float32
        )

        return graph

    # =====================================================
    # LOCAL OBSERVATION
    # =====================================================

    def get_observation(self):

        """
        Return local observation only.

        Size:
            15
        """

        return self.get_local_features()

    # =====================================================
    # OBSERVATION DICTIONARY
    # =====================================================

    def get_observation_dict(self) -> Dict:

        (
            north,
            east,
            south,
            west
        ) = self.get_queue_lengths()

        return {

            "vehicle_count":
                self.get_vehicle_count(),

            "average_speed":
                self.get_average_speed(),

            "waiting_time":
                self.get_waiting_time(),

            "stopped_vehicles":
                self.get_stopped_vehicles(),

            "phase":
                self.get_phase(),

            "north_queue":
                north,

            "east_queue":
                east,

            "south_queue":
                south,

            "west_queue":
                west,

            "lane_density":
                self.get_lane_density(),

        }


# =========================================================
# DIRECT TEST
# =========================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("Observation Builder")
    print("=" * 70)
    print()

    print(
        "Local size     :",
        ObservationBuilder.LOCAL_SIZE
    )

    print(
        "Neighbour size :",
        ObservationBuilder.NEIGHBOUR_SIZE
    )

    print(
        "Graph size     :",
        ObservationBuilder.GRAPH_SIZE
    )

    print(
        "Agents         :",
        EnvironmentConfig.NUMBER_OF_AGENTS
    )

    print(
        "Global size    :",
        ObservationBuilder.GRAPH_SIZE
        * EnvironmentConfig.NUMBER_OF_AGENTS
    )

    print()
    print("=" * 70)

