"""
=========================================================
Traffic MARL Project
=========================================================

Module:
    Scenario-aware Multi-Agent Gymnasium Environment

Description:
    Gymnasium Environment for a
    4-intersection SUMO traffic network.

Supports:
    • TrafficAgent
    • MARLController
    • NeighbourManager
    • ObservationBuilder
    • TrafficController
    • Cooperative MARL reward
    • Scenario-based simulation
    • Reproducible evaluation
    • Graph-based neighbour communication

MARL Observation Architecture:

    Local observation:
        15 features

    Neighbour observation:
        15 features

    Graph observation:
        15 + 15 = 30 features

    Four traffic-light agents:

        4 × 30 = 120 global features

IMPORTANT:

    MARL agent state ALWAYS remains 15.

    Graph observation is a temporary 30-dimensional
    observation used by the Gymnasium environment.

    NEVER store the 30-dimensional graph observation
    back into TrafficAgent.current_state.

Week:
    5

Task:
    5.6

Author:
    Mohammed Piyarji
=========================================================
"""

from pathlib import Path

import gymnasium as gym
from gymnasium import spaces

import numpy as np
import traci

from rl.controllers.marl_controller import MARLController
from rl.environments.config import EnvironmentConfig
from rl.environments.observation import ObservationBuilder
from rl.environments.traffic_controller import TrafficController
from rl.rewards.reward import RewardFunction
from rl.scenarios.scenario_manager import ScenarioManager


class TrafficEnvironment(gym.Env):

    """
    Scenario-aware Multi-Agent Traffic Environment.

    Four traffic-light agents:

        A
        B
        C
        D

    Observation architecture:

        Local:
            15

        Neighbour:
            15

        Graph:
            30

        Global:
            30 × 4 = 120

    PPO receives:

        Box(..., shape=(120,), dtype=float32)

    PPO actions:

        MultiDiscrete([2, 2, 2, 2])
    """

    metadata = {
        "render_modes": [
            "human"
        ]
    }

    # =====================================================
    # SCENARIOS
    # =====================================================

    SCENARIOS = [

        "baseline",

        "low_demand",

        "balanced",

        "morning_rush",

        "evening_rush",

        "high_demand",

        "north_south_heavy",

        "east_west_heavy",

        "random",

        "stress_test"

    ]

    # =====================================================
    # INITIALIZATION
    # =====================================================

    def __init__(
        self,
        scenario_name="baseline",
        render_mode=None
    ):

        super().__init__()

        # -------------------------------------------------
        # Scenario manager
        # -------------------------------------------------

        self.scenario_manager = ScenarioManager()

        # -------------------------------------------------
        # Scenario validation
        # -------------------------------------------------

        if scenario_name not in self.SCENARIOS:

            raise ValueError(

                f"Unknown scenario: {scenario_name}\n\n"

                f"Available scenarios:\n"

                f"{self.SCENARIOS}"

            )

        self.scenario_name = scenario_name

        self.render_mode = render_mode

        # -------------------------------------------------
        # Reward
        # -------------------------------------------------

        self.reward_function = RewardFunction()

        # -------------------------------------------------
        # MARL controller
        # -------------------------------------------------

        self.marl = None

        # -------------------------------------------------
        # Traffic controllers
        # -------------------------------------------------

        self.controllers = {}

        # -------------------------------------------------
        # Observation builders
        # -------------------------------------------------

        self.observers = {}

        # -------------------------------------------------
        # Episode state
        # -------------------------------------------------

        self.step_count = 0

        self.is_initialized = False

        # -------------------------------------------------
        # Episode seed
        # -------------------------------------------------

        # IMPORTANT:
        # This seed is passed to SUMO during reset().
        #
        # This allows evaluation runs such as:
        #
        #     seed = 42
        #     seed = 43
        #     seed = 44
        #
        # to create reproducible but different SUMO
        # simulations when the scenario configuration
        # supports seeded randomness.

        self._episode_seed = None

        # =================================================
        # OBSERVATION DIMENSIONS
        # =================================================

        # Local:
        #
        #     15
        #
        # Neighbour:
        #
        #     15
        #
        # Graph:
        #
        #     15 + 15 = 30
        #
        # Global:
        #
        #     30 × NUMBER_OF_AGENTS
        #
        # =================================================

        self.local_observation_size = (

            ObservationBuilder.LOCAL_SIZE

        )

        self.neighbour_observation_size = (

            ObservationBuilder.NEIGHBOUR_SIZE

        )

        self.graph_observation_size = (

            ObservationBuilder.GRAPH_SIZE

        )

        self.observation_size = (

            self.graph_observation_size

            *

            EnvironmentConfig.NUMBER_OF_AGENTS

        )

        # -------------------------------------------------
        # Validate architecture
        # -------------------------------------------------

        if self.local_observation_size != 15:

            raise ValueError(

                "Local observation must contain "
                "exactly 15 features. "

                f"Received: "
                f"{self.local_observation_size}"

            )

        if self.neighbour_observation_size != 15:

            raise ValueError(

                "Neighbour observation must contain "
                "exactly 15 features. "

                f"Received: "
                f"{self.neighbour_observation_size}"

            )

        if self.graph_observation_size != 30:

            raise ValueError(

                "Graph observation must contain "
                "exactly 30 features. "

                f"Received: "
                f"{self.graph_observation_size}"

            )

        expected_global_size = (

            30

            *

            EnvironmentConfig.NUMBER_OF_AGENTS

        )

        if self.observation_size != expected_global_size:

            raise ValueError(

                "Global observation dimension error. "

                f"Expected: "
                f"{expected_global_size}, "

                f"Received: "
                f"{self.observation_size}"

            )

        # =================================================
        # IMPORTANT RUNTIME STORAGE
        # =================================================

        # The latest graph observations are stored here.
        #
        # Example:
        #
        # {
        #     "A": np.ndarray(30,),
        #     "B": np.ndarray(30,),
        #     "C": np.ndarray(30,),
        #     "D": np.ndarray(30,)
        # }
        #
        # They are NOT written into
        # MARLController agent states.

        self.last_graph_observations = {}

        # -------------------------------------------------
        # Observation space
        # -------------------------------------------------

        self.observation_space = spaces.Box(

            low=-100000,

            high=100000,

            shape=(

                self.observation_size,

            ),

            dtype=np.float32

        )

        # -------------------------------------------------
        # Action space
        # -------------------------------------------------

        self.action_space = spaces.MultiDiscrete(

            [

                EnvironmentConfig.NUMBER_OF_ACTIONS

            ]

            *

            EnvironmentConfig.NUMBER_OF_AGENTS

        )

    # =====================================================
    # SCENARIO DIRECTORY
    # =====================================================

    def _scenario_directory(self):

        """
        Return:

            simulation/scenarios/<scenario>
        """

        scenario_directory = (

            EnvironmentConfig.ROOT_DIR

            / "simulation"

            / "scenarios"

            / self.scenario_name

        )

        return Path(

            scenario_directory

        )

    # =====================================================
    # SCENARIO SUMO CONFIG
    # =====================================================

    def _scenario_config(self):

        """
        Return the SUMO configuration for the
        currently selected scenario.

        Example:

            simulation/scenarios/baseline/simulation.sumocfg
        """

        scenario_directory = (

            self._scenario_directory()

        )

        sumo_config = (

            scenario_directory

            / "simulation.sumocfg"

        )

        if not sumo_config.exists():

            raise FileNotFoundError(

                "Scenario SUMO configuration "
                "not found.\n\n"

                f"Scenario: "
                f"{self.scenario_name}\n"

                f"Expected file:\n"
                f"{sumo_config}\n\n"

                "Run the scenario manager first:\n"

                "python -m "
                "rl.scenarios.scenario_manager"

            )

        return sumo_config

    # =====================================================
    # SCENARIO MANAGEMENT
    # =====================================================

    def set_scenario(
        self,
        scenario_name: str
    ):

        """
        Change the active scenario.

        The new scenario is loaded when reset()
        is called.
        """

        if scenario_name not in self.SCENARIOS:

            raise ValueError(

                f"Unknown scenario: "
                f"{scenario_name}\n\n"

                f"Available scenarios:\n"
                f"{self.SCENARIOS}"

            )

        self.scenario_name = scenario_name

    # =====================================================
    # CURRENT SCENARIO
    # =====================================================

    def get_scenario(self):

        return self.scenario_name

    # =====================================================
    # AVAILABLE SCENARIOS
    # =====================================================

    def available_scenarios(self):

        return list(
            self.SCENARIOS
        )

    # =====================================================
    # RESET
    # =====================================================

    def reset(
        self,
        seed=None,
        options=None
    ):

        # -------------------------------------------------
        # Gymnasium seed
        # -------------------------------------------------

        super().reset(
            seed=seed
        )

        # -------------------------------------------------
        # Store episode seed
        # -------------------------------------------------

        # IMPORTANT:
        # Gymnasium's seed alone does not automatically
        # change the random seed used by SUMO.
        #
        # Store the seed so it can be explicitly passed
        # to SUMO using the --seed argument below.

        self._episode_seed = seed

        # -------------------------------------------------
        # Close previous SUMO
        # -------------------------------------------------

        if traci.isLoaded():

            traci.close()

        # -------------------------------------------------
        # Reset episode state
        # -------------------------------------------------

        self.step_count = 0

        self.is_initialized = False

        self.last_graph_observations = {}

        # -------------------------------------------------
        # Scenario configuration
        # -------------------------------------------------

        sumo_config = (

            self._scenario_config()

        )

        print()

        print(

            f"Starting SUMO scenario: "
            f"{self.scenario_name}"

        )

        print(

            f"Configuration: "
            f"{sumo_config}"

        )

        if self._episode_seed is not None:

            print(

                f"SUMO episode seed: "
                f"{self._episode_seed}"

            )

        else:

            print(
                "SUMO episode seed: default"
            )

        # -------------------------------------------------
        # SUMO binary
        # -------------------------------------------------

        sumo_binary = (

            EnvironmentConfig.SUMO_BINARY

        )

        # -------------------------------------------------
        # SUMO command
        # -------------------------------------------------

        sumo_command = [

            sumo_binary,

            "-c",

            str(
                sumo_config
            ),

            "--start",

            "--quit-on-end"

        ]

        # -------------------------------------------------
        # IMPORTANT SEED FIX
        # -------------------------------------------------

        # Pass the Gymnasium episode seed directly to SUMO.
        #
        # Example:
        #
        #     reset(seed=42)
        #
        # becomes:
        #
        #     sumo ... --seed 42
        #
        # This makes SUMO simulations reproducible for the
        # same seed while allowing different seeds to create
        # different stochastic simulations.

        if self._episode_seed is not None:

            sumo_command.extend(

                [

                    "--seed",

                    str(
                        int(
                            self._episode_seed
                        )
                    )

                ]

            )

        # -------------------------------------------------
        # Start SUMO
        # -------------------------------------------------

        traci.start(
            sumo_command
        )

        # -------------------------------------------------
        # Create MARL controller
        # -------------------------------------------------

        self.marl = MARLController()

        # -------------------------------------------------
        # Reset controllers
        # -------------------------------------------------

        self.controllers = {}

        # -------------------------------------------------
        # Reset observers
        # -------------------------------------------------

        self.observers = {}

        # -------------------------------------------------
        # Create controller + observer for every
        # traffic-light agent
        # -------------------------------------------------

        for traffic_light in (

            EnvironmentConfig
            .TRAFFIC_LIGHT_IDS

        ):

            self.controllers[
                traffic_light
            ] = TrafficController(

                traffic_light

            )

            self.observers[
                traffic_light
            ] = ObservationBuilder(

                traffic_light

            )

        # -------------------------------------------------
        # First SUMO simulation step
        # -------------------------------------------------

        traci.simulationStep()

        # -------------------------------------------------
        # Build FIRST 120-dimensional observation
        # -------------------------------------------------

        observation = (

            self._build_global_observation()

        )

        observation = np.asarray(

            observation,

            dtype=np.float32

        )

        # -------------------------------------------------
        # Validate observation
        # -------------------------------------------------

        self._validate_global_observation(

            observation

        )

        # -------------------------------------------------
        # Initialized
        # -------------------------------------------------

        self.is_initialized = True

        # -------------------------------------------------
        # Reset information
        # -------------------------------------------------

        info = {

            "scenario":
                self.scenario_name,

            "step":
                0,

            "seed":
                self._episode_seed,

            "observation_size":
                int(
                    observation.shape[0]
                ),

            "local_observation_size":
                self.local_observation_size,

            "neighbour_observation_size":
                self.neighbour_observation_size,

            "graph_observation_size":
                self.graph_observation_size,

            "traffic_lights":
                EnvironmentConfig
                .TRAFFIC_LIGHT_IDS,

            "number_of_agents":
                EnvironmentConfig
                .NUMBER_OF_AGENTS,

            "sumo_config":
                str(
                    sumo_config
                )

        }

        return (

            observation,

            info

        )

    # =====================================================
    # BUILD LOCAL OBSERVATIONS
    # =====================================================

    def _build_local_observations(self):

        """
        Build the 15-dimensional local observation
        for every traffic-light agent.

        Returns:

            {
                "A": np.ndarray(15,),
                "B": np.ndarray(15,),
                "C": np.ndarray(15,),
                "D": np.ndarray(15,)
            }

        IMPORTANT:

            These 15-dimensional states are the ONLY
            observations stored inside TrafficAgent.
        """

        local_observations = {}

        for traffic_light in (

            EnvironmentConfig
            .TRAFFIC_LIGHT_IDS

        ):

            observation = (

                self.observers[
                    traffic_light
                ].get_local_features()

            )

            observation = np.asarray(

                observation,

                dtype=np.float32

            ).flatten()

            # -------------------------------------------------
            # Validate local observation
            # -------------------------------------------------

            if observation.size != (

                self.local_observation_size

            ):

                raise ValueError(

                    "\n"
                    "========================================\n"
                    "LOCAL OBSERVATION DIMENSION ERROR\n"
                    "========================================\n"
                    f"Agent: {traffic_light}\n"
                    f"Expected: "
                    f"{self.local_observation_size}\n"
                    f"Received: "
                    f"{observation.size}\n"
                    "========================================"

                )

            local_observations[
                traffic_light
            ] = observation

        return local_observations

    # =====================================================
    # BUILD GRAPH OBSERVATIONS
    # =====================================================

    def _build_graph_observations(
        self,
        local_observations
    ):

        """
        Build exactly one 30-dimensional graph
        observation for each agent.

        IMPORTANT:

            We deliberately DO NOT use:

                self.marl.graph_observations()

        here.

        Why?

            MARLController.graph_observations()
            uses the agent's current state.

            If graph state is accidentally stored
            inside the agent, the next call becomes:

                30 local
                +
                30 neighbour
                =
                60

        Instead, this method explicitly uses:

            local 15
            +
            neighbour status 15
            =
            graph 30
        """

        graph_observations = {}

        for traffic_light in (

            EnvironmentConfig
            .TRAFFIC_LIGHT_IDS

        ):

            # -------------------------------------------------
            # Get neighbour STATUS
            # -------------------------------------------------

            neighbour_information = (

                self.marl
                .communication
                .neighbour_status(

                    traffic_light

                )

            )

            # -------------------------------------------------
            # Build graph observation directly
            # -------------------------------------------------

            graph_observation = (

                self.observers[
                    traffic_light
                ].get_graph_observation(

                    neighbour_information

                )

            )

            graph_observation = np.asarray(

                graph_observation,

                dtype=np.float32

            ).flatten()

            # -------------------------------------------------
            # Validate graph
            # -------------------------------------------------

            if graph_observation.size != (

                self.graph_observation_size

            ):

                raise ValueError(

                    "\n"
                    "========================================\n"
                    "GRAPH OBSERVATION DIMENSION ERROR\n"
                    "========================================\n"
                    f"Agent: {traffic_light}\n"
                    f"Expected: "
                    f"{self.graph_observation_size}\n"
                    f"Received: "
                    f"{graph_observation.size}\n"

                    "\n"
                    "Architecture:\n"
                    "Local = 15\n"
                    "Neighbour = 15\n"
                    "Graph = 30\n"
                    "========================================"

                )

            graph_observations[
                traffic_light
            ] = graph_observation

        # -------------------------------------------------
        # Store latest graph observations separately.
        #
        # DO NOT call:
        #
        #     self.marl.update_states(graph_observations)
        #
        # -------------------------------------------------

        self.last_graph_observations = {

            agent_id:

                observation.copy()

            for agent_id, observation

            in graph_observations.items()

        }

        return graph_observations

    # =====================================================
    # BUILD GLOBAL OBSERVATION
    # =====================================================

    def _build_global_observation(self):

        """
        Build the complete 120-dimensional PPO
        observation.

        Architecture:

            A = 30
            B = 30
            C = 30
            D = 30

            Global = 120

        CRITICAL:

            Agent state remains 15.

            Graph observation remains 30.

            Graph observations are NOT written
            back into MARLController states.
        """

        # -------------------------------------------------
        # Safety check
        # -------------------------------------------------

        if self.marl is None:

            raise RuntimeError(

                "MARLController is not initialized."

            )

        # -------------------------------------------------
        # STEP 1
        # Build local observations
        # -------------------------------------------------

        local_observations = (

            self._build_local_observations()

        )

        # -------------------------------------------------
        # STEP 2
        # Store ONLY local states in MARL
        # -------------------------------------------------

        self.marl.update_states(

            local_observations

        )

        # -------------------------------------------------
        # STEP 3
        # Build graph observations
        # -------------------------------------------------

        graph_observations = (

            self._build_graph_observations(

                local_observations

            )

        )

        # -------------------------------------------------
        # STEP 4
        # Concatenate graph observations
        # -------------------------------------------------

        global_state = []

        for traffic_light in (

            EnvironmentConfig
            .TRAFFIC_LIGHT_IDS

        ):

            if traffic_light not in (

                graph_observations

            ):

                raise ValueError(

                    f"Missing graph observation "
                    f"for agent "
                    f"{traffic_light}."

                )

            graph_observation = np.asarray(

                graph_observations[
                    traffic_light
                ],

                dtype=np.float32

            ).flatten()

            # -------------------------------------------------
            # Validate graph
            # -------------------------------------------------

            if graph_observation.size != (

                self.graph_observation_size

            ):

                raise ValueError(

                    "\n"
                    "========================================\n"
                    "GRAPH OBSERVATION DIMENSION ERROR\n"
                    "========================================\n"
                    f"Agent: {traffic_light}\n"
                    f"Expected: "
                    f"{self.graph_observation_size}\n"
                    f"Received: "
                    f"{graph_observation.size}\n"
                    "========================================"

                )

            global_state.extend(

                graph_observation.tolist()

            )

        # -------------------------------------------------
        # Convert to numpy
        # -------------------------------------------------

        global_state = np.asarray(

            global_state,

            dtype=np.float32

        )

        # -------------------------------------------------
        # Numerical safety
        # -------------------------------------------------

        global_state = np.nan_to_num(

            global_state,

            nan=0.0,

            posinf=100000.0,

            neginf=-100000.0

        ).astype(

            np.float32

        )

        # -------------------------------------------------
        # Validate final global observation
        # -------------------------------------------------

        self._validate_global_observation(

            global_state

        )

        # -------------------------------------------------
        # IMPORTANT
        #
        # We DO NOT do this:
        #
        #     self.marl.update_states(graph_observations)
        #
        # The MARL state remains 15.
        # -------------------------------------------------

        return global_state

    # =====================================================
    # OBSERVATION VALIDATION
    # =====================================================

    def _validate_global_observation(
        self,
        observation
    ):

        observation = np.asarray(

            observation,

            dtype=np.float32

        ).flatten()

        expected_size = (

            self.graph_observation_size

            *

            EnvironmentConfig
            .NUMBER_OF_AGENTS

        )

        if observation.size != expected_size:

            raise ValueError(

                "\n"
                "========================================\n"
                "GLOBAL MARL OBSERVATION ERROR\n"
                "========================================\n"
                f"Expected: {expected_size}\n"
                f"Received: {observation.size}\n"

                "\n"
                "MARL architecture:\n"
                f"Local observation: "
                f"{self.local_observation_size}\n"
                f"Neighbour observation: "
                f"{self.neighbour_observation_size}\n"
                f"Graph observation: "
                f"{self.graph_observation_size}\n"
                f"Agents: "
                f"{EnvironmentConfig.NUMBER_OF_AGENTS}\n"
                f"Global observation: "
                f"{expected_size}\n"
                "========================================"

            )

        if not np.all(
            np.isfinite(observation)
        ):

            raise ValueError(

                "Global MARL observation contains "
                "NaN or infinite values."

            )

        return True

    # =====================================================
    # STEP
    # =====================================================

    def step(
        self,
        actions
    ):

        if not self.is_initialized:

            raise RuntimeError(

                "Environment is not initialized. "
                "Call reset() before step()."

            )

        # -------------------------------------------------
        # Convert actions
        # -------------------------------------------------

        actions = np.asarray(

            actions,

            dtype=np.int32

        ).flatten()

        # -------------------------------------------------
        # Validate action count
        # -------------------------------------------------

        expected_actions = (

            EnvironmentConfig
            .NUMBER_OF_AGENTS

        )

        if actions.shape[0] != expected_actions:

            raise ValueError(

                "Invalid action dimension. "

                f"Expected {expected_actions}, "

                f"received {actions.shape[0]}."

            )

        # -------------------------------------------------
        # Action dictionary
        # -------------------------------------------------

        action_dict = {}

        # =================================================
        # APPLY ACTIONS
        # =================================================

        for index, traffic_light in enumerate(

            EnvironmentConfig
            .TRAFFIC_LIGHT_IDS

        ):

            action = int(

                actions[index]

            )

            # -------------------------------------------------
            # Validate action
            # -------------------------------------------------

            if (

                action < 0

                or

                action >= (

                    EnvironmentConfig
                    .NUMBER_OF_ACTIONS

                )

            ):

                raise ValueError(

                    f"Invalid action {action} "
                    f"for agent {traffic_light}. "

                    f"Valid range: 0 - "

                    f"{EnvironmentConfig.NUMBER_OF_ACTIONS - 1}"

                )

            action_dict[
                traffic_light
            ] = action

            # -------------------------------------------------
            # Apply action to SUMO controller
            # -------------------------------------------------

            self.controllers[
                traffic_light
            ].apply_action(

                action

            )

        # =================================================
        # SYNCHRONIZE ACTIONS
        # =================================================

        self.marl.apply_actions(

            action_dict

        )

        # =================================================
        # ADVANCE SUMO
        # =================================================

        traci.simulationStep()

        self.step_count += 1

        # =================================================
        # BUILD NEXT 120-D OBSERVATION
        # =================================================

        global_state = (

            self._build_global_observation()

        )

        global_state = np.asarray(

            global_state,

            dtype=np.float32

        ).flatten()

        # -------------------------------------------------
        # Validate global observation
        # -------------------------------------------------

        self._validate_global_observation(

            global_state

        )

        # =================================================
        # GET GRAPH OBSERVATIONS
        # =================================================

        # IMPORTANT:
        #
        # Do NOT call:
        #
        #     self.marl.graph_observations()
        #
        # here.
        #
        # The MARL states are intentionally 15.
        #
        # We already built the correct graph observations
        # inside _build_global_observation().
        #
        # Use the cached 30-dimensional observations.

        graph_observations = {

            agent_id:

                observation.copy()

            for agent_id, observation

            in self.last_graph_observations.items()

        }

        # -------------------------------------------------
        # Validate every graph observation
        # -------------------------------------------------

        for traffic_light in (

            EnvironmentConfig
            .TRAFFIC_LIGHT_IDS

        ):

            if traffic_light not in (

                graph_observations

            ):

                raise ValueError(

                    f"Missing cached graph observation "
                    f"for agent {traffic_light}."

                )

            graph_observations[
                traffic_light
            ] = np.asarray(

                graph_observations[
                    traffic_light
                ],

                dtype=np.float32

            ).flatten()

            if graph_observations[
                traffic_light
            ].size != (

                self.graph_observation_size

            ):

                raise ValueError(

                    "\n"
                    "========================================\n"
                    "GRAPH OBSERVATION DIMENSION ERROR\n"
                    "========================================\n"
                    f"Agent: {traffic_light}\n"
                    f"Expected: "
                    f"{self.graph_observation_size}\n"
                    f"Received: "
                    f"{graph_observations[traffic_light].size}\n"
                    "========================================"

                )

        # =================================================
        # LOCAL REWARDS
        # =================================================

        local_rewards = {}

        for traffic_light in (

            EnvironmentConfig
            .TRAFFIC_LIGHT_IDS

        ):

            local_rewards[
                traffic_light
            ] = (

                self.reward_function.local_reward(

                    graph_observations[
                        traffic_light
                    ]

                )

            )

        # =================================================
        # COOPERATIVE REWARDS
        # =================================================

        rewards = {}

        for traffic_light in (

            EnvironmentConfig
            .TRAFFIC_LIGHT_IDS

        ):

            neighbour_information = (

                self.marl
                .communication
                .neighbour_status(

                    traffic_light

                )

            )

            rewards[
                traffic_light
            ] = (

                self.reward_function.calculate_reward(

                    graph_observations[
                        traffic_light
                    ],

                    neighbour_information,

                    local_rewards

                )

            )

        # =================================================
        # SYNCHRONIZE REWARDS
        # =================================================

        self.marl.update_rewards(

            rewards

        )

        # =================================================
        # UPDATE AGENT METRICS
        # =================================================

        for traffic_light in (

            EnvironmentConfig
            .TRAFFIC_LIGHT_IDS

        ):

            graph_observation = (

                graph_observations[
                    traffic_light
                ]

            )

            # -------------------------------------------------
            # Graph structure:
            #
            # [0:15]  = local
            # [15:30] = neighbour
            #
            # Metrics must use LOCAL values only.
            # -------------------------------------------------

            local_observation = (

                graph_observation[
                    :self.local_observation_size
                ]

            )

            # -------------------------------------------------
            # Waiting time
            # -------------------------------------------------

            waiting_time = float(

                local_observation[2]

            )

            # -------------------------------------------------
            # Queue
            # -------------------------------------------------

            queue_length = float(

                local_observation[5]

                +

                local_observation[6]

                +

                local_observation[7]

                +

                local_observation[8]

            )

            # -------------------------------------------------
            # Speed
            # -------------------------------------------------

            average_speed = float(

                local_observation[1]

            )

            self.marl.get_agent(

                traffic_light

            ).update_metrics(

                waiting_time=waiting_time,

                queue_length=queue_length,

                average_speed=average_speed

            )

        # =================================================
        # ENVIRONMENT REWARD
        # =================================================

        reward = (

            float(

                np.mean(

                    list(
                        rewards.values()
                    )

                )

            )

            if rewards

            else 0.0

        )

        # =================================================
        # TERMINATION
        # =================================================

        terminated = (

            self.step_count

            >=

            EnvironmentConfig
            .MAX_EPISODE_STEPS

        )

        truncated = False

        # =================================================
        # SHARED REWARDS
        # =================================================

        shared_rewards = {

            agent_id:

                self.marl
                .get_agent(agent_id)
                .get_shared_reward()

            for agent_id in (

                EnvironmentConfig
                .TRAFFIC_LIGHT_IDS

            )

        }

        # =================================================
        # NEIGHBOUR INFORMATION
        # =================================================

        neighbours = {

            agent_id:

                self.marl
                .communication
                .neighbour_status(

                    agent_id

                )

            for agent_id in (

                EnvironmentConfig
                .TRAFFIC_LIGHT_IDS

            )

        }

        # =================================================
        # NETWORK METRICS
        # =================================================

        network_waiting = (

            self.marl
            .communication
            .network_waiting_time()

        )

        network_queue = (

            self.marl
            .communication
            .network_queue_length()

        )

        network_speed = (

            self.marl
            .communication
            .network_average_speed()

        )

        # =================================================
        # NETWORK REWARD
        # =================================================

        network_reward = (

            float(

                np.mean(

                    list(
                        rewards.values()
                    )

                )

            )

            if rewards

            else 0.0

        )

        # =================================================
        # INFO
        # =================================================

        info = {

            "scenario":
                self.scenario_name,

            "step":
                self.step_count,

            "seed":
                self._episode_seed,

            "episode_length":
                self.step_count,

            "actions":
                action_dict,

            "local_rewards":
                local_rewards,

            "cooperative_rewards":
                rewards,

            "shared_rewards":
                shared_rewards,

            "network_reward":
                network_reward,

            "neighbours":
                neighbours,

            "network_waiting_time":
                float(
                    network_waiting
                ),

            "network_queue_length":
                float(
                    network_queue
                ),

            "network_average_speed":
                float(
                    network_speed
                ),

            "network_status":
                self.marl
                .communication
                .network_status(),

            "network_summary":
                self.marl
                .communication
                .network_summary(),

            "observation_size":
                int(
                    global_state.shape[0]
                ),

            "local_observation_size":
                self.local_observation_size,

            "neighbour_observation_size":
                self.neighbour_observation_size,

            "graph_observation_size":
                self.graph_observation_size,

            "number_of_agents":
                EnvironmentConfig
                .NUMBER_OF_AGENTS,

            "terminated":
                terminated,

            "truncated":
                truncated

        }

        # =================================================
        # FINAL RETURN
        # =================================================

        return (

            global_state,

            reward,

            terminated,

            truncated,

            info

        )

    # =====================================================
    # RENDER
    # =====================================================

    def render(
        self
    ):

        return None

    # =====================================================
    # CLOSE
    # =====================================================

    def close(
        self
    ):

        if traci.isLoaded():

            traci.close()

        self.marl = None

        self.controllers = {}

        self.observers = {}

        self.last_graph_observations = {}

        self.step_count = 0

        self.is_initialized = False

        self._episode_seed = None

    # =====================================================
    # SCENARIO INFORMATION
    # =====================================================

    def scenario_info(self):

        return {

            "scenario":
                self.scenario_name,

            "sumo_config":
                str(
                    self._scenario_config()
                ),

            "max_episode_steps":
                EnvironmentConfig
                .MAX_EPISODE_STEPS,

            "number_of_agents":
                EnvironmentConfig
                .NUMBER_OF_AGENTS,

            "traffic_lights":
                EnvironmentConfig
                .TRAFFIC_LIGHT_IDS,

            "local_observation_size":
                self.local_observation_size,

            "neighbour_observation_size":
                self.neighbour_observation_size,

            "graph_observation_size":
                self.graph_observation_size,

            "global_observation_size":
                self.observation_size,

            "observation_space":
                str(
                    self.observation_space
                ),

            "action_space":
                str(
                    self.action_space
                ),

            "number_of_actions":
                EnvironmentConfig.NUMBER_OF_ACTIONS

        }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print()

    print("=" * 70)

    print(
        "Traffic MARL Scenario Environment Test"
    )

    print("=" * 70)

    print()

    print(
        "MARL Observation Architecture:"
    )

    print(
        "Local observation      : 15"
    )

    print(
        "Neighbour observation  : 15"
    )

    print(
        "Graph observation      : 30"
    )

    print(
        "Agents                 : 4"
    )

    print(
        "Global observation     : 120"
    )

    print()

    scenarios = [

        "baseline",

        "morning_rush",

        "stress_test"

    ]

    for scenario in scenarios:

        print()

        print(
            f"## Testing scenario: "
            f"{scenario}"
        )

        print("-" * 70)

        env = None

        try:

            env = TrafficEnvironment(

                scenario_name=scenario

            )

            # =================================================
            # RESET
            # =================================================

            observation, info = (

                env.reset(

                    seed=42

                )

            )

            print()

            print(
                "Scenario:",
                info["scenario"]
            )

            print(
                "SUMO Config:",
                info["sumo_config"]
            )

            print(
                "Seed:",
                info["seed"]
            )

            print(
                "Local observation size:",
                info[
                    "local_observation_size"
                ]
            )

            print(
                "Neighbour observation size:",
                info[
                    "neighbour_observation_size"
                ]
            )

            print(
                "Graph observation size:",
                info[
                    "graph_observation_size"
                ]
            )

            print(
                "Global observation size:",
                observation.shape[0]
            )

            print(
                "Observation shape:",
                observation.shape
            )

            print(
                "Observation space:",
                env.observation_space.shape
            )

            # =================================================
            # VALIDATE RESET
            # =================================================

            expected_size = (

                30

                *

                EnvironmentConfig
                .NUMBER_OF_AGENTS

            )

            if observation.shape != (

                expected_size,

            ):

                raise ValueError(

                    "Reset observation dimension "
                    "is incorrect. "

                    f"Expected: "
                    f"{expected_size}, "

                    f"received: "
                    f"{observation.shape}."

                )

            print(
                "[PASS] Reset observation = 120"
            )

            # =================================================
            # IMPORTANT STATE VALIDATION
            # =================================================

            # The agent states MUST remain 15.

            agent_states_valid = True

            for agent_id in (

                EnvironmentConfig
                .TRAFFIC_LIGHT_IDS

            ):

                state = (

                    env.marl
                    .get_agent(agent_id)
                    .get_state()

                )

                if state is None:

                    raise ValueError(

                        f"Agent {agent_id} "
                        "state is None."

                    )

                state = np.asarray(

                    state

                ).flatten()

                print(

                    f"Agent {agent_id} "
                    f"MARL state shape: "
                    f"{state.shape}"

                )

                if state.size != 15:

                    agent_states_valid = False

                    raise ValueError(

                        f"Agent {agent_id} "
                        f"state corruption detected. "

                        f"Expected 15, "
                        f"received {state.size}."

                    )

            if agent_states_valid:

                print(
                    "[PASS] Agent states remain 15"
                )

            # =================================================
            # GRAPH VALIDATION
            # =================================================

            for agent_id in (

                EnvironmentConfig
                .TRAFFIC_LIGHT_IDS

            ):

                graph = (

                    env.last_graph_observations[
                        agent_id
                    ]

                )

                print(

                    f"Agent {agent_id} "
                    f"graph shape: "
                    f"{graph.shape}"

                )

                if graph.shape != (30,):

                    raise ValueError(

                        f"Agent {agent_id} "
                        f"graph observation is not 30. "

                        f"Received: "
                        f"{graph.shape}"

                    )

            print(
                "[PASS] All graph observations = 30"
            )

            # =================================================
            # RANDOM ACTION
            # =================================================

            action = (

                env.action_space.sample()

            )

            print()

            print(
                "Test action:",
                action
            )

            # =================================================
            # STEP
            # =================================================

            (

                next_observation,

                reward,

                terminated,

                truncated,

                step_info

            ) = env.step(

                action

            )

            print()

            print(
                "Next observation:",
                next_observation.shape
            )

            print(
                "Step reward:",
                reward
            )

            print(
                "Step:",
                step_info["step"]
            )

            print(
                "Network waiting:",
                step_info[
                    "network_waiting_time"
                ]
            )

            print(
                "Network queue:",
                step_info[
                    "network_queue_length"
                ]
            )

            print(
                "Network speed:",
                step_info[
                    "network_average_speed"
                ]
            )

            # =================================================
            # VALIDATE STEP OBSERVATION
            # =================================================

            if next_observation.shape != (

                expected_size,

            ):

                raise ValueError(

                    "Step observation dimension "
                    "is incorrect. "

                    f"Expected: "
                    f"{expected_size}, "

                    f"received: "
                    f"{next_observation.shape}."

                )

            print(
                "[PASS] Step observation = 120"
            )

            # =================================================
            # VALIDATE AGENT STATES AFTER STEP
            # =================================================

            for agent_id in (

                EnvironmentConfig
                .TRAFFIC_LIGHT_IDS

            ):

                state = (

                    env.marl
                    .get_agent(agent_id)
                    .get_state()

                )

                state = np.asarray(

                    state

                ).flatten()

                if state.size != 15:

                    raise ValueError(

                        f"Agent {agent_id} "
                        f"state became {state.size} "
                        "after step. "

                        "Expected 15."

                    )

            print(
                "[PASS] Agent states remain 15 after step"
            )

            # =================================================
            # VALIDATE GRAPH AFTER STEP
            # =================================================

            for agent_id in (

                EnvironmentConfig
                .TRAFFIC_LIGHT_IDS

            ):

                graph = (

                    env.last_graph_observations[
                        agent_id
                    ]

                )

                graph = np.asarray(
                    graph
                ).flatten()

                if graph.size != 30:

                    raise ValueError(

                        f"Agent {agent_id} "
                        f"graph became {graph.size}. "

                        "Expected 30."

                    )

            print(
                "[PASS] Graph observations remain 30"
            )

            print(
                "[PASS] Action space valid"
            )

            print(
                "[PASS] SUMO simulation advanced"
            )

            print()

            print(
                "[OK]",
                scenario
            )

        except Exception as error:

            print()

            print(
                "[ERROR]",
                scenario
            )

            print(
                type(error).__name__,
                ":",
                error
            )

        finally:

            if env is not None:

                try:

                    env.close()

                except Exception:

                    pass

    print()

    print("=" * 70)

    print(
        "Scenario Environment Test Complete"
    )

    print("=" * 70)

    print()