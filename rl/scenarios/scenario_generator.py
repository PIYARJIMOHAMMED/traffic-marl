"""
=========================================================
Traffic MARL Project

Module:
Scenario Generator

Description:
Generates reproducible traffic-demand scenarios for
the four-intersection SUMO MARL environment.

Supported scenarios:

1. baseline
2. low_demand
3. balanced
4. morning_rush
5. evening_rush
6. high_demand
7. north_south_heavy
8. east_west_heavy
9. random
10. stress_test

The generator works with the existing SUMO route file
and creates scenario-specific route files.

Research purposes:

• Scenario-based evaluation
• Stress testing
• Baseline comparison
• Reproducibility
• Transfer evaluation
• Failure analysis
• Experiment provenance

Week:
5

Task:
5.6

Author:
Mohammed Piyarji
=========================================================
"""

from __future__ import annotations

import copy
import json
import random
import xml.etree.ElementTree as ET

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional


# =========================================================
# SCENARIO CONFIGURATION
# =========================================================


@dataclass
class ScenarioConfig:
    """
    Configuration describing one traffic scenario.

    demand_multiplier:
        Overall traffic demand multiplier.

    seed:
        Random seed used to make the scenario reproducible.

    start_time:
        Beginning of the simulation demand window.

    end_time:
        End of the simulation demand window.

    departure_jitter:
        Maximum random departure-time variation.

    corridor_bias:
        Optional directional traffic bias.

        Supported:

        None
        north_south
        east_west

    description:
        Human-readable scenario description.
    """

    name: str

    demand_multiplier: float

    seed: int

    start_time: float = 0.0

    end_time: float = 900.0

    departure_jitter: float = 0.0

    corridor_bias: Optional[str] = None

    description: str = ""


# =========================================================
# SCENARIO GENERATOR
# =========================================================


class ScenarioGenerator:
    """
    Generates SUMO traffic-demand scenarios.

    The generator does not modify the original route file.

    Every generated scenario is saved separately.

    Example:

        generator = ScenarioGenerator()

        path = generator.generate("morning_rush")

    """

    # =====================================================
    # DEFAULT SCENARIOS
    # =====================================================

    DEFAULT_SCENARIOS = {

        "baseline": ScenarioConfig(

            name="baseline",

            demand_multiplier=1.0,

            seed=42,

            departure_jitter=0.0,

            description=(

                "Normal traffic demand used as "
                "the primary baseline scenario."

            )

        ),

        "low_demand": ScenarioConfig(

            name="low_demand",

            demand_multiplier=0.50,

            seed=43,

            departure_jitter=10.0,

            description=(

                "Low traffic demand scenario."

            )

        ),

        "balanced": ScenarioConfig(

            name="balanced",

            demand_multiplier=1.0,

            seed=44,

            departure_jitter=15.0,

            description=(

                "Balanced traffic distributed "
                "across the network."

            )

        ),

        "morning_rush": ScenarioConfig(

            name="morning_rush",

            demand_multiplier=1.50,

            seed=45,

            departure_jitter=20.0,

            corridor_bias="north_south",

            description=(

                "Morning peak scenario with "
                "higher north-south demand."

            )

        ),

        "evening_rush": ScenarioConfig(

            name="evening_rush",

            demand_multiplier=1.50,

            seed=46,

            departure_jitter=20.0,

            corridor_bias="east_west",

            description=(

                "Evening peak scenario with "
                "higher east-west demand."

            )

        ),

        "high_demand": ScenarioConfig(

            name="high_demand",

            demand_multiplier=2.0,

            seed=47,

            departure_jitter=15.0,

            description=(

                "High traffic demand scenario."

            )

        ),

        "north_south_heavy": ScenarioConfig(

            name="north_south_heavy",

            demand_multiplier=1.75,

            seed=48,

            departure_jitter=15.0,

            corridor_bias="north_south",

            description=(

                "Heavy traffic concentrated "
                "on north-south corridors."

            )

        ),

        "east_west_heavy": ScenarioConfig(

            name="east_west_heavy",

            demand_multiplier=1.75,

            seed=49,

            departure_jitter=15.0,

            corridor_bias="east_west",

            description=(

                "Heavy traffic concentrated "
                "on east-west corridors."

            )

        ),

        "random": ScenarioConfig(

            name="random",

            demand_multiplier=1.0,

            seed=50,

            departure_jitter=30.0,

            description=(

                "Randomized traffic demand "
                "for generalization testing."

            )

        ),

        "stress_test": ScenarioConfig(

            name="stress_test",

            demand_multiplier=3.0,

            seed=51,

            departure_jitter=5.0,

            description=(

                "Extreme traffic demand used "
                "for safety and robustness testing."

            )

        )

    }

    # =====================================================
    # INITIALIZATION
    # =====================================================

    def __init__(

        self,

        route_file: str | Path,

        output_directory: str | Path = "simulation/scenarios",

        seed: int = 42

    ):

        self.route_file = Path(route_file)

        self.output_directory = Path(

            output_directory

        )

        self.seed = seed

        self.random = random.Random(seed)

        # -------------------------------------------------
        # Validate input
        # -------------------------------------------------

        if not self.route_file.exists():

            raise FileNotFoundError(

                f"SUMO route file not found: "
                f"{self.route_file}"

            )

        # -------------------------------------------------
        # Create output directory
        # -------------------------------------------------

        self.output_directory.mkdir(

            parents=True,

            exist_ok=True

        )

    # =====================================================
    # AVAILABLE SCENARIOS
    # =====================================================

    @classmethod
    def available_scenarios(cls) -> List[str]:

        """
        Returns the names of all predefined scenarios.
        """

        return list(

            cls.DEFAULT_SCENARIOS.keys()

        )

    # =====================================================
    # GET CONFIGURATION
    # =====================================================

    @classmethod
    def get_scenario_config(

        cls,

        scenario_name: str

    ) -> ScenarioConfig:

        """
        Returns configuration for a scenario.
        """

        if scenario_name not in cls.DEFAULT_SCENARIOS:

            raise ValueError(

                f"Unknown scenario: "
                f"{scenario_name}\n"
                f"Available scenarios: "
                f"{cls.available_scenarios()}"

            )

        return copy.deepcopy(

            cls.DEFAULT_SCENARIOS[

                scenario_name

            ]

        )

    # =====================================================
    # LOAD XML
    # =====================================================

    def _load_route_file(self):

        """
        Loads the original SUMO route file.
        """

        tree = ET.parse(

            self.route_file

        )

        root = tree.getroot()

        return tree, root

    # =====================================================
    # RANDOM DEPARTURE
    # =====================================================

    def _randomize_departure(

        self,

        original_depart: float,

        config: ScenarioConfig

    ) -> float:

        """
        Applies reproducible departure-time jitter.
        """

        if config.departure_jitter <= 0:

            return original_depart

        jitter = self.random.uniform(

            -config.departure_jitter,

            config.departure_jitter

        )

        new_depart = (

            original_depart

            +

            jitter

        )

        # -------------------------------------------------
        # Keep departure inside scenario window
        # -------------------------------------------------

        new_depart = max(

            config.start_time,

            new_depart

        )

        new_depart = min(

            config.end_time,

            new_depart

        )

        return new_depart

    # =====================================================
    # ROUTE DIRECTION
    # =====================================================

    def _route_direction(

        self,

        vehicle_element

    ) -> Optional[str]:

        """
        Attempts to determine the dominant traffic
        direction from a vehicle's route.

        This is intentionally conservative because
        SUMO route definitions can vary between networks.
        """

        route_element = vehicle_element.find(

            "route"

        )

        if route_element is None:

            return None

        edges = route_element.get(

            "edges",

            ""

        )

        if not edges:

            return None

        edges_upper = edges.upper()

        north_score = 0

        east_score = 0

        south_score = 0

        west_score = 0

        for edge in edges_upper.split():

            if edge.startswith("N"):

                north_score += 1

            elif edge.startswith("E"):

                east_score += 1

            elif edge.startswith("S"):

                south_score += 1

            elif edge.startswith("W"):

                west_score += 1

        ns_score = (

            north_score

            +

            south_score

        )

        ew_score = (

            east_score

            +

            west_score

        )

        if ns_score == 0 and ew_score == 0:

            return None

        if ns_score >= ew_score:

            return "north_south"

        return "east_west"

    # =====================================================
    # CORRIDOR PROBABILITY
    # =====================================================

    def _corridor_probability(

        self,

        vehicle_element,

        corridor_bias: Optional[str]

    ) -> float:

        """
        Determines probability that a vehicle should
        be retained under a directional scenario.

        This is used only when corridor bias is enabled.
        """

        if corridor_bias is None:

            return 1.0

        direction = self._route_direction(

            vehicle_element

        )

        if direction is None:

            return 1.0

        if direction == corridor_bias:

            return 1.0

        return 0.55

    # =====================================================
    # VEHICLE MULTIPLIER
    # =====================================================

    def _calculate_copies(

        self,

        multiplier: float

    ) -> tuple[int, float]:

        """
        Converts a demand multiplier into:

        integer number of copies

        + residual probability

        Example:

        1.5

        -> 1 full copy

        -> 50% probability of second copy
        """

        if multiplier <= 0:

            return 0, 0.0

        full_copies = int(

            multiplier

        )

        remainder = (

            multiplier

            -

            full_copies

        )

        return full_copies, remainder

    # =====================================================
    # CREATE VEHICLE COPY
    # =====================================================

    def _copy_vehicle(

        self,

        vehicle,

        vehicle_id: str,

        config: ScenarioConfig,

        copy_index: int

    ):

        """
        Creates a copy of a SUMO vehicle.
        """

        new_vehicle = copy.deepcopy(

            vehicle

        )

        new_vehicle.set(

            "id",

            vehicle_id

        )

        # -------------------------------------------------
        # Departure time
        # -------------------------------------------------

        depart = vehicle.get(

            "depart"

        )

        if depart is not None:

            try:

                original_depart = float(

                    depart

                )

                new_depart = (

                    self._randomize_departure(

                        original_depart,

                        config

                    )

                )

                new_vehicle.set(

                    "depart",

                    f"{new_depart:.2f}"

                )

            except ValueError:

                pass

        return new_vehicle

    # =====================================================
    # GENERATE VEHICLES
    # =====================================================

    def _generate_vehicle_demand(

        self,

        root,

        config: ScenarioConfig

    ):

        """
        Generates vehicle demand while preserving
        SUMO route definitions and vehicle types.
        """

        children = list(root)

        vehicle_elements = [

            child

            for child in children

            if child.tag == "vehicle"

        ]

        if len(vehicle_elements) == 0:

            return {

                "vehicles_original": 0,

                "vehicles_generated": 0

            }

        full_copies, remainder = (

            self._calculate_copies(

                config.demand_multiplier

            )

        )

        generated = []

        generated_count = 0

        original_count = len(

            vehicle_elements

        )

        for vehicle in vehicle_elements:

            direction_probability = (

                self._corridor_probability(

                    vehicle,

                    config.corridor_bias

                )

            )

            # -------------------------------------------------
            # Generate copies
            # -------------------------------------------------

            for copy_index in range(

                full_copies

            ):

                if self.random.random() > direction_probability:

                    continue

                if copy_index == 0:

                    new_vehicle = copy.deepcopy(

                        vehicle

                    )

                else:

                    new_vehicle = self._copy_vehicle(

                        vehicle,

                        vehicle_id=(

                            f"{vehicle.get('id')}"
                            f"_{config.name}"
                            f"_{copy_index}"

                        ),

                        config=config,

                        copy_index=copy_index

                    )

                # -------------------------------------------------
                # Randomize departure for non-baseline scenarios
                # -------------------------------------------------

                if copy_index == 0:

                    depart = vehicle.get(

                        "depart"

                    )

                    if depart is not None:

                        try:

                            depart_value = float(

                                depart

                            )

                            new_depart = (

                                self._randomize_departure(

                                    depart_value,

                                    config

                                )

                            )

                            new_vehicle.set(

                                "depart",

                                f"{new_depart:.2f}"

                            )

                        except ValueError:

                            pass

                generated.append(

                    new_vehicle

                )

                generated_count += 1

            # -------------------------------------------------
            # Fractional copy
            # -------------------------------------------------

            if remainder > 0:

                probability = (

                    remainder

                    *

                    direction_probability

                )

                if self.random.random() < probability:

                    copy_number = full_copies + 1

                    new_vehicle = self._copy_vehicle(

                        vehicle,

                        vehicle_id=(

                            f"{vehicle.get('id')}"
                            f"_{config.name}"
                            f"_{copy_number}"

                        ),

                        config=config,

                        copy_index=copy_number

                    )

                    generated.append(

                        new_vehicle

                    )

                    generated_count += 1

        # -------------------------------------------------
        # Remove original vehicles
        # -------------------------------------------------

        for vehicle in vehicle_elements:

            root.remove(

                vehicle

            )

        # -------------------------------------------------
        # Add generated vehicles
        # -------------------------------------------------

        for vehicle in generated:

            root.append(

                vehicle

            )

        return {

            "vehicles_original": original_count,

            "vehicles_generated": generated_count

        }

    # =====================================================
    # GENERATE FLOW DEMAND
    # =====================================================

    def _generate_flow_demand(

        self,

        root,

        config: ScenarioConfig

    ):

        """
        Adjusts SUMO <flow> demand.

        Supported attributes:

        vehsPerHour

        number

        period
        """

        flow_count = 0

        modified_flows = 0

        for flow in root.findall(

            "flow"

        ):

            flow_count += 1

            modified_flows += 1

            # -------------------------------------------------
            # vehsPerHour
            # -------------------------------------------------

            if flow.get(

                "vehsPerHour"

            ) is not None:

                try:

                    value = float(

                        flow.get(

                            "vehsPerHour"

                        )

                    )

                    value *= config.demand_multiplier

                    flow.set(

                        "vehsPerHour",

                        f"{value:.2f}"

                    )

                except ValueError:

                    pass

            # -------------------------------------------------
            # number
            # -------------------------------------------------

            elif flow.get(

                "number"

            ) is not None:

                try:

                    value = float(

                        flow.get(

                            "number"

                        )

                    )

                    value *= config.demand_multiplier

                    flow.set(

                        "number",

                        str(

                            max(

                                1,

                                int(

                                    round(value)

                                )

                            )

                        )

                    )

                except ValueError:

                    pass

            # -------------------------------------------------
            # period
            # -------------------------------------------------

            elif flow.get(

                "period"

            ) is not None:

                try:

                    value = float(

                        flow.get(

                            "period"

                        )

                    )

                    if config.demand_multiplier > 0:

                        value /= (

                            config.demand_multiplier

                        )

                        flow.set(

                            "period",

                            f"{value:.3f}"

                        )

                except ValueError:

                    pass

        return {

            "flows_found": flow_count,

            "flows_modified": modified_flows

        }

    # =====================================================
    # GENERATE SCENARIO
    # =====================================================

    def generate(

        self,

        scenario_name: str,

        custom_config: Optional[ScenarioConfig] = None

    ) -> Path:

        """
        Generate one scenario.

        Returns the generated SUMO route-file path.
        """

        if custom_config is None:

            config = self.get_scenario_config(

                scenario_name

            )

        else:

            config = custom_config

        # -------------------------------------------------
        # Reproducibility
        # -------------------------------------------------

        self.random.seed(

            config.seed

        )

        # -------------------------------------------------
        # Load original route file
        # -------------------------------------------------

        tree, root = self._load_route_file()

        # -------------------------------------------------
        # Determine whether route file contains vehicles
        # or flows.
        # -------------------------------------------------

        vehicle_count = len(

            root.findall(

                "vehicle"

            )

        )

        flow_count = len(

            root.findall(

                "flow"

            )

        )

        statistics = {}

        if vehicle_count > 0:

            statistics.update(

                self._generate_vehicle_demand(

                    root,

                    config

                )

            )

        if flow_count > 0:

            statistics.update(

                self._generate_flow_demand(

                    root,

                    config

                )

            )

        # -------------------------------------------------
        # Output directory
        # -------------------------------------------------

        scenario_directory = (

            self.output_directory

            / scenario_name

        )

        scenario_directory.mkdir(

            parents=True,

            exist_ok=True

        )

        # -------------------------------------------------
        # Route file
        # -------------------------------------------------

        output_route_file = (

            scenario_directory

            / "routes.rou.xml"

        )

        tree.write(

            output_route_file,

            encoding="utf-8",

            xml_declaration=True

        )

        # -------------------------------------------------
        # Metadata
        # -------------------------------------------------

        metadata = {

            "project": "Traffic MARL",

            "module": "Scenario Generator",

            "scenario": scenario_name,

            "configuration": asdict(

                config

            ),

            "source_route_file": str(

                self.route_file.resolve()

            ),

            "generated_route_file": str(

                output_route_file.resolve()

            ),

            "random_seed": config.seed,

            "statistics": statistics

        }

        metadata_file = (

            scenario_directory

            / "scenario_metadata.json"

        )

        with open(

            metadata_file,

            "w",

            encoding="utf-8"

        ) as file:

            json.dump(

                metadata,

                file,

                indent=4

            )

        return output_route_file

    # =====================================================
    # GENERATE ALL
    # =====================================================

    def generate_all(self) -> Dict[str, Path]:

        """
        Generate every predefined scenario.
        """

        results = {}

        for scenario_name in (

            self.available_scenarios()

        ):

            results[scenario_name] = (

                self.generate(

                    scenario_name

                )

            )

        return results

    # =====================================================
    # SAVE EXPERIMENT MANIFEST
    # =====================================================

    def create_experiment_manifest(

        self

    ) -> Path:

        """
        Creates a research manifest describing
        all scenarios.

        This can later be used for:

        • Evaluation

        • MLflow

        • Ablation studies

        • Stress testing

        • Dissertation reproducibility
        """

        manifest = {

            "project": "Traffic MARL",

            "generator_seed": self.seed,

            "source_route_file": str(

                self.route_file.resolve()

            ),

            "scenarios": [

                asdict(

                    self.get_scenario_config(

                        scenario

                    )

                )

                for scenario

                in self.available_scenarios()

            ]

        }

        manifest_path = (

            self.output_directory

            / "experiment_manifest.json"

        )

        with open(

            manifest_path,

            "w",

            encoding="utf-8"

        ) as file:

            json.dump(

                manifest,

                file,

                indent=4

            )

        return manifest_path


# =========================================================
# COMMAND-LINE EXECUTION
# =========================================================


def main():

    """
    Command-line entry point.

    Run:

        python -m rl.scenarios.scenario_generator
    """

    print()

    print("=" * 70)

    print("Traffic MARL")

    print("Scenario Generator")

    print("=" * 70)

    print()

    # -----------------------------------------------------
    # Default project route file
    # -----------------------------------------------------

    project_root = (

        Path(__file__)

        .resolve()

        .parents[2]

    )

    route_file = (

        project_root

        / "simulation"

        / "networks"

        / "four_intersection"

        / "routes.rou.xml"

    )

    output_directory = (

        project_root

        / "simulation"

        / "scenarios"

    )

    # -----------------------------------------------------
    # Create generator
    # -----------------------------------------------------

    generator = ScenarioGenerator(

        route_file=route_file,

        output_directory=output_directory,

        seed=42

    )

    print(

        "Source Route File:"

    )

    print(

        route_file

    )

    print()

    print(

        "Output Directory:"

    )

    print(

        output_directory

    )

    print()

    print(

        "Available Scenarios:"

    )

    for scenario in (

        generator.available_scenarios()

    ):

        print(

            f"  - {scenario}"

        )

    print()

    # -----------------------------------------------------
    # Generate scenarios
    # -----------------------------------------------------

    print(

        "Generating scenarios..."

    )

    print()

    results = generator.generate_all()

    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    for scenario, path in results.items():

        print(

            f"[OK] {scenario}"

        )

        print(

            f"     {path}"

        )

    # -----------------------------------------------------
    # Manifest
    # -----------------------------------------------------

    manifest = (

        generator.create_experiment_manifest()

    )

    print()

    print(

        "Experiment Manifest:"

    )

    print(

        manifest

    )

    print()

    print("=" * 70)

    print("Scenario Generation Complete")

    print("=" * 70)


# =========================================================
# MAIN
# =========================================================


if __name__ == "__main__":

    main()