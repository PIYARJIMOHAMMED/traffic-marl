"""
=========================================================
Traffic MARL Project

Module:
Scenario Manager

Description:
Creates, validates, and manages SUMO configuration
files and traffic-demand route files for all scenarios.

Scenarios
---------
baseline
low_demand
balanced
morning_rush
evening_rush
high_demand
north_south_heavy
east_west_heavy
random
stress_test

Author:
Mohammed Piyarji
=========================================================
"""

from pathlib import Path
import json
import random
import xml.etree.ElementTree as ET


class ScenarioManager:
    """
    Generates and manages SUMO traffic-demand scenarios.
    """

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
        "stress_test",
    ]

    # -----------------------------------------------------
    # Base demand from the original network route file
    # -----------------------------------------------------

    BASE_PROBABILITIES = {
        "flow1": 0.22,   # west -> east top
        "flow2": 0.22,   # east -> west top
        "flow3": 0.20,   # west -> east bottom
        "flow4": 0.20,   # east -> west bottom
        "flow5": 0.18,   # north -> south left
        "flow6": 0.18,   # south -> north left
        "flow7": 0.18,   # north -> south right
        "flow8": 0.18,   # south -> north right
    }

    # -----------------------------------------------------
    # Fixed random seed for reproducibility
    # -----------------------------------------------------

    RANDOM_SEED = 42

    def __init__(self):

        # -------------------------------------------------
        # Project directories
        # -------------------------------------------------

        self.project_root = (
            Path(__file__).resolve().parents[2]
        )

        self.scenario_directory = (
            self.project_root
            / "simulation"
            / "scenarios"
        )

        self.network_directory = (
            self.project_root
            / "simulation"
            / "networks"
            / "four_intersection"
        )

        self.network_file = (
            self.network_directory
            / "network.net.xml"
        )

        self.base_route_file = (
            self.network_directory
            / "routes.rou.xml"
        )

        self.experiment_config = (
            self.scenario_directory
            / "scenario_experiment_config.json"
        )

    # =====================================================
    # DISPLAY
    # =====================================================

    def display_information(self):

        print()
        print("=" * 70)
        print("Traffic MARL Scenario Manager")
        print("=" * 70)

        print()
        print("Project Root:")
        print(self.project_root)

        print()
        print("Scenario Directory:")
        print(self.scenario_directory)

        print()
        print("Common Network:")
        print(self.network_file)

        print()
        print("Base Route File:")
        print(self.base_route_file)

    # =====================================================
    # CHECK COMMON NETWORK
    # =====================================================

    def check_common_network(self):

        print()
        print("Checking common network...")

        if not self.network_file.exists():
            raise FileNotFoundError(
                f"Common network not found:\n"
                f"{self.network_file}"
            )

        if not self.base_route_file.exists():
            raise FileNotFoundError(
                f"Base route file not found:\n"
                f"{self.base_route_file}"
            )

        print("[OK] Common network found")
        print("[OK] Base route file found")

    # =====================================================
    # SCENARIO ROUTE
    # =====================================================

    def get_route_file(self, scenario):

        return (
            self.scenario_directory
            / scenario
            / "routes.rou.xml"
        )

    # =====================================================
    # SCENARIO CONFIG
    # =====================================================

    def get_config_file(self, scenario):

        return (
            self.scenario_directory
            / scenario
            / "simulation.sumocfg"
        )

    # =====================================================
    # LOAD BASE ROUTE XML
    # =====================================================

    def load_base_routes(self):

        if not self.base_route_file.exists():
            raise FileNotFoundError(
                f"Base routes file not found:\n"
                f"{self.base_route_file}"
            )

        tree = ET.parse(self.base_route_file)

        root = tree.getroot()

        if root.tag != "routes":
            raise ValueError(
                "Base route file root must be <routes>"
            )

        return tree, root

    # =====================================================
    # DEMAND PROFILES
    # =====================================================

    def get_demand_profile(self, scenario):

        base = self.BASE_PROBABILITIES.copy()

        # -------------------------------------------------
        # Baseline
        # -------------------------------------------------

        if scenario == "baseline":
            return base

        # -------------------------------------------------
        # Low demand
        # -------------------------------------------------

        if scenario == "low_demand":

            return {
                flow: round(value * 0.50, 3)
                for flow, value in base.items()
            }

        # -------------------------------------------------
        # Balanced
        #
        # All eight directions receive approximately equal
        # demand.
        # -------------------------------------------------

        if scenario == "balanced":

            return {
                "flow1": 0.18,
                "flow2": 0.18,
                "flow3": 0.18,
                "flow4": 0.18,
                "flow5": 0.18,
                "flow6": 0.18,
                "flow7": 0.18,
                "flow8": 0.18,
            }

        # -------------------------------------------------
        # Morning rush
        #
        # Stronger traffic entering the network from
        # west and south.
        # -------------------------------------------------

        if scenario == "morning_rush":

            return {
                "flow1": 0.38,   # west -> east
                "flow2": 0.08,   # east -> west
                "flow3": 0.34,   # west -> east
                "flow4": 0.08,   # east -> west
                "flow5": 0.30,   # north -> south
                "flow6": 0.08,   # south -> north
                "flow7": 0.28,   # north -> south
                "flow8": 0.08,   # south -> north
            }

        # -------------------------------------------------
        # Evening rush
        #
        # Reverse directional pressure.
        # Stronger traffic toward west/north.
        # -------------------------------------------------

        if scenario == "evening_rush":

            return {
                "flow1": 0.08,   # west -> east
                "flow2": 0.38,   # east -> west
                "flow3": 0.08,   # west -> east
                "flow4": 0.34,   # east -> west
                "flow5": 0.08,   # north -> south
                "flow6": 0.30,   # south -> north
                "flow7": 0.08,   # north -> south
                "flow8": 0.28,   # south -> north
            }

        # -------------------------------------------------
        # High demand
        # -------------------------------------------------

        if scenario == "high_demand":

            return {
                flow: min(
                    round(value * 1.50, 3),
                    0.90
                )
                for flow, value in base.items()
            }

        # -------------------------------------------------
        # North-South heavy
        # -------------------------------------------------

        if scenario == "north_south_heavy":

            return {
                "flow1": 0.08,
                "flow2": 0.08,
                "flow3": 0.08,
                "flow4": 0.08,
                "flow5": 0.40,
                "flow6": 0.36,
                "flow7": 0.40,
                "flow8": 0.36,
            }

        # -------------------------------------------------
        # East-West heavy
        # -------------------------------------------------

        if scenario == "east_west_heavy":

            return {
                "flow1": 0.40,
                "flow2": 0.36,
                "flow3": 0.40,
                "flow4": 0.36,
                "flow5": 0.08,
                "flow6": 0.08,
                "flow7": 0.08,
                "flow8": 0.08,
            }

        # -------------------------------------------------
        # Random
        #
        # Reproducible pseudo-random traffic profile.
        # -------------------------------------------------

        if scenario == "random":

            rng = random.Random(
                self.RANDOM_SEED
            )

            return {
                flow: round(
                    rng.uniform(
                        0.08,
                        0.40
                    ),
                    3
                )
                for flow in base
            }

        # -------------------------------------------------
        # Stress test
        #
        # Very high and deliberately unbalanced demand.
        # -------------------------------------------------

        if scenario == "stress_test":

            return {
                "flow1": 0.55,
                "flow2": 0.20,
                "flow3": 0.50,
                "flow4": 0.18,
                "flow5": 0.45,
                "flow6": 0.12,
                "flow7": 0.50,
                "flow8": 0.12,
            }

        raise ValueError(
            f"Unknown scenario: {scenario}"
        )

    # =====================================================
    # GENERATE ROUTE FILE
    # =====================================================

    def create_route_file(self, scenario):

        scenario_directory = (
            self.scenario_directory
            / scenario
        )

        scenario_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        output_file = (
            scenario_directory
            / "routes.rou.xml"
        )

        tree, root = self.load_base_routes()

        demand = self.get_demand_profile(
            scenario
        )

        # -------------------------------------------------
        # Update vehicle flow probabilities
        # -------------------------------------------------

        flow_count = 0

        for flow in root.findall("flow"):

            flow_id = flow.get("id")

            if flow_id in demand:

                probability = demand[
                    flow_id
                ]

                flow.set(
                    "probability",
                    f"{probability:.3f}"
                )

                flow_count += 1

        if flow_count != 8:

            raise ValueError(
                f"Expected 8 demand flows but found "
                f"{flow_count} matching flows in "
                f"{scenario}"
            )

        # -------------------------------------------------
        # Scenario-specific bus demand
        # -------------------------------------------------

        bus_top = root.find(
            "flow[@id='bus_top']"
        )

        bus_bottom = root.find(
            "flow[@id='bus_bottom']"
        )

        if bus_top is not None:
            bus_top.set(
                "period",
                self.get_bus_period(
                    scenario,
                    "top"
                )
            )

        if bus_bottom is not None:
            bus_bottom.set(
                "period",
                self.get_bus_period(
                    scenario,
                    "bottom"
                )
            )

        # -------------------------------------------------
        # Scenario-specific truck demand
        # -------------------------------------------------

        truck1 = root.find(
            "flow[@id='truck1']"
        )

        truck2 = root.find(
            "flow[@id='truck2']"
        )

        if truck1 is not None:
            truck1.set(
                "period",
                self.get_truck_period(
                    scenario,
                    "truck1"
                )
            )

        if truck2 is not None:
            truck2.set(
                "period",
                self.get_truck_period(
                    scenario,
                    "truck2"
                )
            )

        # -------------------------------------------------
        # Pretty XML
        # -------------------------------------------------

        try:
            ET.indent(
                tree,
                space="    "
            )
        except AttributeError:
            # Python versions without ET.indent
            pass

        tree.write(
            output_file,
            encoding="UTF-8",
            xml_declaration=True
        )

        return output_file

    # =====================================================
    # BUS PERIOD
    # =====================================================

    def get_bus_period(
        self,
        scenario,
        direction
    ):

        profiles = {

            "baseline": {
                "top": 120,
                "bottom": 150
            },

            "low_demand": {
                "top": 180,
                "bottom": 210
            },

            "balanced": {
                "top": 140,
                "bottom": 140
            },

            "morning_rush": {
                "top": 75,
                "bottom": 90
            },

            "evening_rush": {
                "top": 90,
                "bottom": 75
            },

            "high_demand": {
                "top": 80,
                "bottom": 90
            },

            "north_south_heavy": {
                "top": 150,
                "bottom": 150
            },

            "east_west_heavy": {
                "top": 70,
                "bottom": 80
            },

            "random": {
                "top": 110,
                "bottom": 170
            },

            "stress_test": {
                "top": 60,
                "bottom": 70
            },
        }

        return str(
            profiles[scenario][direction]
        )

    # =====================================================
    # TRUCK PERIOD
    # =====================================================

    def get_truck_period(
        self,
        scenario,
        truck
    ):

        profiles = {

            "baseline": {
                "truck1": 90,
                "truck2": 110
            },

            "low_demand": {
                "truck1": 150,
                "truck2": 170
            },

            "balanced": {
                "truck1": 120,
                "truck2": 120
            },

            "morning_rush": {
                "truck1": 70,
                "truck2": 100
            },

            "evening_rush": {
                "truck1": 100,
                "truck2": 70
            },

            "high_demand": {
                "truck1": 60,
                "truck2": 75
            },

            "north_south_heavy": {
                "truck1": 60,
                "truck2": 80
            },

            "east_west_heavy": {
                "truck1": 100,
                "truck2": 120
            },

            "random": {
                "truck1": 100,
                "truck2": 130
            },

            "stress_test": {
                "truck1": 45,
                "truck2": 55
            },
        }

        return str(
            profiles[scenario][truck]
        )

    # =====================================================
    # CREATE CONFIG
    # =====================================================

    def create_config(self, scenario):

        scenario_directory = (
            self.scenario_directory
            / scenario
        )

        scenario_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        route_file = self.get_route_file(
            scenario
        )

        config_file = self.get_config_file(
            scenario
        )

        if not route_file.exists():

            raise FileNotFoundError(
                f"Route file missing for scenario "
                f"'{scenario}':\n"
                f"{route_file}"
            )

        xml_content = """<?xml version="1.0" encoding="UTF-8"?>

<configuration>

    <input>

        <net-file value="../../networks/four_intersection/network.net.xml"/>

        <route-files value="routes.rou.xml"/>

    </input>

    <time>

        <begin value="0"/>

        <end value="3600"/>

        <step-length value="1"/>

    </time>

    <processing>

        <collision.action value="warn"/>

        <time-to-teleport value="-1"/>

    </processing>

    <report>

        <verbose value="false"/>

        <no-step-log value="true"/>

    </report>

</configuration>
"""

        config_file.write_text(
            xml_content,
            encoding="utf-8"
        )

        return config_file

    # =====================================================
    # VALIDATE ROUTE FILE
    # =====================================================

    def validate_route_file(self, scenario):

        route_file = self.get_route_file(
            scenario
        )

        if not route_file.exists():

            return False, "Route file missing"

        if route_file.stat().st_size == 0:

            return False, "Route file empty"

        try:

            tree = ET.parse(
                route_file
            )

            root = tree.getroot()

        except Exception as error:

            return (
                False,
                f"Invalid route XML: {error}"
            )

        if root.tag != "routes":

            return (
                False,
                "Route root must be <routes>"
            )

        flows = root.findall("flow")

        if len(flows) < 8:

            return (
                False,
                f"Expected at least 8 flows, "
                f"found {len(flows)}"
            )

        return True, "Valid"

    # =====================================================
    # VALIDATE CONFIG
    # =====================================================

    def validate_config(self, scenario):

        config_file = self.get_config_file(
            scenario
        )

        route_file = self.get_route_file(
            scenario
        )

        if not config_file.exists():

            return False, "Configuration file missing"

        if config_file.stat().st_size == 0:

            return False, "Configuration file is empty"

        if not route_file.exists():

            return False, "Route file missing"

        try:

            tree = ET.parse(
                config_file
            )

            root = tree.getroot()

        except Exception as error:

            return (
                False,
                f"Invalid XML: {error}"
            )

        if root.tag != "configuration":

            return (
                False,
                "Root element must be configuration"
            )

        input_element = root.find(
            "input"
        )

        if input_element is None:

            return (
                False,
                "Missing input section"
            )

        net_element = input_element.find(
            "net-file"
        )

        route_element = input_element.find(
            "route-files"
        )

        if net_element is None:

            return (
                False,
                "Missing net-file"
            )

        if route_element is None:

            return (
                False,
                "Missing route-files"
            )

        expected_network = (
            "../../networks/four_intersection/"
            "network.net.xml"
        )

        actual_network = (
            net_element.get("value")
        )

        if actual_network != expected_network:

            return (
                False,
                f"Incorrect network path: "
                f"{actual_network}"
            )

        actual_route = (
            route_element.get("value")
        )

        if actual_route != "routes.rou.xml":

            return (
                False,
                f"Incorrect route path: "
                f"{actual_route}"
            )

        return True, "Valid"

    # =====================================================
    # PREPARE ALL SCENARIOS
    # =====================================================

    def prepare_scenarios(self):

        print()
        print("Available Scenarios:")

        for scenario in self.SCENARIOS:
            print(f"- {scenario}")

        print()
        print(
            "Generating scenario-specific "
            "traffic demand..."
        )

        results = {}

        for scenario in self.SCENARIOS:

            try:

                # -----------------------------------------
                # Generate unique route file
                # -----------------------------------------

                route_file = (
                    self.create_route_file(
                        scenario
                    )
                )

                # -----------------------------------------
                # Generate scenario config
                # -----------------------------------------

                config_file = (
                    self.create_config(
                        scenario
                    )
                )

                print()
                print(
                    f"[OK] {scenario}"
                )

                print(
                    f"     Route: {route_file}"
                )

                print(
                    f"     Config: {config_file}"
                )

                print(
                    "     Demand:"
                )

                demand = (
                    self.get_demand_profile(
                        scenario
                    )
                )

                for flow, probability in demand.items():

                    print(
                        f"       {flow}: "
                        f"{probability:.3f}"
                    )

                results[scenario] = {

                    "config": str(
                        config_file
                    ),

                    "route": str(
                        route_file
                    ),

                    "demand": demand,

                }

            except Exception as error:

                print()
                print(
                    f"[ERROR] {scenario}"
                )

                print(error)

                results[scenario] = {
                    "error": str(error)
                }

        return results

    # =====================================================
    # VALIDATE ALL
    # =====================================================

    def validate_all(self):

        print()
        print(
            "Validating generated scenarios..."
        )

        validation_results = {}

        for scenario in self.SCENARIOS:

            route_valid, route_message = (
                self.validate_route_file(
                    scenario
                )
            )

            config_valid, config_message = (
                self.validate_config(
                    scenario
                )
            )

            valid = (
                route_valid
                and config_valid
            )

            if valid:

                message = "Valid"

                print(
                    f"[VALID] {scenario}"
                )

            else:

                messages = []

                if not route_valid:
                    messages.append(
                        f"route: {route_message}"
                    )

                if not config_valid:
                    messages.append(
                        f"config: {config_message}"
                    )

                message = "; ".join(
                    messages
                )

                print(
                    f"[INVALID] {scenario}"
                )

                print(
                    f"          {message}"
                )

            validation_results[
                scenario
            ] = {

                "valid": valid,

                "message": message

            }

        return validation_results

    # =====================================================
    # CREATE EXPERIMENT CONFIGURATION
    # =====================================================

    def create_experiment_config(
        self,
        validation_results
    ):

        configuration = {

            "project":
                "Traffic MARL",

            "version":
                "2.0",

            "network": {

                "name":
                    "four_intersection",

                "file":
                    str(
                        self.network_file
                    )

            },

            "simulation": {

                "begin": 0,

                "end": 3600,

                "step_length": 1

            },

            "random_seed":
                self.RANDOM_SEED,

            "scenarios": {}

        }

        for scenario in self.SCENARIOS:

            config_file = (
                self.get_config_file(
                    scenario
                )
            )

            route_file = (
                self.get_route_file(
                    scenario
                )
            )

            configuration[
                "scenarios"
            ][scenario] = {

                "config_file":
                    str(
                        config_file
                    ),

                "route_file":
                    str(
                        route_file
                    ),

                "valid":
                    validation_results[
                        scenario
                    ][
                        "valid"
                    ],

                "demand_profile":
                    self.get_demand_profile(
                        scenario
                    )

            }

        self.experiment_config.write_text(

            json.dumps(
                configuration,
                indent=4
            ),

            encoding="utf-8"

        )

        return self.experiment_config

    # =====================================================
    # RUN
    # =====================================================

    def run(self):

        self.display_information()

        self.check_common_network()

        preparation_results = (
            self.prepare_scenarios()
        )

        validation_results = (
            self.validate_all()
        )

        experiment_file = (
            self.create_experiment_config(
                validation_results
            )
        )

        print()
        print(
            "Experiment Configuration:"
        )

        print(
            experiment_file
        )

        print()
        print("=" * 70)
        print(
            "Scenario Management Complete"
        )
        print("=" * 70)

        invalid = [
            scenario
            for scenario, result
            in validation_results.items()
            if not result["valid"]
        ]

        if invalid:

            print()
            print(
                "[WARNING] Invalid scenarios:"
            )

            for scenario in invalid:
                print(
                    f" - {scenario}"
                )

        else:

            print()
            print(
                "[SUCCESS] All 10 scenarios are valid."
            )

        return {

            "preparation":
                preparation_results,

            "validation":
                validation_results,

            "experiment_config":
                str(
                    experiment_file
                )

        }


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    manager = ScenarioManager()

    manager.run()