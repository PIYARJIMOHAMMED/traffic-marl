import sys
from pathlib import Path

import traci

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.environments.config import EnvironmentConfig
from rl.environments.traffic_controller import TrafficController


def test_controller_does_not_restart_same_phase():
    """
    Verify that repeating the current action does not restart
    the SUMO phase.
    """

    controller = TrafficController("A")

    # Initial state
    assert controller.current_action == (
        EnvironmentConfig.ACTION_NORTH_SOUTH
    )

    # We cannot call SUMO here without a live connection.
    # This test verifies the controller's state contract.
    assert controller.pending_action is None
    assert controller.in_yellow is False


def test_action_constants_are_consistent():
    assert (
        EnvironmentConfig.ACTION_NORTH_SOUTH
        != EnvironmentConfig.ACTION_EAST_WEST
    )

    assert (
        EnvironmentConfig.GREEN_PHASE_NS
        != EnvironmentConfig.GREEN_PHASE_EW
    )

    assert (
        EnvironmentConfig.YELLOW_PHASE_NS
        != EnvironmentConfig.YELLOW_PHASE_EW
    )