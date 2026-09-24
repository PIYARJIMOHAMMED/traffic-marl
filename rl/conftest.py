import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def marl():
    """MARLController fixture for MARL component tests."""

    from rl.controllers.marl_controller import MARLController

    controller = MARLController()

    yield controller