import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from backend.app.core.security import (
    authenticate,
    require_role,
)
from backend.app.core.validation import (
    TrafficControlRequest,
    ScenarioRequest,
)


def test_valid_authentication():
    user = authenticate("Bearer demo-admin-token")

    assert user.username == "admin"
    assert user.role == "admin"


def test_operator_authorization():
    user = authenticate("Bearer demo-operator-token")

    result = require_role(user, {"admin", "operator"})

    assert result.role == "operator"


def test_invalid_token():
    try:
        authenticate("Bearer invalid-token")
        assert False
    except Exception:
        assert True


def test_valid_control_request():
    request = TrafficControlRequest(
        intersection="a",
        action=1,
        green_timer=20,
        in_yellow=False,
    )

    assert request.intersection == "A"
    assert request.action == 1


def test_invalid_action():
    try:
        TrafficControlRequest(
            intersection="A",
            action=5,
            green_timer=20,
        )
        assert False
    except Exception:
        assert True


def test_valid_scenario():
    request = ScenarioRequest(
        scenario="stress_test"
    )

    assert request.scenario == "stress_test"