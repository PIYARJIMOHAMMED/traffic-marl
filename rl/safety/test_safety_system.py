from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from rl.safety import (
    SafetyValidator,
    HumanApprovalGate,
    AuditLogger,
)


def main():

    print("=" * 70)
    print("SAFETY / GOVERNANCE SYSTEM TEST")
    print("=" * 70)

    validator = SafetyValidator(
        min_green_steps=10,
        yellow_protection=True,
    )

    gate = HumanApprovalGate(
        mode="automatic"
    )

    audit_path = (
        PROJECT_ROOT
        / "rl"
        / "logs"
        / "safety"
        / "decision_audit.csv"
    )

    logger = AuditLogger(
        audit_path
    )

    # --------------------------------------------------------
    # Test 1: safe action
    # --------------------------------------------------------

    safe_decision = validator.validate_action(
        intersection="A",
        proposed_action=1,
        green_timer=20,
        in_yellow=False,
    )

    print("\nTest 1 - Safe action")
    print(safe_decision.to_dict())

    assert safe_decision.approved_action == 1
    assert safe_decision.overridden is False

    approval = gate.process(
        safe_decision
    )

    logger.log(
        approval
    )

    assert approval.final_action == 1

    # --------------------------------------------------------
    # Test 2: minimum-green violation
    # --------------------------------------------------------

    unsafe_green = validator.validate_action(
        intersection="B",
        proposed_action=1,
        green_timer=3,
        in_yellow=False,
    )

    print("\nTest 2 - Minimum green violation")
    print(unsafe_green.to_dict())

    assert unsafe_green.overridden is True
    assert unsafe_green.approved_action == 0

    approval = gate.process(
        unsafe_green
    )

    logger.log(
        approval
    )

    assert approval.final_action == 0

    # --------------------------------------------------------
    # Test 3: yellow protection
    # --------------------------------------------------------

    unsafe_yellow = validator.validate_action(
        intersection="C",
        proposed_action=1,
        green_timer=20,
        in_yellow=True,
    )

    print("\nTest 3 - Yellow transition")
    print(unsafe_yellow.to_dict())

    assert unsafe_yellow.overridden is True
    assert unsafe_yellow.approved_action == 0

    approval = gate.process(
        unsafe_yellow
    )

    logger.log(
        approval
    )

    assert approval.final_action == 0

    # --------------------------------------------------------
    # Test 4: invalid action
    # --------------------------------------------------------

    invalid = validator.validate_action(
        intersection="D",
        proposed_action=99,
        green_timer=20,
        in_yellow=False,
    )

    print("\nTest 4 - Invalid action")
    print(invalid.to_dict())

    assert invalid.overridden is True
    assert invalid.approved_action == 0

    approval = gate.process(
        invalid
    )

    logger.log(
        approval
    )

    assert approval.final_action == 0

    # --------------------------------------------------------
    # Test 5: network action validation
    # --------------------------------------------------------

    states = {
        "A": {
            "green_timer": 20,
            "in_yellow": False,
        },
        "B": {
            "green_timer": 3,
            "in_yellow": False,
        },
        "C": {
            "green_timer": 20,
            "in_yellow": True,
        },
        "D": {
            "green_timer": 20,
            "in_yellow": False,
        },
    }

    actions = np.array(
        [1, 1, 1, 0]
    )

    decisions = validator.validate_network_actions(
        actions=actions,
        intersection_states=states,
    )

    final_actions = validator.approved_actions(
        decisions
    )

    violations = validator.violation_count(
        decisions
    )

    print("\nNetwork test")
    print(
        "Proposed actions:",
        actions
    )

    print(
        "Approved actions:",
        final_actions
    )

    print(
        "Safety overrides:",
        violations
    )

    assert final_actions.shape == (4,)
    assert final_actions.tolist() == [
        1,
        0,
        0,
        0,
    ]

    assert violations == 2

    print("\nAudit file:")
    print(audit_path)

    print("\n" + "=" * 70)
    print("ALL SAFETY / GOVERNANCE TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()