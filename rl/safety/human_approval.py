from __future__ import annotations
from pathlib import Path
import sys
import os

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))




from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict

from rl.safety.safety_validator import SafetyDecision


@dataclass
class ApprovalDecision:
    """
    Human-governance decision for one intersection action.

    Governance hierarchy:

        PPO proposal
            ↓
        SafetyValidator
            ↓
        HumanApprovalGate
            ↓
        Final action

    Important:
        Human approval can govern a safety-approved action,
        but it can NEVER override the SafetyValidator.
    """

    intersection: str

    # Action proposed by PPO-MARL.
    proposed_action: int

    # Action approved by the safety layer.
    safety_approved_action: int

    # Action ultimately passed to the traffic environment.
    final_action: int

    # Governance information.
    human_approval_required: bool
    human_approved: bool

    # Audit/status information.
    status: str
    reason: str

    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Return this approval decision as a dictionary."""
        return asdict(self)


class HumanApprovalGate:
    """
    Human-governance layer between the safety validator
    and the traffic controller.

    Three supported modes:

    1. automatic
       Safety-approved actions execute automatically.

    2. approval_required
       A human operator must explicitly approve the
       safety-approved action when the application
       requests human review.

    3. emergency
       The safety-approved action executes immediately.
       Human approval cannot delay a safety-critical action.

    Governance hierarchy:

        PPO
         ↓
        SafetyValidator
         ↓
        HumanApprovalGate
         ↓
        TrafficEnvironment
         ↓
        SUMO

    Safety always has higher priority than human preference.
    """

    ALLOWED_MODES = {
        "automatic",
        "approval_required",
        "emergency",
    }

    def __init__(
        self,
        mode: str = "automatic",
    ):

        if mode not in self.ALLOWED_MODES:
            raise ValueError(
                f"Invalid approval mode '{mode}'. "
                f"Allowed modes: "
                f"{sorted(self.ALLOWED_MODES)}"
            )

        self.mode = mode

    # ========================================================
    # INTERNAL HELPERS
    # ========================================================

    @staticmethod
    def _timestamp() -> str:
        """Return the current UTC timestamp."""
        return datetime.now(
            timezone.utc
        ).isoformat()

    @staticmethod
    def _safe_action(
        action: Any,
    ) -> int:
        """
        Convert an action to the project's valid action space.

        Project action space:

            0 = North/South
            1 = East/West
        """

        try:
            value = int(action)
        except (
            TypeError,
            ValueError,
        ):
            return 0

        if value not in (0, 1):
            return 0

        return value

    # ========================================================
    # PROCESS ONE SAFETY DECISION
    # ========================================================

    def process(
        self,
        safety_decision: SafetyDecision,
        human_approved: bool = True,
    ) -> ApprovalDecision:
        """
        Process one SafetyDecision through the governance layer.

        Parameters
        ----------
        safety_decision:
            Decision already produced by SafetyValidator.

        human_approved:
            Human decision used when the application requests
            approval in approval_required mode.

            In automatic/emergency mode this value is ignored.

        Returns
        -------
        ApprovalDecision
            Complete auditable governance decision.
        """

        timestamp = self._timestamp()

        proposed_action = self._safe_action(
            safety_decision.proposed_action
        )

        safety_action = self._safe_action(
            safety_decision.approved_action
        )

        # ====================================================
        # RULE 1 — SAFETY OVERRIDE ALWAYS HAS PRIORITY
        # ====================================================
        #
        # If SafetyValidator blocked the PPO proposal,
        # HumanApprovalGate must NOT restore the unsafe PPO
        # action.
        #
        # The safety-approved action is executed directly.
        # ====================================================

        if safety_decision.overridden:

            return ApprovalDecision(
                intersection=safety_decision.intersection,

                proposed_action=proposed_action,

                safety_approved_action=safety_action,

                final_action=safety_action,

                human_approval_required=False,

                human_approved=False,

                status="SAFETY_OVERRIDE",

                reason=(
                    "SafetyValidator overrode the PPO "
                    "proposal; safety-approved action "
                    "executed without human override"
                ),

                timestamp=timestamp,
            )

        # ====================================================
        # RULE 2 — EMERGENCY MODE
        # ====================================================
        #
        # Emergency mode does not wait for a human.
        #
        # The safety-approved action is executed immediately.
        # ====================================================

        if self.mode == "emergency":

            return ApprovalDecision(
                intersection=safety_decision.intersection,

                proposed_action=proposed_action,

                safety_approved_action=safety_action,

                final_action=safety_action,

                human_approval_required=False,

                human_approved=False,

                status="EMERGENCY_APPROVED",

                reason=(
                    "Emergency governance mode: "
                    "safety-approved action executed "
                    "immediately without waiting for "
                    "human approval"
                ),

                timestamp=timestamp,
            )

        # ====================================================
        # RULE 3 — AUTOMATIC MODE
        # ====================================================
        #
        # Safe actions are automatically accepted.
        # ====================================================

        if self.mode == "automatic":

            return ApprovalDecision(
                intersection=safety_decision.intersection,

                proposed_action=proposed_action,

                safety_approved_action=safety_action,

                final_action=safety_action,

                human_approval_required=False,

                human_approved=True,

                status="AUTO_APPROVED",

                reason=(
                    "Safety-approved action automatically "
                    "accepted in automatic governance mode"
                ),

                timestamp=timestamp,
            )

        # ====================================================
        # RULE 4 — HUMAN APPROVAL REQUIRED
        # ====================================================
        #
        # At this point:
        #
        # - action is safety-approved
        # - governance mode requires human review
        #
        # Human approval determines whether the proposal
        # receives governance approval.
        #
        # IMPORTANT:
        # Human rejection does NOT force action 0.
        #
        # Action 0 is a real traffic direction, so forcing
        # it would be an incorrect rejection implementation.
        #
        # The safety-approved action remains the executable
        # safe action. The rejection is recorded in the audit.
        #
        # A future "reject = hold current signal" policy can
        # be added by explicitly supplying current_action.
        # ====================================================

        if not bool(human_approved):

            return ApprovalDecision(
                intersection=safety_decision.intersection,

                proposed_action=proposed_action,

                safety_approved_action=safety_action,

                final_action=safety_action,

                human_approval_required=True,

                human_approved=False,

                status="HUMAN_REJECTED",

                reason=(
                    "Human operator rejected the proposed "
                    "action; safety-approved action retained "
                    "to prevent the governance layer from "
                    "introducing an unsafe or arbitrary "
                    "traffic-signal command"
                ),

                timestamp=timestamp,
            )

        # ====================================================
        # RULE 5 — HUMAN APPROVED
        # ====================================================

        return ApprovalDecision(
            intersection=safety_decision.intersection,

            proposed_action=proposed_action,

            safety_approved_action=safety_action,

            final_action=safety_action,

            human_approval_required=True,

            human_approved=True,

            status="HUMAN_APPROVED",

            reason=(
                "Human operator approved the "
                "safety-approved action"
            ),

            timestamp=timestamp,
        )


# ============================================================
# STANDALONE GOVERNANCE TESTS
# ============================================================

def _make_decision(
    intersection: str = "A",
    proposed_action: int = 1,
    approved_action: int = 1,
    overridden: bool = False,
    valid_action: bool = True,
) -> SafetyDecision:
    """
    Create a SafetyDecision for standalone governance tests.
    """

    return SafetyDecision(
        intersection=intersection,

        proposed_action=proposed_action,

        approved_action=approved_action,

        valid_action=valid_action,

        minimum_green_ok=True,

        yellow_transition_ok=True,

        overridden=overridden,

        safety_status=(
            "OVERRIDDEN"
            if overridden
            else "APPROVED"
        ),

        reason=(
            "Safety test override"
            if overridden
            else "Safety test approved"
        ),

        timestamp=datetime.now(
            timezone.utc
        ).isoformat(),
    )


def _run_governance_tests() -> None:
    """
    Standalone HumanApprovalGate verification.

    These tests do not start SUMO and do not retrain PPO.
    """

    # ========================================================
    # TEST 1 — AUTOMATIC MODE
    # ========================================================

    gate = HumanApprovalGate(
        mode="automatic"
    )

    safety_decision = _make_decision(
        proposed_action=1,
        approved_action=1,
        overridden=False,
    )

    approval = gate.process(
        safety_decision,
        human_approved=True,
    )

    assert approval.final_action == 1
    assert approval.human_approval_required is False
    assert approval.human_approved is True
    assert approval.status == "AUTO_APPROVED"

    # ========================================================
    # TEST 2 — APPROVAL REQUIRED + APPROVED
    # ========================================================

    gate = HumanApprovalGate(
        mode="approval_required"
    )

    safety_decision = _make_decision(
        proposed_action=1,
        approved_action=1,
        overridden=False,
    )

    approval = gate.process(
        safety_decision,
        human_approved=True,
    )

    assert approval.final_action == 1
    assert approval.human_approval_required is True
    assert approval.human_approved is True
    assert approval.status == "HUMAN_APPROVED"

    # ========================================================
    # TEST 3 — APPROVAL REQUIRED + REJECTED
    # ========================================================
    #
    # IMPORTANT:
    # Rejection must NOT force action 0.
    #
    # The safe approved action remains the final executable
    # action because the gate does not have current_action.
    # ========================================================

    gate = HumanApprovalGate(
        mode="approval_required"
    )

    safety_decision = _make_decision(
        proposed_action=1,
        approved_action=1,
        overridden=False,
    )

    approval = gate.process(
        safety_decision,
        human_approved=False,
    )

    assert approval.final_action == 1
    assert approval.human_approval_required is True
    assert approval.human_approved is False
    assert approval.status == "HUMAN_REJECTED"

    # ========================================================
    # TEST 4 — SAFETY OVERRIDE BEATS HUMAN APPROVAL
    # ========================================================
    #
    # PPO proposes 1.
    # Safety layer blocks it and approves 0.
    # Human says TRUE.
    #
    # Human MUST NOT restore action 1.
    # ========================================================

    gate = HumanApprovalGate(
        mode="approval_required"
    )

    safety_decision = _make_decision(
        proposed_action=1,
        approved_action=0,
        overridden=True,
    )

    approval = gate.process(
        safety_decision,
        human_approved=True,
    )

    assert approval.final_action == 0
    assert approval.status == "SAFETY_OVERRIDE"
    assert approval.human_approval_required is False

    # ========================================================
    # TEST 5 — SAFETY OVERRIDE BEATS HUMAN REJECTION
    # ========================================================

    gate = HumanApprovalGate(
        mode="approval_required"
    )

    safety_decision = _make_decision(
        proposed_action=1,
        approved_action=0,
        overridden=True,
    )

    approval = gate.process(
        safety_decision,
        human_approved=False,
    )

    assert approval.final_action == 0
    assert approval.status == "SAFETY_OVERRIDE"

    # ========================================================
    # TEST 6 — EMERGENCY MODE
    # ========================================================

    gate = HumanApprovalGate(
        mode="emergency"
    )

    safety_decision = _make_decision(
        proposed_action=1,
        approved_action=1,
        overridden=False,
    )

    approval = gate.process(
        safety_decision,
        human_approved=False,
    )

    assert approval.final_action == 1
    assert approval.human_approval_required is False
    assert approval.human_approved is False
    assert approval.status == "EMERGENCY_APPROVED"

    # ========================================================
    # TEST 7 — EMERGENCY MODE WITH SAFETY OVERRIDE
    # ========================================================

    gate = HumanApprovalGate(
        mode="emergency"
    )

    safety_decision = _make_decision(
        proposed_action=1,
        approved_action=0,
        overridden=True,
    )

    approval = gate.process(
        safety_decision,
        human_approved=True,
    )

    assert approval.final_action == 0
    assert approval.status == "SAFETY_OVERRIDE"

    # ========================================================
    # TEST 8 — INVALID GOVERNANCE MODE
    # ========================================================

    try:

        HumanApprovalGate(
            mode="invalid_mode"
        )

        raise AssertionError(
            "Invalid governance mode was accepted"
        )

    except ValueError:

        pass

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print(
        "HumanApprovalGate tests: PASS"
    )

    print(
        "Automatic mode: PASS"
    )

    print(
        "Human approval: PASS"
    )

    print(
        "Human rejection safety handling: PASS"
    )

    print(
        "Safety override priority: PASS"
    )

    print(
        "Emergency mode: PASS"
    )


if __name__ == "__main__":
    _run_governance_tests()