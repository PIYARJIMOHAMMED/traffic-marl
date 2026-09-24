"""
Traffic control service.

Provides a single controlled interface between the dashboard/API
and the traffic simulation.

Control flow:

Dashboard/API
    -> requested actions
    -> safety validation
    -> human approval
    -> audit
    -> simulation

No dashboard action bypasses the safety layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np


@dataclass
class ControlResult:
    requested_actions: Dict[str, int]
    approved_actions: Dict[str, int]
    overrides: int
    violations: int
    approval_required: bool
    status: str
    reason: str
    timestamp: str

    def to_dict(self):
        return {
            "requested_actions": self.requested_actions,
            "approved_actions": self.approved_actions,
            "overrides": self.overrides,
            "violations": self.violations,
            "approval_required": self.approval_required,
            "status": self.status,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


class TrafficControlService:
    """
    Safe dashboard control service.

    This class is deliberately independent from Streamlit so it can
    also be called by FastAPI.
    """

    AGENTS = ("A", "B", "C", "D")

    def __init__(
        self,
        safety_validator=None,
        approval_gate=None,
        audit_logger=None,
    ):
        self.safety_validator = safety_validator
        self.approval_gate = approval_gate
        self.audit_logger = audit_logger

        self.last_requested_actions = {
            agent: 0 for agent in self.AGENTS
        }

        self.last_approved_actions = {
            agent: 0 for agent in self.AGENTS
        }

        self.total_overrides = 0
        self.total_violations = 0

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    def validate_actions(
        self,
        actions: Dict[str, int],
    ) -> Dict[str, int]:

        if not isinstance(actions, dict):
            raise ValueError("actions must be a dictionary")

        unknown = set(actions) - set(self.AGENTS)

        if unknown:
            raise ValueError(
                f"Unknown intersection(s): {sorted(unknown)}"
            )

        result = {}

        for agent in self.AGENTS:

            value = actions.get(
                agent,
                self.last_approved_actions[agent],
            )

            if isinstance(value, bool):
                value = int(value)

            value = int(value)

            if value not in (0, 1):
                raise ValueError(
                    f"Action for {agent} must be 0 or 1."
                )

            result[agent] = value

        return result

    # ---------------------------------------------------------
    # SAFE CONTROL
    # ---------------------------------------------------------

    def request_control(
        self,
        actions: Dict[str, int],
        human_approved: bool = False,
    ) -> ControlResult:

        requested = self.validate_actions(actions)

        self.last_requested_actions = requested.copy()

        # -----------------------------------------------------
        # If the safety stack is unavailable, fail closed.
        # -----------------------------------------------------

        if self.safety_validator is None:
            return ControlResult(
                requested_actions=requested,
                approved_actions=self.last_approved_actions.copy(),
                overrides=0,
                violations=1,
                approval_required=False,
                status="REJECTED",
                reason="Safety validator is not configured.",
                timestamp=self._timestamp(),
            )

        # -----------------------------------------------------
        # Convert dictionary -> ordered vector
        # -----------------------------------------------------

        vector = np.asarray(
            [requested[a] for a in self.AGENTS],
            dtype=np.int32,
        )

        # -----------------------------------------------------
        # Safety validation
        # -----------------------------------------------------

        decisions = self.safety_validator.validate_network_actions(
            vector
        )

        approved_vector = self.safety_validator.approved_actions(
            decisions
        )

        approved = {
            agent: int(approved_vector[index])
            for index, agent in enumerate(self.AGENTS)
        }

        overrides = sum(
            1
            for decision in decisions
            if getattr(decision, "overridden", False)
        )

        violations = sum(
            1
            for decision in decisions
            if not getattr(decision, "valid_action", True)
        )

        self.total_overrides += overrides
        self.total_violations += violations

        # -----------------------------------------------------
        # Human approval
        # -----------------------------------------------------

        if self.approval_gate is not None:

            final_actions = {}

            for index, agent in enumerate(self.AGENTS):

                decision = self.approval_gate.process(
                    decisions[index],
                    human_approved=human_approved,
                )

                final_actions[agent] = int(
                    decision.final_action
                )

                if self.audit_logger is not None:
                    self.audit_logger.log(decision)

        else:
            final_actions = approved

        self.last_approved_actions = final_actions.copy()

        rejected = (
            final_actions != requested
            and overrides == 0
        )

        if violations > 0:
            status = "SAFETY_OVERRIDE"

        elif rejected:
            status = "HUMAN_REJECTED"

        else:
            status = "APPROVED"

        return ControlResult(
            requested_actions=requested,
            approved_actions=final_actions,
            overrides=overrides,
            violations=violations,
            approval_required=(
                self.approval_gate is not None
                and not human_approved
            ),
            status=status,
            reason="Control request processed through governance layer.",
            timestamp=self._timestamp(),
        )

    # ---------------------------------------------------------
    # STATUS
    # ---------------------------------------------------------

    def status(self):

        return {
            "requested_actions": self.last_requested_actions.copy(),
            "approved_actions": self.last_approved_actions.copy(),
            "total_overrides": self.total_overrides,
            "total_violations": self.total_violations,
        }

    @staticmethod
    def _timestamp():

        return datetime.now(
            timezone.utc
        ).isoformat()