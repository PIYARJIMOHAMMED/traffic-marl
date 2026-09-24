"""
=========================================================
Traffic MARL Project
=========================================================

Module:
    Safety Validator

Purpose:
    Validate PPO-MARL traffic-signal actions before they
    are passed to the traffic-control execution layer.

Safety responsibilities:
    1. Validate action type and action space.
    2. Validate current signal state.
    3. Detect HOLD versus phase-change requests.
    4. Check minimum-green protection using the value
       supplied by the caller.
    5. Check yellow-transition protection.
    6. Produce an auditable safety decision.
    7. Never modify the trained PPO model.
    8. Never retrain the model.
    9. Never invent traffic timing values at runtime.

Action semantics:
    0 = North/South
    1 = East/West

Important architecture:
    PPO-MARL
        ↓
    SafetyValidator
        ↓
    TrafficController
        ↓
    SUMO

The TrafficController remains responsible for the actual
signal transition, yellow phase, pending action and SUMO
execution.

SafetyValidator only validates the proposed action and
reports whether the requested transition is currently
permitted.

This module does not depend on PPO implementation details.

=========================================================
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np


# =========================================================
# SAFETY DECISION
# =========================================================

@dataclass
class SafetyDecision:
    """
    Complete safety decision for one intersection.
    """

    intersection: str

    proposed_action: int
    approved_action: int

    valid_action: bool

    minimum_green_ok: bool
    yellow_transition_ok: bool

    overridden: bool

    safety_status: str
    reason: str

    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the decision to a dictionary.

        Useful for:
            - audit logs
            - API responses
            - dashboard display
            - debugging
        """

        return asdict(self)


# =========================================================
# SAFETY VALIDATOR
# =========================================================

class SafetyValidator:
    """
    Safety and governance validation layer.

    This class does NOT control SUMO directly.

    It validates the action proposed by PPO-MARL.

    Action semantics:

        0 = North/South
        1 = East/West

    Important:

        The minimum-green value is NOT hardcoded here.

        The caller must provide the current minimum-green
        requirement from the project's TrafficController /
        EnvironmentConfig.

    This prevents the safety layer from silently using a
    different timing rule than the actual traffic controller.
    """

    # -----------------------------------------------------
    # Project action space
    # -----------------------------------------------------

    VALID_ACTIONS = (0, 1)

    # -----------------------------------------------------
    # Default intersection identifiers
    #
    # These are identifiers, not traffic-control rules.
    # -----------------------------------------------------

    DEFAULT_INTERSECTIONS = (
        "A",
        "B",
        "C",
        "D",
    )

    # =====================================================
    # INITIALIZATION
    # =====================================================

    def __init__(
        self,
        min_green_steps: Optional[int] = None,
        yellow_protection: bool = True,
    ):
        """
        Parameters
        ----------
        min_green_steps:
            Minimum green duration used by the actual
            traffic-control system.

            IMPORTANT:
            This value should be supplied by the caller from
            the project's authoritative configuration.

            No traffic timing value is invented here.

        yellow_protection:
            Whether phase changes are blocked while the
            intersection is in a yellow transition.
        """

        if min_green_steps is not None:

            try:
                min_green_steps = int(min_green_steps)

            except (
                TypeError,
                ValueError,
            ) as error:

                raise ValueError(
                    "min_green_steps must be an integer "
                    "or None."
                ) from error

            if min_green_steps < 0:

                raise ValueError(
                    "min_green_steps must be >= 0."
                )

        self.min_green_steps = min_green_steps

        self.yellow_protection = bool(
            yellow_protection
        )

    # =====================================================
    # INTERNAL HELPERS
    # =====================================================

    @staticmethod
    def _timestamp() -> str:
        """
        Return an ISO-8601 UTC timestamp.
        """

        return datetime.now(
            timezone.utc
        ).isoformat()

    # -----------------------------------------------------

    @staticmethod
    def _safe_int(
        value: Any,
        default: int = 0,
    ) -> int:
        """
        Safely convert a value to integer.
        """

        try:

            return int(
                round(
                    float(value)
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            return int(default)

    # -----------------------------------------------------

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0,
    ) -> float:
        """
        Safely convert a value to finite float.
        """

        try:

            value = float(value)

            if not np.isfinite(value):

                return float(default)

            return value

        except (
            TypeError,
            ValueError,
        ):

            return float(default)

    # -----------------------------------------------------

    @classmethod
    def _normalise_action(
        cls,
        action: Any,
    ) -> Optional[int]:
        """
        Convert an action to a valid project action.

        Returns None when unavailable or invalid.
        """

        if action is None:

            return None

        try:

            value = int(
                round(
                    float(action)
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            return None

        if value not in cls.VALID_ACTIONS:

            return None

        return value

    # =====================================================
    # VALIDATE ONE ACTION
    # =====================================================

    def validate_action(
        self,
        intersection: str,
        proposed_action: int,
        current_action: Optional[int] = None,
        green_timer: float = 0.0,
        in_yellow: bool = False,
        min_green_steps: Optional[int] = None,
    ) -> SafetyDecision:
        """
        Validate one PPO action.

        Parameters
        ----------
        intersection:
            Intersection identifier.

        proposed_action:
            Action proposed by PPO-MARL.

            0 = North/South
            1 = East/West

        current_action:
            Actual current directional action.

        green_timer:
            Current green duration.

        in_yellow:
            Whether the intersection is currently in yellow.

        min_green_steps:
            Authoritative minimum-green requirement.

            If supplied, this value is used for this
            validation.

            If omitted, the validator's configured value
            is used.

        Returns
        -------
        SafetyDecision
        """

        timestamp = self._timestamp()

        # =================================================
        # RESOLVE MINIMUM-GREEN RULE
        # =================================================

        effective_min_green = (
            self.min_green_steps
            if min_green_steps is None
            else min_green_steps
        )

        if effective_min_green is None:

            return SafetyDecision(
                intersection=str(intersection),
                proposed_action=-1,
                approved_action=-1,
                valid_action=False,
                minimum_green_ok=False,
                yellow_transition_ok=False,
                overridden=True,
                safety_status="CONFIGURATION_ERROR",
                reason=(
                    "Authoritative minimum-green "
                    "requirement was not supplied."
                ),
                timestamp=timestamp,
            )

        try:

            effective_min_green = int(
                effective_min_green
            )

        except (
            TypeError,
            ValueError,
        ):

            return SafetyDecision(
                intersection=str(intersection),
                proposed_action=-1,
                approved_action=-1,
                valid_action=False,
                minimum_green_ok=False,
                yellow_transition_ok=False,
                overridden=True,
                safety_status="CONFIGURATION_ERROR",
                reason=(
                    "Minimum-green requirement is invalid."
                ),
                timestamp=timestamp,
            )

        if effective_min_green < 0:

            return SafetyDecision(
                intersection=str(intersection),
                proposed_action=-1,
                approved_action=-1,
                valid_action=False,
                minimum_green_ok=False,
                yellow_transition_ok=False,
                overridden=True,
                safety_status="CONFIGURATION_ERROR",
                reason=(
                    "Minimum-green requirement "
                    "cannot be negative."
                ),
                timestamp=timestamp,
            )

        # =================================================
        # VALIDATE PROPOSED ACTION
        # =================================================

        proposed = self._normalise_action(
            proposed_action
        )

        current = self._normalise_action(
            current_action
        )

        # -------------------------------------------------
        # Invalid / non-numeric PPO action
        # -------------------------------------------------

        if proposed is None:

            # If the current action is known, HOLD it.

            if current is not None:

                fallback_action = current

            else:

                # There is no trustworthy action that can
                # be returned when both proposed and current
                # actions are unavailable.

                fallback_action = -1

            return SafetyDecision(
                intersection=str(intersection),
                proposed_action=-1,
                approved_action=fallback_action,
                valid_action=False,
                minimum_green_ok=False,
                yellow_transition_ok=False,
                overridden=True,
                safety_status="INVALID_ACTION",
                reason=(
                    "PPO proposed an invalid or "
                    "non-numeric action."
                ),
                timestamp=timestamp,
            )

        # =================================================
        # VALIDATE CURRENT ACTION
        # =================================================

        if current is None:

            return SafetyDecision(
                intersection=str(intersection),
                proposed_action=proposed,
                approved_action=-1,
                valid_action=True,
                minimum_green_ok=False,
                yellow_transition_ok=False,
                overridden=True,
                safety_status="STATE_ERROR",
                reason=(
                    "Current signal action is unavailable "
                    "or invalid; the requested phase "
                    "transition cannot be safely validated."
                ),
                timestamp=timestamp,
            )

        # =================================================
        # READ CURRENT STATE
        # =================================================

        green_timer_value = self._safe_float(
            green_timer
        )

        yellow_state = bool(
            in_yellow
        )

        minimum_green_ok = (
            green_timer_value
            >= effective_min_green
        )

        yellow_transition_ok = (
            not yellow_state
            if self.yellow_protection
            else True
        )

        # =================================================
        # DETERMINE REQUEST TYPE
        # =================================================

        is_change_request = (
            proposed != current
        )

        # =================================================
        # HOLD
        # =================================================

        if not is_change_request:

            return SafetyDecision(
                intersection=str(intersection),
                proposed_action=proposed,
                approved_action=current,
                valid_action=True,
                minimum_green_ok=minimum_green_ok,
                yellow_transition_ok=yellow_transition_ok,
                overridden=False,
                safety_status="APPROVED",
                reason=(
                    "PPO action matches the current signal "
                    "direction; no phase change requested."
                ),
                timestamp=timestamp,
            )

        # =================================================
        # CHANGE REQUEST DURING YELLOW
        # =================================================

        if (
            self.yellow_protection
            and yellow_state
        ):

            return SafetyDecision(
                intersection=str(intersection),
                proposed_action=proposed,

                # The safety layer does not authorize a
                # phase change during yellow.
                #
                # TrafficController remains responsible for
                # the actual transition state.

                approved_action=current,

                valid_action=True,

                minimum_green_ok=minimum_green_ok,
                yellow_transition_ok=False,

                overridden=True,

                safety_status="DEFERRED_YELLOW",

                reason=(
                    "Phase change request received while "
                    "yellow transition is active; current "
                    "signal direction must be maintained "
                    "until the transition is complete."
                ),

                timestamp=timestamp,
            )

        # =================================================
        # CHANGE REQUEST BEFORE MINIMUM GREEN
        # =================================================

        if not minimum_green_ok:

            return SafetyDecision(
                intersection=str(intersection),
                proposed_action=proposed,
                approved_action=current,

                valid_action=True,

                minimum_green_ok=False,
                yellow_transition_ok=yellow_transition_ok,

                overridden=True,

                safety_status="DEFERRED_MINIMUM_GREEN",

                reason=(
                    "Phase change request is valid, but "
                    "the authoritative minimum-green "
                    "duration has not elapsed."
                ),

                timestamp=timestamp,
            )

        # =================================================
        # CHANGE REQUEST APPROVED
        # =================================================

        return SafetyDecision(
            intersection=str(intersection),
            proposed_action=proposed,
            approved_action=proposed,

            valid_action=True,

            minimum_green_ok=True,
            yellow_transition_ok=yellow_transition_ok,

            overridden=False,

            safety_status="APPROVED",

            reason=(
                "Phase change request passed safety "
                "validation."
            ),

            timestamp=timestamp,
        )

    # =====================================================
    # VALIDATE NETWORK ACTIONS
    # =====================================================

    def validate_network_actions(
        self,
        actions,
        intersection_states: Optional[Dict[str, Dict[str, Any]]] = None,
        intersection_names: Optional[List[str]] = None,
    ) -> Dict[str, SafetyDecision]:
        """
        Validate PPO actions for the complete network.

        Expected state structure:

            {
                "A": {
                    "current_action": 0,
                    "current_phase": 0,
                    "green_timer": ...,
                    "in_yellow": False,
                    "min_green_steps": ...
                },
                ...
            }

        IMPORTANT:

            min_green_steps can be supplied per intersection.

            This allows the validator to use the actual
            controller/configuration value rather than
            inventing a fixed project rule.
        """

        actions = np.asarray(
            actions
        ).reshape(-1)

        # -------------------------------------------------
        # Intersection identifiers
        # -------------------------------------------------

        if intersection_names is None:

            intersection_names = list(
                self.DEFAULT_INTERSECTIONS
            )

        # -------------------------------------------------
        # Validate action count
        # -------------------------------------------------

        if len(actions) != len(
            intersection_names
        ):

            raise ValueError(
                "Number of actions does not match "
                "number of intersections. "
                f"Received {len(actions)} actions for "
                f"{len(intersection_names)} intersections."
            )

        if intersection_states is None:

            intersection_states = {}

        decisions: Dict[
            str,
            SafetyDecision
        ] = {}

        # =================================================
        # VALIDATE EACH INTERSECTION
        # =================================================

        for index, intersection in enumerate(
            intersection_names
        ):

            state = intersection_states.get(
                intersection,
                {}
            )

            if not isinstance(
                state,
                dict
            ):

                state = {}

            current_action = state.get(
                "current_action",
                None
            )

            green_timer = state.get(
                "green_timer",
                0.0
            )

            in_yellow = state.get(
                "in_yellow",
                False
            )

            # -------------------------------------------------
            # Use the authoritative per-intersection timing
            # when supplied by the controller/application.
            # -------------------------------------------------

            state_min_green = state.get(
                "min_green_steps",
                self.min_green_steps
            )

            decisions[
                str(intersection)
            ] = self.validate_action(
                intersection=str(intersection),
                proposed_action=actions[index],
                current_action=current_action,
                green_timer=green_timer,
                in_yellow=in_yellow,
                min_green_steps=state_min_green,
            )

        return decisions

    # =====================================================
    # EXTRACT APPROVED ACTIONS
    # =====================================================

    @staticmethod
    def approved_actions(
        decisions: Dict[str, SafetyDecision],
        intersection_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Return approved actions in the requested
        intersection order.

        The default order remains A/B/C/D because that is
        the project's four-agent ordering.
        """

        if intersection_names is None:

            intersection_names = [
                "A",
                "B",
                "C",
                "D",
            ]

        missing = [
            name
            for name in intersection_names
            if name not in decisions
        ]

        if missing:

            raise ValueError(
                "Missing safety decisions for "
                f"intersections: {missing}"
            )

        approved = []

        for name in intersection_names:

            action = decisions[
                name
            ].approved_action

            if action not in (
                0,
                1,
            ):

                raise ValueError(
                    f"Intersection {name} has no "
                    f"safe approved action: {action}"
                )

            approved.append(
                action
            )

        return np.asarray(
            approved,
            dtype=np.int64,
        )

    # =====================================================
    # SAFETY INTERVENTION COUNT
    # =====================================================

    @staticmethod
    def intervention_count(
        decisions: Dict[str, SafetyDecision]
    ) -> int:
        """
        Count valid PPO requests that were deferred by
        the safety layer.

        This includes:

            DEFERRED_MINIMUM_GREEN
            DEFERRED_YELLOW

        It does NOT count invalid PPO actions.
        """

        return sum(
            1
            for decision in decisions.values()
            if (
                decision.valid_action
                and decision.overridden
            )
        )

    # =====================================================
    # VIOLATION COUNT
    # =====================================================

    @staticmethod
    def violation_count(
        decisions: Dict[str, SafetyDecision]
    ) -> int:
        """
        Count invalid PPO action proposals.

        Important distinction:

            valid + deferred
                = safety intervention

            invalid action
                = safety violation

        A valid phase-change request blocked because of
        minimum-green or yellow protection is NOT a
        violation.
        """

        return sum(
            1
            for decision in decisions.values()
            if not decision.valid_action
        )

    # =====================================================
    # INVALID ACTION COUNT
    # =====================================================

    @staticmethod
    def invalid_action_count(
        decisions: Dict[str, SafetyDecision]
    ) -> int:
        """
        Count invalid action proposals.
        """

        return sum(
            1
            for decision in decisions.values()
            if not decision.valid_action
        )

    # =====================================================
    # MINIMUM GREEN BLOCK COUNT
    # =====================================================

    @staticmethod
    def minimum_green_block_count(
        decisions: Dict[str, SafetyDecision]
    ) -> int:
        """
        Count requests deferred because minimum green
        had not elapsed.
        """

        return sum(
            1
            for decision in decisions.values()
            if (
                decision.valid_action
                and decision.overridden
                and decision.safety_status
                == "DEFERRED_MINIMUM_GREEN"
            )
        )

    # =====================================================
    # YELLOW BLOCK COUNT
    # =====================================================

    @staticmethod
    def yellow_block_count(
        decisions: Dict[str, SafetyDecision]
    ) -> int:
        """
        Count requests deferred because a yellow
        transition was active.
        """

        return sum(
            1
            for decision in decisions.values()
            if (
                decision.valid_action
                and decision.overridden
                and decision.safety_status
                == "DEFERRED_YELLOW"
            )
        )

    # =====================================================
    # STATE ERROR COUNT
    # =====================================================

    @staticmethod
    def state_error_count(
        decisions: Dict[str, SafetyDecision]
    ) -> int:
        """
        Count decisions where the current signal state
        was unavailable or invalid.
        """

        return sum(
            1
            for decision in decisions.values()
            if decision.safety_status
            == "STATE_ERROR"
        )

    # =====================================================
    # CONFIGURATION ERROR COUNT
    # =====================================================

    @staticmethod
    def configuration_error_count(
        decisions: Dict[str, SafetyDecision]
    ) -> int:
        """
        Count configuration errors.
        """

        return sum(
            1
            for decision in decisions.values()
            if decision.safety_status
            == "CONFIGURATION_ERROR"
        )


# =========================================================
# STANDALONE SAFETY TESTS
# =========================================================

def _run_safety_tests() -> None:
    """
    Standalone verification suite.

    These tests:

        - do not start SUMO
        - do not load PPO
        - do not retrain PPO
        - do not modify the model

    The test timing value is explicitly supplied to the
    validator. It is test configuration, not a hidden
    runtime traffic rule.
    """

    TEST_MIN_GREEN = 10

    validator = SafetyValidator(
        min_green_steps=TEST_MIN_GREEN,
        yellow_protection=True,
    )

    # =====================================================
    # TEST 1
    # HOLD CURRENT ACTION
    # =====================================================

    decision = validator.validate_action(
        intersection="A",
        proposed_action=0,
        current_action=0,
        green_timer=5,
        in_yellow=False,
    )

    assert decision.valid_action is True
    assert decision.approved_action == 0
    assert decision.overridden is False
    assert decision.safety_status == "APPROVED"

    # =====================================================
    # TEST 2
    # CHANGE BEFORE MINIMUM GREEN
    # =====================================================

    decision = validator.validate_action(
        intersection="A",
        proposed_action=1,
        current_action=0,
        green_timer=5,
        in_yellow=False,
    )

    assert decision.valid_action is True
    assert decision.approved_action == 0
    assert decision.overridden is True
    assert decision.minimum_green_ok is False
    assert decision.safety_status == (
        "DEFERRED_MINIMUM_GREEN"
    )

    # =====================================================
    # TEST 3
    # CHANGE AFTER MINIMUM GREEN
    # =====================================================

    decision = validator.validate_action(
        intersection="A",
        proposed_action=1,
        current_action=0,
        green_timer=15,
        in_yellow=False,
    )

    assert decision.valid_action is True
    assert decision.approved_action == 1
    assert decision.overridden is False
    assert decision.minimum_green_ok is True
    assert decision.yellow_transition_ok is True
    assert decision.safety_status == "APPROVED"

    # =====================================================
    # TEST 4
    # CHANGE DURING YELLOW
    # =====================================================

    decision = validator.validate_action(
        intersection="A",
        proposed_action=1,
        current_action=0,
        green_timer=15,
        in_yellow=True,
    )

    assert decision.valid_action is True
    assert decision.approved_action == 0
    assert decision.overridden is True
    assert decision.yellow_transition_ok is False
    assert decision.safety_status == (
        "DEFERRED_YELLOW"
    )

    # =====================================================
    # TEST 5
    # HOLD DURING YELLOW
    # =====================================================

    decision = validator.validate_action(
        intersection="A",
        proposed_action=0,
        current_action=0,
        green_timer=15,
        in_yellow=True,
    )

    assert decision.valid_action is True
    assert decision.approved_action == 0
    assert decision.overridden is False
    assert decision.safety_status == "APPROVED"

    # =====================================================
    # TEST 6
    # INVALID ACTION
    # =====================================================

    decision = validator.validate_action(
        intersection="A",
        proposed_action=7,
        current_action=1,
        green_timer=20,
        in_yellow=False,
    )

    assert decision.valid_action is False
    assert decision.approved_action == 1
    assert decision.overridden is True
    assert decision.safety_status == "INVALID_ACTION"

    # =====================================================
    # TEST 7
    # INVALID NON-NUMERIC ACTION
    # =====================================================

    decision = validator.validate_action(
        intersection="A",
        proposed_action="invalid",
        current_action=1,
        green_timer=20,
        in_yellow=False,
    )

    assert decision.valid_action is False
    assert decision.approved_action == 1
    assert decision.overridden is True
    assert decision.safety_status == "INVALID_ACTION"

    # =====================================================
    # TEST 8
    # MISSING CURRENT STATE
    # =====================================================

    decision = validator.validate_action(
        intersection="A",
        proposed_action=1,
        current_action=None,
        green_timer=20,
        in_yellow=False,
    )

    assert decision.valid_action is True
    assert decision.approved_action == -1
    assert decision.overridden is True
    assert decision.safety_status == "STATE_ERROR"

    # =====================================================
    # TEST 9
    # NETWORK VALIDATION
    # =====================================================

    actions = np.asarray(
        [1, 0, 1, 0],
        dtype=np.int64,
    )

    states = {
        "A": {
            "current_action": 0,
            "current_phase": 0,
            "green_timer": 15,
            "in_yellow": False,
            "min_green_steps": TEST_MIN_GREEN,
        },

        "B": {
            "current_action": 0,
            "current_phase": 0,
            "green_timer": 5,
            "in_yellow": False,
            "min_green_steps": TEST_MIN_GREEN,
        },

        "C": {
            "current_action": 1,
            "current_phase": 2,
            "green_timer": 20,
            "in_yellow": False,
            "min_green_steps": TEST_MIN_GREEN,
        },

        "D": {
            "current_action": 1,
            "current_phase": 2,
            "green_timer": 20,
            "in_yellow": True,
            "min_green_steps": TEST_MIN_GREEN,
        },
    }

    decisions = validator.validate_network_actions(
        actions=actions,
        intersection_states=states,
    )

    # A:
    # 0 -> 1
    # minimum green satisfied
    # approved

    assert decisions["A"].approved_action == 1
    assert decisions["A"].overridden is False

    # B:
    # 0 -> 0
    # HOLD

    assert decisions["B"].approved_action == 0
    assert decisions["B"].overridden is False

    # C:
    # 1 -> 1
    # HOLD

    assert decisions["C"].approved_action == 1
    assert decisions["C"].overridden is False

    # D:
    # 1 -> 0
    # yellow active
    # deferred

    assert decisions["D"].approved_action == 1
    assert decisions["D"].overridden is True
    assert decisions["D"].safety_status == (
        "DEFERRED_YELLOW"
    )

    # =====================================================
    # TEST 10
    # APPROVED ACTION ARRAY
    # =====================================================

    approved = validator.approved_actions(
        decisions
    )

    expected = np.asarray(
        [1, 0, 1, 1],
        dtype=np.int64,
    )

    assert np.array_equal(
        approved,
        expected,
    )

    # =====================================================
    # TEST 11
    # COUNTERS
    # =====================================================

    intervention_count = (
        validator.intervention_count(
            decisions
        )
    )

    violation_count = (
        validator.violation_count(
            decisions
        )
    )

    minimum_green_count = (
        validator.minimum_green_block_count(
            decisions
        )
    )

    yellow_count = (
        validator.yellow_block_count(
            decisions
        )
    )

    # Only D is overridden in this network test:
    # A = approved change
    # B = HOLD
    # C = HOLD
    # D = yellow-protected change request
    assert intervention_count == 1
    assert violation_count == 0
    assert minimum_green_count == 0
    assert yellow_count == 1

    # =====================================================
    # TEST 12
    # AUTHORITATIVE MINIMUM-GREEN VALUE
    # =====================================================

    # The validator instance is configured with 10, but
    # this call explicitly supplies 20. The supplied value
    # must be authoritative for this decision.

    decision = validator.validate_action(
        intersection="A",
        proposed_action=1,
        current_action=0,
        green_timer=15,
        in_yellow=False,
        min_green_steps=20,
    )

    assert decision.minimum_green_ok is False
    assert decision.approved_action == 0
    assert decision.safety_status == (
        "DEFERRED_MINIMUM_GREEN"
    )

    print(
        "SafetyValidator tests: PASS"
    )

    print(
        f"Interventions: {intervention_count}"
    )

    print(
        f"Violations: {violation_count}"
    )

    print(
        f"Minimum-green deferrals: "
        f"{minimum_green_count}"
    )

    print(
        f"Yellow deferrals: "
        f"{yellow_count}"
    )

    print(
        f"Approved actions: "
        f"{approved.tolist()}"
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    _run_safety_tests()