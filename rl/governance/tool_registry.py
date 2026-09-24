"""
=========================================================
Traffic MARL Project

Module:
Tool Allow-List / Agent Tool Registry

Purpose:
Provides a controlled set of tools that the PPO-MARL
agent/copilot may request during operation.

Governance flow:

    PPO-MARL
        ↓
    Tool Allow-List
        ↓
    SafetyValidator
        ↓
    HumanApprovalGate
        ↓
    TrafficController / SUMO
        ↓
    AuditLogger

Design goals:
    - explicit tool allow-list
    - least-privilege execution
    - approval boundary for control actions
    - safety tools cannot be bypassed
    - traceable tool requests
    - no arbitrary Python/system/file execution

Author:
    Mohammed Piyarji
=========================================================
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, Optional


# =========================================================
# TOOL DEFINITION
# =========================================================

@dataclass(frozen=True)
class ToolDefinition:
    """
    Definition of one permitted agent tool.
    """

    name: str
    description: str
    requires_human_approval: bool
    safety_critical: bool


# =========================================================
# TOOL REQUEST
# =========================================================

@dataclass
class ToolRequest:
    """
    Auditable record of a tool request.
    """

    tool_name: str
    allowed: bool
    requires_human_approval: bool
    safety_critical: bool
    status: str
    reason: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =========================================================
# TOOL ALLOW-LIST
# =========================================================

class ToolAllowList:
    """
    Central allow-list for agent/copilot capabilities.

    Only explicitly registered traffic-control tools
    are permitted.

    No arbitrary:
        - shell execution
        - Python execution
        - filesystem access
        - code execution
        - database access
        - unrestricted network access
    """

    DEFAULT_TOOLS = (
        ToolDefinition(
            name="read_traffic_state",
            description=(
                "Read current traffic and signal state."
            ),
            requires_human_approval=False,
            safety_critical=False,
        ),

        ToolDefinition(
            name="propose_signal_action",
            description=(
                "Propose a signal action for an intersection."
            ),
            requires_human_approval=False,
            safety_critical=False,
        ),

        ToolDefinition(
            name="validate_safety",
            description=(
                "Validate a proposed signal action "
                "through the safety layer."
            ),
            requires_human_approval=False,
            safety_critical=True,
        ),

        ToolDefinition(
            name="request_human_approval",
            description=(
                "Request operator approval for a "
                "governed action."
            ),
            requires_human_approval=True,
            safety_critical=False,
        ),

        ToolDefinition(
            name="apply_approved_action",
            description=(
                "Apply an action only after governance "
                "and safety approval."
            ),
            requires_human_approval=True,
            safety_critical=True,
        ),

        ToolDefinition(
            name="emergency_stop",
            description=(
                "Stop or override normal control flow "
                "for an emergency condition."
            ),
            requires_human_approval=False,
            safety_critical=True,
        ),

        ToolDefinition(
            name="write_audit_log",
            description=(
                "Record an auditable control decision."
            ),
            requires_human_approval=False,
            safety_critical=False,
        ),
    )

    # =====================================================
    # INITIALIZATION
    # =====================================================

    def __init__(
        self,
        tools: Optional[
            Iterable[ToolDefinition]
        ] = None,
    ) -> None:

        selected = tuple(
            tools
            if tools is not None
            else self.DEFAULT_TOOLS
        )

        self._tools: Dict[
            str,
            ToolDefinition
        ] = {}

        for tool in selected:

            if not tool.name.strip():
                raise ValueError(
                    "Tool name cannot be empty"
                )

            if tool.name in self._tools:
                raise ValueError(
                    f"Duplicate tool: {tool.name}"
                )

            self._tools[
                tool.name
            ] = tool

    # =====================================================
    # ALLOWED TOOLS
    # =====================================================

    @property
    def allowed_tools(self) -> tuple[str, ...]:
        """
        Return the explicit allow-list.
        """

        return tuple(
            self._tools.keys()
        )

    # =====================================================
    # CHECK TOOL
    # =====================================================

    def is_allowed(
        self,
        tool_name: str,
    ) -> bool:
        """
        Return True only when the tool is explicitly
        registered.
        """

        return tool_name in self._tools

    # =====================================================
    # GET TOOL
    # =====================================================

    def get_tool(
        self,
        tool_name: str,
    ) -> ToolDefinition:
        """
        Return a registered tool.

        Unknown tools are rejected.
        """

        if not self.is_allowed(
            tool_name
        ):

            raise PermissionError(
                f"Tool '{tool_name}' is not "
                f"in the agent allow-list."
            )

        return self._tools[
            tool_name
        ]

    # =====================================================
    # REQUEST TOOL
    # =====================================================

    def request(
        self,
        tool_name: str,
    ) -> ToolRequest:
        """
        Check a requested tool before execution.

        This method only authorizes the capability.
        It does not execute the tool.
        """

        timestamp = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        # -------------------------------------------------
        # UNKNOWN TOOL
        # -------------------------------------------------

        if not self.is_allowed(
            tool_name
        ):

            return ToolRequest(

                tool_name=str(
                    tool_name
                ),

                allowed=False,

                requires_human_approval=False,

                safety_critical=False,

                status="DENIED",

                reason=(
                    "Tool is not in the "
                    "explicit allow-list."
                ),

                timestamp=timestamp,
            )

        # -------------------------------------------------
        # REGISTERED TOOL
        # -------------------------------------------------

        definition = self._tools[
            tool_name
        ]

        return ToolRequest(

            tool_name=definition.name,

            allowed=True,

            requires_human_approval=(
                definition
                .requires_human_approval
            ),

            safety_critical=(
                definition
                .safety_critical
            ),

            status=(
                "APPROVAL_REQUIRED"
                if definition.requires_human_approval
                else "ALLOWED"
            ),

            reason=(
                "Tool is explicitly registered "
                "in the allow-list."
            ),

            timestamp=timestamp,
        )

    # =====================================================
    # CONTROLLED EXECUTION
    # =====================================================

    def execute(
        self,
        tool_name: str,
        handler: Callable[..., Any],
        *,
        human_approved: bool = False,
        safety_validated: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Execute a registered tool only when all
        required governance conditions are satisfied.
        """

        definition = self.get_tool(
            tool_name
        )

        # -------------------------------------------------
        # SAFETY CHECK
        # -------------------------------------------------

        if (
            definition.safety_critical
            and not safety_validated
        ):

            raise PermissionError(
                f"Safety validation required "
                f"before executing '{tool_name}'."
            )

        # -------------------------------------------------
        # HUMAN APPROVAL CHECK
        # -------------------------------------------------

        if (
            definition.requires_human_approval
            and not human_approved
        ):

            raise PermissionError(
                f"Human approval required "
                f"before executing '{tool_name}'."
            )

        # -------------------------------------------------
        # HANDLER CHECK
        # -------------------------------------------------

        if not callable(handler):

            raise TypeError(
                "handler must be callable"
            )

        return handler(
            **kwargs
        )

    # =====================================================
    # DESCRIPTION
    # =====================================================

    def describe(
        self,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return allow-list information suitable for
        dashboard/reporting.
        """

        return {
            name: asdict(
                definition
            )
            for name, definition
            in self._tools.items()
        }


# =========================================================
# STANDALONE TESTS
# =========================================================

def _run_tool_registry_tests() -> None:

    registry = ToolAllowList()

    expected = {
        "read_traffic_state",
        "propose_signal_action",
        "validate_safety",
        "request_human_approval",
        "apply_approved_action",
        "emergency_stop",
        "write_audit_log",
    }

    # =====================================================
    # TEST 1
    # Explicit allow-list
    # =====================================================

    assert (
        set(registry.allowed_tools)
        == expected
    )

    # =====================================================
    # TEST 2
    # Allowed tool
    # =====================================================

    request = registry.request(
        "read_traffic_state"
    )

    assert request.allowed is True

    assert (
        request.status
        == "ALLOWED"
    )

    # =====================================================
    # TEST 3
    # Unknown tool denied
    # =====================================================

    denied = registry.request(
        "execute_shell_command"
    )

    assert denied.allowed is False

    assert (
        denied.status
        == "DENIED"
    )

    # =====================================================
    # TEST 4
    # Unknown tool cannot execute
    # =====================================================

    try:

        registry.execute(
            "execute_shell_command",
            lambda: "BAD",
        )

        raise AssertionError(
            "Unknown tool was executed"
        )

    except PermissionError:
        pass

    # =====================================================
    # TEST 5
    # Safety-critical action requires safety validation
    # =====================================================

    try:

        registry.execute(

            "apply_approved_action",

            lambda: "APPLIED",

            human_approved=True,

            safety_validated=False,
        )

        raise AssertionError(
            "Unsafe action was executed"
        )

    except PermissionError:
        pass

    # =====================================================
    # TEST 6
    # Human approval required
    # =====================================================

    try:

        registry.execute(

            "apply_approved_action",

            lambda: "APPLIED",

            human_approved=False,

            safety_validated=True,
        )

        raise AssertionError(
            "Unapproved action was executed"
        )

    except PermissionError:
        pass

    # =====================================================
    # TEST 7
    # Fully approved action
    # =====================================================

    result = registry.execute(

        "apply_approved_action",

        lambda action: action,

        human_approved=True,

        safety_validated=True,

        action=1,
    )

    assert result == 1

    # =====================================================
    # TEST 8
    # Emergency stop is safety critical
    # =====================================================

    emergency = registry.request(
        "emergency_stop"
    )

    assert emergency.allowed is True

    assert (
        emergency.safety_critical
        is True
    )

    assert (
        emergency.requires_human_approval
        is False
    )

    # =====================================================
    # TEST 9
    # Dashboard description
    # =====================================================

    description = registry.describe()

    assert (
        "validate_safety"
        in description
    )

    assert (
        description[
            "apply_approved_action"
        ][
            "requires_human_approval"
        ]
        is True
    )

    # =====================================================
    # FINAL RESULT
    # =====================================================

    print(
        "ToolAllowList tests: PASS"
    )

    print(
        "Allowed tools:"
    )

    for name in registry.allowed_tools:

        print(
            f"  - {name}"
        )

    print(
        "Unknown tool denial: PASS"
    )

    print(
        "Safety gate: PASS"
    )

    print(
        "Human approval gate: PASS"
    )

    print(
        "Approved execution: PASS"
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    _run_tool_registry_tests()