"""
=========================================================
Traffic MARL Project

Module:
Multi-Agent Traffic Controller

Description:
Controls one traffic signal in the
Multi-Agent Reinforcement Learning system.

Each intersection owns one independent controller.

Important control semantics:
    ACTION_NORTH_SOUTH = 0
    ACTION_EAST_WEST   = 1

The controller is state-aware:
- Repeating the current action does NOT restart the SUMO phase.
- A phase change is latched before entering yellow.
- The requested action cannot change during yellow.
- Yellow completion activates the latched target phase.
- Minimum-green protection is enforced before a switch.
=========================================================
"""

from dataclasses import dataclass

import traci

from rl.environments.config import EnvironmentConfig


@dataclass
class ControllerStatistics:
    """Runtime statistics for one controller."""

    total_switches: int = 0
    total_steps: int = 0
    forced_switches: int = 0
    rejected_actions: int = 0


class TrafficController:
    """
    State-aware controller for one SUMO traffic light.

    Action semantics:
        0 -> NORTH/SOUTH
        1 -> EAST/WEST

    Critical design rule:
        An RL action is a REQUEST, not a direct repeated SUMO
        setPhase() command.

    Therefore:
        same action  -> hold current phase
        new action   -> enter yellow, latch target, then switch
    """

    def __init__(self, traffic_light_id: str):
        self.traffic_light_id = traffic_light_id

        self.current_action = (
            EnvironmentConfig.ACTION_NORTH_SOUTH
        )

        self.current_phase = (
            EnvironmentConfig.GREEN_PHASE_NS
        )

        self.green_timer = 0
        self.yellow_timer = 0
        self.in_yellow = False

        # CRITICAL:
        # Once yellow starts, the target action is latched.
        # PPO is allowed to change its next prediction, but that
        # new prediction must NOT cancel the transition already
        # in progress.
        self.pending_action = None

        self.statistics = ControllerStatistics()

    # =====================================================
    # TIMER
    # =====================================================

    def update(self):
        """
        Called once per environment simulation step.
        """

        self.statistics.total_steps += 1

        if self.in_yellow:
            self.yellow_timer += 1
        else:
            self.green_timer += 1

    def reset_green_timer(self):
        self.green_timer = 0

    def reset_yellow_timer(self):
        self.yellow_timer = 0

    # =====================================================
    # SAFETY
    # =====================================================

    def can_switch(self) -> bool:
        """
        Enforce the configured minimum green time.
        """

        return (
            self.green_timer
            >= EnvironmentConfig.MIN_GREEN_TIME
        )

    def yellow_finished(self) -> bool:
        """
        Check whether the controller has completed the
        configured yellow interval.
        """

        return (
            self.yellow_timer
            >= EnvironmentConfig.YELLOW_TIME
        )

    # =====================================================
    # PHASE
    # =====================================================

    def get_phase(self):
        """
        Return the actual phase reported by SUMO.
        """

        return traci.trafficlight.getPhase(
            self.traffic_light_id
        )

    def set_phase(self, phase):
        """
        Set a SUMO phase and keep internal state synchronized.
        """

        phase = int(phase)

        traci.trafficlight.setPhase(
            self.traffic_light_id,
            phase
        )

        self.current_phase = phase

    def sync_from_sumo(self):
        """
        Synchronize the controller's internal phase state with
        the actual SUMO traffic light.

        This protects the controller from stale internal state
        if another project component changes the TLS phase.
        """

        phase = int(
            traci.trafficlight.getPhase(
                self.traffic_light_id
            )
        )

        self.current_phase = phase

        if phase == EnvironmentConfig.GREEN_PHASE_NS:
            self.current_action = (
                EnvironmentConfig.ACTION_NORTH_SOUTH
            )
            self.in_yellow = False

        elif phase == EnvironmentConfig.GREEN_PHASE_EW:
            self.current_action = (
                EnvironmentConfig.ACTION_EAST_WEST
            )
            self.in_yellow = False

        elif phase == EnvironmentConfig.YELLOW_PHASE_NS:
            self.in_yellow = True

        elif phase == EnvironmentConfig.YELLOW_PHASE_EW:
            self.in_yellow = True

    # =====================================================
    # YELLOW
    # =====================================================

    def start_yellow(self, next_action: int):
        """
        Start the yellow transition for the requested
        target direction.

        The target action is latched here.
        """

        next_action = int(next_action)

        # The target must be different from the current direction.
        if next_action == self.current_action:
            return False

        if self.current_action == (
            EnvironmentConfig.ACTION_NORTH_SOUTH
        ):
            self.set_phase(
                EnvironmentConfig.YELLOW_PHASE_NS
            )
        else:
            self.set_phase(
                EnvironmentConfig.YELLOW_PHASE_EW
            )

        self.pending_action = next_action
        self.in_yellow = True
        self.reset_yellow_timer()

        return True

    def finish_yellow(self) -> bool:
        """
        Complete the previously requested transition.

        IMPORTANT:
        The method uses pending_action, NOT the latest PPO action.
        This prevents PPO action flickering during yellow from
        cancelling a transition.
        """

        if self.pending_action is None:
            # Defensive recovery.
            self.in_yellow = False
            self.reset_yellow_timer()
            return False

        next_action = int(self.pending_action)

        if next_action == (
            EnvironmentConfig.ACTION_NORTH_SOUTH
        ):
            self.set_phase(
                EnvironmentConfig.GREEN_PHASE_NS
            )
        else:
            self.set_phase(
                EnvironmentConfig.GREEN_PHASE_EW
            )

        self.current_action = next_action
        self.pending_action = None

        self.in_yellow = False
        self.reset_yellow_timer()
        self.reset_green_timer()

        self.statistics.total_switches += 1

        return True

    # =====================================================
    # ACTION
    # =====================================================

    def apply_action(self, action: int) -> bool:
        """
        Apply one RL action request.

        Returns
        -------
        True
            Controller state changed.

        False
            No controller state change occurred.

        Action semantics
        ----------------
        ACTION_NORTH_SOUTH:
            request N/S service.

        ACTION_EAST_WEST:
            request E/W service.

        A repeated action NEVER calls setPhase() again.
        """

        # Keep the controller clock aligned with simulation steps.
        self.update()

        try:
            action = int(action)
        except (TypeError, ValueError):
            self.statistics.rejected_actions += 1
            return False

        # -------------------------------------------------
        # Validate action BEFORE transition logic.
        # -------------------------------------------------

        if action not in EnvironmentConfig.ACTIONS:
            self.statistics.rejected_actions += 1
            return False

        # -------------------------------------------------
        # YELLOW IS A COMMITTED TRANSITION.
        # -------------------------------------------------
        #
        # Do not let the newest PPO prediction overwrite the
        # already-requested target while yellow is active.
        #
        # Example:
        #
        #   PPO: 1 -> start NS yellow -> pending=EW
        #   PPO: 0 -> ignored
        #   PPO: 0 -> ignored
        #   yellow complete -> EW green
        #
        # This is the key fix for directional starvation caused
        # by action changes during the yellow interval.
        # -------------------------------------------------

        if self.in_yellow:

            if self.yellow_finished():
                return self.finish_yellow()

            return False

        # -------------------------------------------------
        # SAME ACTION = HOLD CURRENT GREEN.
        # -------------------------------------------------
        #
        # DO NOT call setPhase() here.
        # SUMO must continue its current phase timer naturally.
        # -------------------------------------------------

        if action == self.current_action:
            return False

        # -------------------------------------------------
        # MINIMUM GREEN SAFETY CONSTRAINT.
        # -------------------------------------------------

        if not self.can_switch():
            self.statistics.rejected_actions += 1
            return False

        # -------------------------------------------------
        # START LATCHED YELLOW TRANSITION.
        # -------------------------------------------------

        return self.start_yellow(action)

    # =====================================================
    # INFORMATION
    # =====================================================

    def get_current_action(self) -> int:
        return self.current_action

    def get_green_timer(self) -> int:
        return self.green_timer

    def get_yellow_timer(self) -> int:
        return self.yellow_timer

    def get_pending_action(self):
        return self.pending_action

    def is_in_yellow(self) -> bool:
        return self.in_yellow

    def get_statistics(self):
        return {
            "traffic_light": self.traffic_light_id,
            "steps": self.statistics.total_steps,
            "switches": self.statistics.total_switches,
            "forced_switches": self.statistics.forced_switches,
            "rejected_actions": self.statistics.rejected_actions,
            "current_action": self.current_action,
            "current_phase": self.current_phase,
            "green_timer": self.green_timer,
            "yellow_timer": self.yellow_timer,
            "yellow_active": self.in_yellow,
            "pending_action": self.pending_action,
        }

    # =====================================================
    # RESET
    # =====================================================

    def reset(self):
        """
        Reset controller for a new episode.
        """

        self.current_action = (
            EnvironmentConfig.ACTION_NORTH_SOUTH
        )

        self.current_phase = (
            EnvironmentConfig.GREEN_PHASE_NS
        )

        self.green_timer = 0
        self.yellow_timer = 0
        self.in_yellow = False
        self.pending_action = None

        self.statistics = ControllerStatistics()

        traci.trafficlight.setPhase(
            self.traffic_light_id,
            EnvironmentConfig.GREEN_PHASE_NS
        )

    # =====================================================
    # STRING
    # =====================================================

    def __repr__(self):
        return (
            f"TrafficController("
            f"id={self.traffic_light_id}, "
            f"action={self.current_action}, "
            f"phase={self.current_phase}, "
            f"green={self.green_timer}, "
            f"yellow={self.yellow_timer}, "
            f"pending={self.pending_action})"
        )
