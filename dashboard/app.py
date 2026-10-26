from __future__ import annotations

import atexit
import json
import os
import shutil
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import traci
from flask import Flask, jsonify, render_template_string, request, send_file
from stable_baselines3 import PPO

# ============================================================
# PROJECT BOOTSTRAP
# ============================================================
def find_project_root() -> Path:
    """Find the project root without depending on a machine-specific user path."""
    here = Path(__file__).resolve()
    cwd = Path.cwd().resolve()

    candidates = [
        here.parent,
        *here.parents,
        cwd,
        *cwd.parents,
    ]

    seen: set[str] = set()
    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except OSError:
            continue

        key = os.path.normcase(str(candidate))
        if key in seen:
            continue
        seen.add(key)

        if (candidate / "rl" / "environments" / "traffic_env.py").is_file():
            return candidate

    raise RuntimeError(
        "Could not locate the traffic-marl project root. "
        "Run the dashboard from inside the project repository or place "
        "dashboard/app.py inside the repository."
    )

PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.environments.config import EnvironmentConfig
from rl.environments.traffic_env import TrafficEnvironment
from rl.safety.audit_logger import AuditLogger
from rl.safety.human_approval import HumanApprovalGate
from rl.safety.safety_validator import SafetyValidator
from rl.governance.tool_registry import ToolAllowList


# ============================================================
# CONFIGURATION
# ============================================================
APP_TITLE = "MAI-07 | Traffic Control Center"

# FINAL PROPOSED MODEL: PPO-MARL 500K (120-D graph-aware model input)
MODEL_PATH = (
    PROJECT_ROOT
    / "rl"
    / "checkpoints"
    / "ppo_marl_500k"
    / "ppo_marl_500k_final.zip"
)

RESULTS_DIR = PROJECT_ROOT / "rl" / "results" / "ppo_marl_500k_evaluation"
AUDIT_PATH = PROJECT_ROOT / "rl" / "logs" / "safety" / "flask_dashboard_audit.csv"
AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)

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

AGENTS = ["A", "B", "C", "D"]

NEIGHBOURS = {
    "A": ["B", "C"],
    "B": ["A", "D"],
    "C": ["A", "D"],
    "D": ["B", "C"],
}

GRAPH_EDGES = [
    ("A", "B"),
    ("A", "C"),
    ("B", "D"),
    ("C", "D"),
]

MAX_STEPS = int(getattr(EnvironmentConfig, "MAX_EPISODE_STEPS", 300))
REVIEW_QUEUE_THRESHOLD = float(getattr(EnvironmentConfig, "REVIEW_QUEUE_THRESHOLD", 10.0))
REVIEW_WAIT_THRESHOLD = float(getattr(EnvironmentConfig, "REVIEW_WAIT_THRESHOLD", 120.0))
ACTION_NS = int(getattr(EnvironmentConfig, "ACTION_NORTH_SOUTH", 0))
ACTION_EW = int(getattr(EnvironmentConfig, "ACTION_EAST_WEST", 1))


# ============================================================
# HELPERS
# ============================================================
def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def normalize_action(action: Any) -> np.ndarray:
    """Normalize PPO output shape without hiding invalid values from safety validation."""
    values = np.asarray(action).reshape(-1)
    if values.size != len(AGENTS):
        raise ValueError(
            f"Expected {len(AGENTS)} PPO actions, got {values.size}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("PPO action contains non-finite values.")
    return np.rint(values).astype(np.int64)


def _locate_binary(binary_name: str) -> Path | None:
    """
    Locate a SUMO executable without assuming a specific user's machine.

    Search order:
      1. project-local SUMO locations
      2. SUMO_HOME
      3. operating-system PATH
    """
    names = [binary_name]
    if os.name == "nt" and not binary_name.lower().endswith(".exe"):
        names.insert(0, f"{binary_name}.exe")

    candidates: list[Path] = []

    for name in names:
        candidates.extend(
            [
                PROJECT_ROOT / "envir" / "Scripts" / name,
                PROJECT_ROOT / "envir" / "bin" / name,
                PROJECT_ROOT / "sumo" / "bin" / name,
                PROJECT_ROOT / "tools" / "sumo" / "bin" / name,
            ]
        )

        sumo_home = os.environ.get("SUMO_HOME")
        if sumo_home:
            candidates.append(
                Path(sumo_home).expanduser() / "bin" / name
            )

    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate.resolve()
        except OSError:
            continue

    for name in names:
        found = shutil.which(name)
        if found:
            return Path(found).resolve()

    return None


def find_sumo() -> Path:
    """Prefer SUMO-GUI for the live demo, otherwise use headless SUMO."""
    gui = _locate_binary("sumo-gui")
    if gui is not None:
        return gui

    headless = _locate_binary("sumo")
    if headless is not None:
        return headless

    return Path("sumo-gui.exe" if os.name == "nt" else "sumo-gui")


def find_sumo_gui() -> Path | None:
    """Return SUMO-GUI when available, otherwise None."""
    return _locate_binary("sumo-gui")

def build_safety_states(
    observation: np.ndarray,
) -> dict[str, dict[str, Any]]:

    observation = np.asarray(
        observation,
        dtype=np.float32,
    ).reshape(-1)

    if observation.shape != (120,):
        raise ValueError(
            f"Expected 120-D observation, "
            f"received {observation.shape}."
        )

    states = {}

    for index, agent in enumerate(AGENTS):

        start = index * 30

        local = observation[
            start:start + 15
        ]

        states[agent] = {
            "current_action": int(
                np.clip(
                    round(float(local[12])),
                    0,
                    1,
                )
            ),

            "current_phase": int(
                max(
                    0,
                    round(float(local[4])),
                )
            ),

            "green_timer": safe_float(
                local[10]
            ),

            "in_yellow": bool(
                local[13] > 0.5
            ),
        }

    return states

def process_approval(
    gate: HumanApprovalGate,
    decision: Any,
    human_approved: bool,
):
    try:
        return gate.process(
            decision,
            human_approved=human_approved,
        )
    except TypeError:
        return gate.process(
            intersection=decision.intersection,
            proposed_action=int(decision.proposed_action),
            safety_approved_action=int(decision.approved_action),
            human_approved=human_approved,
        )


def write_audit(logger: AuditLogger, approval: Any) -> None:
    try:
        logger.log(approval)
        return
    except TypeError:
        pass

    if hasattr(approval, "to_dict"):
        logger.log(approval.to_dict())
    else:
        logger.log(
            {
                "intersection": getattr(approval, "intersection", ""),
                "proposed_action": int(
                    getattr(approval, "proposed_action", 0)
                ),
                "safety_approved_action": int(
                    getattr(approval, "safety_approved_action", 0)
                ),
                "final_action": int(getattr(approval, "final_action", 0)),
                "human_approval_required": bool(
                    getattr(approval, "human_approval_required", False)
                ),
                "human_approved": bool(
                    getattr(approval, "human_approved", True)
                ),
                "status": str(getattr(approval, "status", "APPROVED")),
                "reason": str(getattr(approval, "reason", "")),
            }
        )


def audit_record_stats() -> dict[str, Any]:
    """Read the real number of persisted audit records from the audit CSV files."""
    audit_files = [
        AUDIT_PATH,
        AUDIT_PATH.parent / "flask_episode_approval_audit.csv",
    ]

    total = 0
    files: list[dict[str, Any]] = []

    for path in audit_files:
        count = 0
        if path.exists():
            try:
                # CSV headers are not audit records.
                with path.open("r", encoding="utf-8", newline="") as handle:
                    row_count = sum(1 for _ in handle)
                count = max(0, row_count - 1)
            except (OSError, UnicodeError):
                try:
                    count = int(len(pd.read_csv(path, low_memory=False)))
                except Exception:
                    count = 0

        total += count
        files.append({
            "name": path.name,
            "records": count,
            "exists": path.exists(),
        })

    return {
        "enabled": True,
        "record_count": total,
        "files": files,
    }


# ============================================================
# LIVE PPO-MARL ENGINE
# ============================================================
class LiveTrafficEngine:
    """
    One authoritative SUMO/TraCI session.

    LIVE CONTROL:
        TrafficEnvironment
            -> 120-D environment observation
            -> PPO-MARL 500K
            -> 4 actions
            -> SafetyValidator
            -> HumanApprovalGate
            -> TrafficEnvironment.step()
            -> SUMO

    The Flask browser is only the operator/evaluation console.
    """

    def __init__(self):
        self.lock = threading.RLock()
        self.tools = ToolAllowList()

        self.env = None
        self.model = None

        self.validator = SafetyValidator(
            min_green_steps=10,
            yellow_protection=True,
        )
        self.gate = HumanApprovalGate(mode="automatic")
        self.audit = AuditLogger(str(AUDIT_PATH))

        self.scenario = "baseline"
        self.governance = "automatic"

        self.observation = None
        self.running = False
        self.finished = False
        self.pending = None
        self.pending_episode = None
        self.episode_authorization = {
            "status": "NOT_AUTHORIZED",
            "episode": 0,
            "scenario": "baseline",
            "governance": "automatic",
            "message": "Prepare a new episode. Human approval is required before SUMO starts.",
        }
        self.last_error = ""
        self.episode_number = 0

        self.step = 0
        self.total_reward = 0.0
        self.total_arrivals = 0
        self.safety_overrides = 0
        self.safety_violations = 0

        # Distinct safety-intervention event tracking.
        self.active_safety_blocks = {}

        # Short grace period after human review resolution.
        self.risk_review_cooldown_until = 0

        self.decisions = {}
        self.final_actions = {}

        self.history = deque(maxlen=240)

        self.lane_shapes = {}
        self.tls_positions = {}

        self.last_step_at = 0.0
        self.loop_interval = 0.8

    # --------------------------------------------------------
    # lifecycle
    # --------------------------------------------------------
    @property
    def connected(self) -> bool:
        try:
            return bool(self.env is not None and traci.isLoaded())
        except Exception:
            try:
                return bool(self.env is not None and traci.isConnected())
            except Exception:
                return False

    def _configure_sumo(self):
        # The browser and native SUMO-GUI mirror the same TraCI session.
        # If SUMO-GUI is not installed, ordinary SUMO is used.
        sumo = find_sumo()

        if not sumo.exists() and shutil.which(str(sumo)) is None:
            raise FileNotFoundError(
                "SUMO executable was not found. Install SUMO and either "
                "set SUMO_HOME or add SUMO to PATH."
            )

        EnvironmentConfig.SUMO_BINARY = str(sumo)
        return sumo

    def _load_model(self):
        if self.model is not None:
            return

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "PPO-MARL 500K model not found:\n"
                f"{MODEL_PATH}"
            )

        self.model = PPO.load(str(MODEL_PATH), device="cpu")

        shape = tuple(self.model.observation_space.shape or ())
        if shape != (120,):
            raise ValueError(
                f"Wrong PPO-MARL model. Expected (120,), got {shape}."
            )

        nvec = tuple(self.model.action_space.nvec)
        if nvec != (2, 2, 2, 2):
            raise ValueError(
                "Wrong PPO-MARL action space. "
                f"Expected (2,2,2,2), got {nvec}."
            )

    def start(self, scenario: str, governance: str):
        """Prepare a NEW episode. Episode authorization is always required.

        No SUMO process is started until the operator approves the episode.
        This is the human-governed episode-level control gate.
        """
        with self.lock:
            self.stop(clear_error=False)

            if scenario not in SCENARIOS:
                raise ValueError(f"Unsupported scenario: {scenario}")

            if governance not in {
                "automatic",
                "approval_required",
            }:
                raise ValueError(f"Unsupported governance mode: {governance}")

            self.scenario = scenario
            self.governance = governance
            self.last_error = ""
            self.finished = False
            self.running = False
            self.pending = None
            self.pending_episode = {
                "scenario": scenario,
                "governance": governance,
                "reason": (
                    "Every simulation episode requires an operator "
                    "authorization before PPO-MARL can control SUMO."
                ),
                "status": "WAITING_FOR_HUMAN_APPROVAL",
            }
            self.episode_number += 1
            self.episode_authorization = {
                "status": "PENDING_HUMAN_APPROVAL",
                "episode": self.episode_number,
                "scenario": scenario,
                "governance": governance,
                "message": "SUMO is not started until the operator authorizes this episode.",
            }

            self.step = 0
            self.total_reward = 0.0
            self.total_arrivals = 0
            self.safety_overrides = 0
            self.safety_violations = 0
            self.active_safety_blocks = {}
            self.risk_review_cooldown_until = 0
            self.decisions = {}
            self.final_actions = {}
            self.history.clear()
            self.last_step_at = 0.0

    def approve_episode(self, approved: bool):
        """Authorize or reject the complete simulation episode."""
        with self.lock:
            if not self.pending_episode:
                raise RuntimeError("No episode is waiting for approval.")

            if not approved:
                self.pending_episode["status"] = "REJECTED"
                self.running = False
                self.finished = True
                self.episode_authorization = {
                    "status": "REJECTED",
                    "episode": self.episode_number,
                    "scenario": self.scenario,
                    "governance": self.governance,
                    "message": "Operator rejected this episode. Prepare a new episode to try again.",
                }
                self._write_episode_audit(False)
                self.pending_episode = None
                return

            scenario = self.scenario
            governance = self.governance
            self.pending_episode = None

            self.episode_authorization = {
                "status": "AUTHORIZED",
                "episode": self.episode_number,
                "scenario": scenario,
                "governance": governance,
                "message": "Operator authorized this complete episode. PPO-MARL may now control SUMO through the safety layer.",
            }

            try:
                self._configure_sumo()
                self._load_model()
                self.gate = HumanApprovalGate(mode=governance)
                self.env = TrafficEnvironment(
                    scenario_name=scenario,
                    render_mode=None,
                )

                seed = 1000 + (self.episode_number * 17)
                self.observation, _ = self.env.reset(seed=seed)
                self.observation = np.asarray(self.observation, dtype=np.float32)

                if self.observation.shape != (120,):
                    raise ValueError(
                        f"Environment returned {self.observation.shape}; "
                        "PPO-MARL uses the full 120-D graph-aware observation."
                    )
                if not self.connected:
                    raise RuntimeError("SUMO/TraCI did not remain connected after reset.")

                self._load_network_geometry()
                self._write_episode_audit(True)
                self.running = True
                self.finished = False
                self.last_step_at = 0.0
            except Exception:
                self.running = False
                self.finished = True
                raise

    def _write_episode_audit(self, approved: bool):
        import csv
        path = AUDIT_PATH.parent / "flask_episode_approval_audit.csv"
        new_file = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp", "episode", "scenario", "governance",
                    "human_approved", "reason",
                ],
            )
            if new_file:
                writer.writeheader()
            writer.writerow({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "episode": self.episode_number,
                "scenario": self.scenario,
                "governance": self.governance,
                "human_approved": approved,
                "reason": "Operator authorized episode" if approved else "Operator rejected episode",
            })

    def stop(self, clear_error=True):
        with self.lock:
            self.running = False
            self.pending = None
            self.pending_episode = None

            if self.env is not None:
                try:
                    self.env.close()
                except Exception:
                    pass

            self.env = None

            try:
                if traci.isLoaded():
                    traci.close()
                else:
                    try:
                        if traci.isConnected():
                            traci.close()
                    except Exception:
                        pass
            except Exception:
                pass

            self.finished = True

            self.step = 0
            self.total_reward = 0.0
            self.total_arrivals = 0
            self.safety_overrides = 0
            self.safety_violations = 0
            self.active_safety_blocks = {}
            self.risk_review_cooldown_until = 0

            self.decisions = {}
            self.final_actions = {}
            self.history.clear()

            if clear_error:
                self.last_error = ""

    def pause(self):
        with self.lock:
            self.running = False

    def resume(self):
        with self.lock:
            if self.connected and not self.finished and not self.pending:
                self.running = True

    # --------------------------------------------------------
    # real SUMO geometry
    # --------------------------------------------------------
    def _load_network_geometry(self):
        self.lane_shapes.clear()
        self.tls_positions.clear()

        if not self.connected:
            return

        tls_ids = list(traci.trafficlight.getIDList())

        for tls_id in tls_ids:
            try:
                x, y = traci.junction.getPosition(tls_id)
                self.tls_positions[tls_id] = (float(x), float(y))
            except Exception:
                pass

        for tls_id in tls_ids:
            if tls_id in self.tls_positions:
                continue
            try:
                controlled = traci.trafficlight.getControlledLanes(tls_id)
                points = []
                for lane_id in controlled:
                    try:
                        shape = traci.lane.getShape(lane_id)
                        points.extend(shape or [])
                    except Exception:
                        pass
                if points:
                    self.tls_positions[tls_id] = (
                        float(np.mean([p[0] for p in points])),
                        float(np.mean([p[1] for p in points])),
                    )
            except Exception:
                pass

        for lane_id in traci.lane.getIDList():
            if lane_id.startswith(":"):
                continue
            try:
                shape = traci.lane.getShape(lane_id)
                if len(shape) >= 2:
                    self.lane_shapes[lane_id] = [
                        (float(x), float(y)) for x, y in shape
                    ]
            except Exception:
                pass

    # --------------------------------------------------------
    # real signal state
    # --------------------------------------------------------
    def signal_states(self):
        result = {}

        if not self.connected:
            return result

        tls_ids = set(traci.trafficlight.getIDList())

        for agent in AGENTS:
            if agent not in tls_ids:
                continue

            try:
                phase = int(traci.trafficlight.getPhase(agent))
                duration = safe_float(
                    traci.trafficlight.getPhaseDuration(agent)
                )
                raw = str(
                    traci.trafficlight.getRedYellowGreenState(agent)
                )
                links = traci.trafficlight.getControlledLinks(agent)
            except Exception as exc:
                result[agent] = {
                    "phase": None,
                    "phase_duration": 0.0,
                    "raw_state": "?",
                    "approaches": {
                        "NORTH": "UNKNOWN",
                        "EAST": "UNKNOWN",
                        "SOUTH": "UNKNOWN",
                        "WEST": "UNKNOWN",
                    },
                    "error": str(exc),
                }
                continue

            approaches = {
                "NORTH": "RED",
                "EAST": "RED",
                "SOUTH": "RED",
                "WEST": "RED",
            }

            priority = {
                "RED": 0,
                "YELLOW": 1,
                "GREEN": 2,
                "UNKNOWN": -1,
            }

            def promote(direction, value):
                if priority.get(value, -1) > priority.get(
                    approaches[direction], 0
                ):
                    approaches[direction] = value

            tls_x, tls_y = self.tls_positions.get(agent, (0.0, 0.0))

            for index, link_group in enumerate(links):
                if index >= len(raw) or not link_group:
                    continue

                try:
                    incoming_lane = str(link_group[0][0])
                except Exception:
                    continue

                direction = None

                try:
                    shape = traci.lane.getShape(incoming_lane)
                    if shape:
                        x0, y0 = shape[0]
                        dx = float(x0) - tls_x
                        dy = float(y0) - tls_y

                        if abs(dx) >= abs(dy):
                            direction = "EAST" if dx > 0 else "WEST"
                        else:
                            direction = "NORTH" if dy > 0 else "SOUTH"
                except Exception:
                    pass

                if direction is None:
                    upper = incoming_lane.upper()
                    if upper.startswith("N"):
                        direction = "NORTH"
                    elif upper.startswith("S"):
                        direction = "SOUTH"
                    elif upper.startswith("E"):
                        direction = "EAST"
                    elif upper.startswith("W"):
                        direction = "WEST"

                if direction is None:
                    continue

                char = raw[index].lower()

                if char == "g":
                    value = "GREEN"
                elif char == "y":
                    value = "YELLOW"
                elif char == "r":
                    value = "RED"
                else:
                    value = "UNKNOWN"

                promote(direction, value)

            result[agent] = {
                "phase": phase,
                "phase_duration": duration,
                "raw_state": raw,
                "approaches": approaches,
                "error": "",
            }

        return result

    # --------------------------------------------------------
    # real traffic telemetry
    # --------------------------------------------------------
    def intersection_metrics(self):
        result = {
            agent: {
                "vehicles": 0,
                "queue": 0,
                "waiting": 0.0,
                "speed": 0.0,
            }
            for agent in AGENTS
        }

        if not self.connected:
            return result

        tls_ids = set(traci.trafficlight.getIDList())

        for agent in AGENTS:
            if agent not in tls_ids:
                continue

            try:
                lanes = list(
                    dict.fromkeys(
                        traci.trafficlight.getControlledLanes(agent)
                    )
                )
            except Exception:
                continue

            vehicle_ids = set()

            for lane_id in lanes:
                try:
                    vehicle_ids.update(
                        traci.lane.getLastStepVehicleIDs(lane_id)
                    )
                except Exception:
                    pass

            speeds = []
            waiting = 0.0
            queue = 0

            for vehicle_id in vehicle_ids:
                try:
                    speed = safe_float(
                        traci.vehicle.getSpeed(vehicle_id)
                    )
                    wait = safe_float(
                        traci.vehicle.getAccumulatedWaitingTime(vehicle_id)
                    )

                    speeds.append(speed)
                    waiting += wait

                    if speed <= 0.1:
                        queue += 1
                except Exception:
                    pass

            result[agent] = {
                "vehicles": float(len(vehicle_ids)),
                "queue": float(queue),
                "waiting": float(waiting),
                "speed": float(np.mean(speeds)) if speeds else 0.0,
            }

        return result

    def vehicle_positions(self):
        """Return real vehicle positions from the active SUMO simulation."""
        vehicles = []
        if not self.connected:
            return vehicles

        try:
            for vehicle_id in traci.vehicle.getIDList():
                try:
                    x, y = traci.vehicle.getPosition(vehicle_id)
                    angle = safe_float(traci.vehicle.getAngle(vehicle_id))
                    speed = safe_float(traci.vehicle.getSpeed(vehicle_id))
                    waiting = safe_float(traci.vehicle.getAccumulatedWaitingTime(vehicle_id))

                    # Get the real SUMO vehicle type for the live dashboard.
                    try:
                        vehicle_type = str(
                            traci.vehicle.getTypeID(vehicle_id)
                        ).lower()
                    except Exception:
                        vehicle_type = "unknown"

                    try:
                        vehicle_class = str(
                            traci.vehicle.getVehicleClass(vehicle_id)
                        ).lower()
                    except Exception:
                        vehicle_class = ""

                    if "bus" in vehicle_type or "bus" in vehicle_class:
                        display_type = "bus"
                    elif "truck" in vehicle_type or "truck" in vehicle_class:
                        display_type = "truck"
                    elif (
                        vehicle_type in {"car", "passenger"}
                        or "passenger" in vehicle_class
                    ):
                        display_type = "car"
                    else:
                        display_type = vehicle_type or "unknown"

                    vehicles.append({
                        "id": str(vehicle_id),
                        "type": display_type,
                        "type_id": vehicle_type,
                        "vehicle_class": vehicle_class,
                        "x": float(x),
                        "y": float(y),
                        "angle": float(angle),
                        "speed": float(speed),
                        "waiting": float(waiting),
                    })
                except Exception:
                    continue
        except Exception:
            pass

        return vehicles

    def live_metrics(self):
        result = {
            "vehicles": 0,
            "queue": 0,
            "waiting": 0.0,
            "speed": 0.0,
            "arrived_step": 0,
            "arrivals": self.total_arrivals,
            "sim_time": 0.0,
        }

        if not self.connected:
            return result

        try:
            vehicle_ids = traci.vehicle.getIDList()
            speeds = []
            waiting = 0.0
            queue = 0

            for vehicle_id in vehicle_ids:
                speed = safe_float(traci.vehicle.getSpeed(vehicle_id))
                speeds.append(speed)

                waiting += safe_float(
                    traci.vehicle.getAccumulatedWaitingTime(vehicle_id)
                )

                if speed <= 0.1:
                    queue += 1

            arrived = int(traci.simulation.getArrivedNumber())

            result.update(
                {
                    "vehicles": len(vehicle_ids),
                    "queue": queue,
                    "waiting": waiting,
                    "speed": float(np.mean(speeds))
                    if speeds
                    else 0.0,
                    "arrived_step": arrived,
                    "arrivals": self.total_arrivals,
                    "sim_time": safe_float(
                        traci.simulation.getTime()
                    ),
                }
            )
        except Exception:
            pass

        return result

    # --------------------------------------------------------
    # safety + PPO
    # --------------------------------------------------------
    def proposed_actions(self):
        if self.model is None:
            self._load_model()

        observation = np.asarray(self.observation, dtype=np.float32).reshape(-1)
        if observation.shape != (120,):
            raise ValueError(
                f"Expected 120-D PPO-MARL observation, got {observation.shape}."
            )
        if not np.all(np.isfinite(observation)):
            raise ValueError("PPO-MARL observation contains non-finite values.")

        proposal = self.tools.request("propose_signal_action")
        if not proposal.allowed:
            raise PermissionError(proposal.reason)

        action, _ = self.model.predict(
            observation,
            deterministic=True,
        )
        return normalize_action(action)

    def make_pending(self, proposed, decisions, reason="High-risk traffic state"):
        self.pending = {
            "proposed": proposed.tolist(),
            "decisions": decisions,
            "reason": reason,
        }

    def _record_safety_intervention(self, agent: str, decision) -> None:
        """Count a distinct safety-intervention event."""
        proposed = int(getattr(decision, "proposed_action", -1))
        approved = int(getattr(decision, "approved_action", -1))
        reason = str(getattr(decision, "reason", "")).lower()

        if "minimum-green" in reason or "minimum green" in reason:
            category = "minimum_green"
        elif "yellow" in reason:
            category = "yellow"
        elif "invalid" in reason:
            category = "invalid"
        else:
            category = "other"

        signature = (proposed, approved, category)
        if self.active_safety_blocks.get(agent) == signature:
            return

        self.safety_overrides += 1
        self.active_safety_blocks[agent] = signature

    def _clear_resolved_safety_blocks(self, decisions) -> None:
        """Clear intervention-event state after a block is resolved."""
        for agent in AGENTS:
            decision = decisions.get(agent)
            if decision is not None and not decision.overridden:
                self.active_safety_blocks.pop(agent, None)

    def approve_pending(self, approved: bool):
        """Resolve risk review, execute safety-approved actions, and resume."""
        with self.lock:
            if not self.pending:
                return

            pending = self.pending
            final_actions = []

            for agent in AGENTS:
                decision = pending["decisions"][agent]

                if decision.overridden:
                    self._record_safety_intervention(agent, decision)

                if not decision.valid_action:
                    self.safety_violations += 1

                approval = process_approval(
                    self.gate,
                    decision,
                    human_approved=approved,
                )

                # Rejection is a governance decision, not a request to
                # force action 0. The SafetyValidator remains authoritative.
                final_action = int(
                    np.clip(
                        getattr(
                            approval,
                            "final_action",
                            decision.approved_action,
                        ) if approved else decision.approved_action,
                        0,
                        1,
                    )
                )

                final_actions.append(final_action)
                self.final_actions[agent] = final_action
                audit_request = self.tools.request("write_audit_log")
                if not audit_request.allowed:
                    raise PermissionError(audit_request.reason)
                write_audit(self.audit, approval)

            # Clear the pending review before stepping.
            self.pending = None

            # Execute only through the explicitly allowed control tool.
            apply_request = self.tools.request("apply_approved_action")
            if not apply_request.allowed:
                raise PermissionError(apply_request.reason)
            self.tools.execute(
                "apply_approved_action",
                lambda actions: self._environment_step(actions),
                human_approved=True,
                safety_validated=True,
                actions=np.asarray(final_actions, dtype=np.int64),
            )

            # Automatically resume if the episode is still alive.
            if not self.finished and self.connected:
                self.risk_review_cooldown_until = self.step + 10
                self.running = True
            else:
                self.running = False

    def step_once(self):
        with self.lock:
            if not self.connected or not self.running:
                return

            if self.pending:
                return

            try:
                proposed = self.proposed_actions()

                safety_states = build_safety_states(self.observation)

                safety_request = self.tools.request("validate_safety")
                if not safety_request.allowed:
                    raise PermissionError(safety_request.reason)

                decisions = self.validator.validate_network_actions(
                    proposed,
                    safety_states,
                )

                self.decisions = decisions

                if (
                    self.governance == "approval_required"
                    and self.step >= self.risk_review_cooldown_until
                ):
                    risky = []
                    current = self.intersection_metrics()

                    for agent in AGENTS:
                        item = current.get(agent, {})
                        queue_value = safe_float(item.get("queue"))
                        waiting_value = safe_float(item.get("waiting"))

                        if (
                            queue_value >= REVIEW_QUEUE_THRESHOLD
                            or waiting_value >= REVIEW_WAIT_THRESHOLD
                        ):
                            risky.append(
                                f"{agent}: queue {queue_value:.0f}, "
                                f"waiting {waiting_value:.0f}s"
                            )

                    if risky:
                        approval_request = self.tools.request("request_human_approval")
                        if not approval_request.allowed:
                            raise PermissionError(approval_request.reason)
                        self.make_pending(
                            proposed,
                            decisions,
                            reason=(
                                "Risk-triggered review: " + "; ".join(risky)
                            ),
                        )
                        self.running = False
                        return

                final_actions = []

                for agent in AGENTS:
                    decision = decisions[agent]

                    if decision.overridden:
                        self._record_safety_intervention(
                            agent,
                            decision,
                        )

                    if not decision.valid_action:
                        self.safety_violations += 1

                    approval = process_approval(
                        self.gate,
                        decision,
                        human_approved=True,
                    )

                    final_action = int(
                        np.clip(
                            getattr(
                                approval,
                                "final_action",
                                decision.approved_action,
                            ),
                            0,
                            1,
                        )
                    )

                    final_actions.append(final_action)
                    self.final_actions[agent] = final_action

                    audit_request = self.tools.request("write_audit_log")
                    if not audit_request.allowed:
                        raise PermissionError(audit_request.reason)
                    write_audit(self.audit, approval)

                self._clear_resolved_safety_blocks(decisions)

                apply_request = self.tools.request("apply_approved_action")
                if not apply_request.allowed:
                    raise PermissionError(apply_request.reason)
                self.tools.execute(
                    "apply_approved_action",
                    lambda actions: self._environment_step(actions),
                    human_approved=True,
                    safety_validated=True,
                    actions=np.asarray(final_actions, dtype=np.int64),
                )

            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.running = False

    def _environment_step(self, final_actions):
        if self.step >= MAX_STEPS or self.finished:
            self.running = False
            self.finished = True
            return

        (
            next_observation,
            reward,
            terminated,
            truncated,
            info,
        ) = self.env.step(final_actions)

        self.observation = np.asarray(
            next_observation,
            dtype=np.float32,
        )

        self.total_reward += safe_float(reward)
        self.step += 1

        metrics = self.live_metrics()
        self.total_arrivals += int(metrics.get("arrived_step", 0))
        metrics["arrivals"] = self.total_arrivals

        self.history.append(
            {
                "step": self.step,
                "queue": metrics["queue"],
                "waiting": metrics["waiting"],
                "speed": metrics["speed"],
                "arrivals": metrics["arrivals"],
                "reward": self.total_reward,
            }
        )

        if terminated or truncated or self.step >= MAX_STEPS:
            self.running = False
            self.finished = True

    # --------------------------------------------------------
    # dashboard state
    # --------------------------------------------------------
    def network_health(self, metrics):
        queue = max(0.0, metrics.get("queue", 0.0))
        waiting = max(0.0, metrics.get("waiting", 0.0))
        speed = max(0.0, metrics.get("speed", 0.0))

        speed_component = min(1.0, speed / 10.0)
        queue_component = 1.0 / (1.0 + queue / 10.0)
        waiting_component = 1.0 / (1.0 + waiting / 100.0)
        safety_component = 1.0 if self.safety_violations == 0 else 0.0

        score = 100.0 * (
            0.40 * speed_component
            + 0.25 * queue_component
            + 0.20 * waiting_component
            + 0.15 * safety_component
        )

        return float(np.clip(score, 0.0, 100.0))

    def action_state(self):
        state = {}

        for agent in AGENTS:
            decision = self.decisions.get(agent)

            if decision is None:
                state[agent] = {
                    "proposed": None,
                    "safety": None,
                    "final": self.final_actions.get(agent),
                    "override": False,
                    "status": "WAITING",
                    "reason": "",
                }
            else:
                state[agent] = {
                    "proposed": int(decision.proposed_action),
                    "safety": int(decision.approved_action),
                    "final": self.final_actions.get(agent),
                    "override": bool(decision.overridden),
                    "status": str(
                        getattr(
                            decision,
                            "safety_status",
                            "VALID",
                        )
                    ),
                    "reason": str(
                        getattr(decision, "reason", "")
                    ),
                }

        return state

    def command_verification(self):
        result = {}

        if not self.connected:
            return result

        signals = self.signal_states()

        for agent in AGENTS:
            action = self.final_actions.get(agent)
            signal = signals.get(agent, {})
            approaches = signal.get("approaches", {})

            if action is None:
                result[agent] = {
                    "expected": "—",
                    "actual": "—",
                    "status": "WAITING",
                }
                continue

            expected = (
                "NORTH / SOUTH"
                if int(action) == ACTION_NS
                else "EAST / WEST"
            )

            expected_dirs = (
                ["NORTH", "SOUTH"]
                if int(action) == ACTION_NS
                else ["EAST", "WEST"]
            )

            expected_values = [approaches.get(d, "UNKNOWN") for d in expected_dirs]
            opposite_dirs = [
                d for d in ["NORTH", "EAST", "SOUTH", "WEST"]
                if d not in expected_dirs
            ]
            opposite_values = [approaches.get(d, "UNKNOWN") for d in opposite_dirs]

            reached = (
                all(value == "GREEN" for value in expected_values)
                and all(value == "RED" for value in opposite_values)
            )
            transition = (
                all(value in {"GREEN", "YELLOW"} for value in expected_values)
                and all(value == "RED" for value in opposite_values)
                and any(value == "YELLOW" for value in expected_values)
            )

            result[agent] = {
                "expected": expected,
                "actual": ", ".join(
                    f"{d}: {approaches.get(d, 'UNKNOWN')}"
                    for d in [
                        "NORTH",
                        "EAST",
                        "SOUTH",
                        "WEST",
                    ]
                ),
                "status": (
                    "COMMAND REACHED SUMO"
                    if reached
                    else "COMMAND IN TRANSITION"
                    if transition
                    else "COMMAND NOT CONFIRMED"
                ),
                "phase": signal.get("phase"),
                "raw_state": signal.get("raw_state", "?"),
            }

        return result

    def snapshot(self):
        with self.lock:
            metrics = self.live_metrics()
            metrics["arrivals"] = self.total_arrivals

            intersections = self.intersection_metrics()
            signals = self.signal_states()

            try:
                process = psutil.Process()
                memory_mb = (
                    process.memory_info().rss
                    / (1024 * 1024)
                )
                cpu_percent = process.cpu_percent(interval=None)
            except Exception:
                memory_mb = 0.0
                cpu_percent = 0.0

            return {
                "connected": self.connected,
                "running": self.running,
                "finished": self.finished,
                "pending": bool(self.pending),
                "pending_episode": self.pending_episode,
                "episode_authorization": self.episode_authorization,
                "episode_number": self.episode_number,
                "scenario": self.scenario,
                "governance": self.governance,
                "controller": "PPO-MARL",
                "observation_dimension": 120,
                "tool_allow_list": self.tools.allowed_tools,
                "tool_count": len(self.tools.allowed_tools),
                "model": str(MODEL_PATH),
                "model_exists": MODEL_PATH.exists(),
                "step": self.step,
                "max_steps": MAX_STEPS,
                "reward": self.total_reward,
                "overrides": self.safety_overrides,
                "violations": self.safety_violations,
                "risk_review_cooldown_until": self.risk_review_cooldown_until,
                "metrics": metrics,
                "health": (
                    self.network_health(metrics)
                    if self.step > 0
                    else None
                ),
                "audit": {
                    **audit_record_stats(),
                    "path": str(AUDIT_PATH),
                    "episode_approval_path": str(
                        AUDIT_PATH.parent / "flask_episode_approval_audit.csv"
                    ),
                },
                "governance_tools": {
                    "allowed": list(self.tools.allowed_tools),
                    "count": len(self.tools.allowed_tools),
                },
                "telemetry": {
                    "cpu_percent": float(cpu_percent),
                    "memory_mb": float(memory_mb),
                },
                "intersections": intersections,
                "signals": signals,
                "actions": self.action_state(),
                "commands": self.command_verification(),
                "history": list(self.history),
                "pending_detail": self._pending_detail(),
                "error": self.last_error,
                "geometry": {
                    "lanes": self.lane_shapes,
                    "tls": self.tls_positions,
                    "edges": GRAPH_EDGES,
                    "vehicles": self.vehicle_positions(),
                },
            }

    def _pending_detail(self):
        if not self.pending:
            return None

        data = {"reason": str(self.pending.get("reason", "High-risk traffic state"))}

        for agent, decision in self.pending["decisions"].items():
            data[agent] = {
                "proposed": int(decision.proposed_action),
                "approved": int(decision.approved_action),
                "override": bool(decision.overridden),
                "valid": bool(decision.valid_action),
                "status": str(
                    getattr(
                        decision,
                        "safety_status",
                        "VALID",
                    )
                ),
                "reason": str(
                    getattr(decision, "reason", "")
                ),
            }

        return data


# ============================================================
# BACKGROUND LIVE LOOP
# ============================================================
engine = LiveTrafficEngine()

stop_worker = threading.Event()
worker_thread: threading.Thread | None = None


def live_worker():
    """
    Single background loop for automatic simulation stepping.

    All engine methods that touch SUMO/TraCI are protected by engine.lock,
    so Flask request handlers and this worker cannot access the connection
    concurrently.
    """
    while not stop_worker.is_set():
        try:
            if engine.connected and engine.running and not engine.pending:
                now = time.monotonic()

                if now - engine.last_step_at >= engine.loop_interval:
                    engine.step_once()
                    engine.last_step_at = now
        except Exception as exc:
            with engine.lock:
                engine.last_error = (
                    f"Background loop error: "
                    f"{type(exc).__name__}: {exc}"
                )
                engine.running = False

        stop_worker.wait(0.05)


def shutdown_app() -> None:
    """Stop the worker and close the active SUMO/TraCI session on exit."""
    stop_worker.set()
    try:
        engine.stop(clear_error=True)
    except Exception:
        # Shutdown must never prevent Python from exiting.
        pass


atexit.register(shutdown_app)

worker_thread = threading.Thread(
    target=live_worker,
    daemon=True,
    name="traffic-live-loop",
)
worker_thread.start()

# ============================================================
# FLASK APPLICATION — THREE SEPARATE PAGES
# ============================================================
app = Flask(__name__)

@app.errorhandler(Exception)
def handle_unexpected_error(exc):
    if request.path.startswith("/api/"):
        return jsonify({
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }), 500
    raise exc


def _csv_json(filename):
    path = RESULTS_DIR / filename
    if not path.exists():
        return {"available": False, "rows": [], "error": f"Missing {filename}"}
    try:
        df = pd.read_csv(path)
        df = df.replace({np.nan: None})
        return {"available": True, "rows": df.to_dict(orient="records")}
    except Exception as exc:
        return {"available": False, "rows": [], "error": str(exc)}


@app.get("/")
def project_page():
    return render_template_string(PROJECT_HTML, page="project")


@app.get("/live")
def live_page():
    return render_template_string(LIVE_HTML, page="live")


@app.get("/evaluation")
def evaluation_page():
    return render_template_string(EVALUATION_HTML, page="evaluation")


@app.get("/api/state")
def api_state():
    return jsonify(engine.snapshot())


@app.get("/api/health")
def api_health():
    sumo = find_sumo()
    return jsonify({
        "app": APP_TITLE,
        "project_root": str(PROJECT_ROOT),
        "model": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "sumo": str(sumo),
        "sumo_exists": sumo.exists(),
        "controller": "PPO-MARL 500K",
        "environment_observation_dimension": 120,
        "model_observation_dimension": 120,
        "observation_type": "Global-120D graph-aware",
        "action_space": "MultiDiscrete([2,2,2,2])",
        "agents": AGENTS,
        "scenarios": SCENARIOS,
        "episode_approval": "MANDATORY BEFORE EVERY EPISODE",
        "audit_logging": True,
        "tool_allow_list": list(engine.tools.allowed_tools),
        "live_metric_waiting_definition": "Accumulated waiting time",
        "live_metric_throughput_definition": "Cumulative SUMO per-step arrivals",
    })


@app.post("/api/start")
def api_start():
    payload = request.get_json(silent=True) or {}
    scenario = payload.get("scenario", "baseline")
    governance = payload.get("governance", "automatic")
    try:
        engine.start(scenario, governance)
        return jsonify({
            "ok": True,
            "needs_episode_approval": True,
            "message": "Episode prepared. Human approval is required before SUMO starts.",
        })
    except Exception as exc:
        return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 500


@app.post("/api/approve-episode")
def api_approve_episode():
    payload = request.get_json(silent=True) or {}
    approved = bool(payload.get("approved", False))
    try:
        engine.approve_episode(approved)
        return jsonify({
            "ok": True,
            "approved": approved,
            "message": "Episode authorized and SUMO started." if approved else "Episode rejected.",
        })
    except Exception as exc:
        return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 500


@app.post("/api/pause")
def api_pause():
    engine.pause()
    return jsonify({"ok": True})


@app.post("/api/resume")
def api_resume():
    engine.resume()
    return jsonify({"ok": True})


@app.post("/api/step")
def api_step():
    with engine.lock:
        if engine.connected and not engine.pending and not engine.pending_episode:
            was_running = engine.running
            engine.running = True
            try:
                engine.step_once()
            finally:
                if not was_running and not engine.pending:
                    engine.running = False
    return jsonify({"ok": True})


@app.post("/api/stop")
def api_stop():
    engine.stop()
    return jsonify({"ok": True})


@app.post("/api/approve-action")
def api_approve_action():
    payload = request.get_json(silent=True) or {}
    approved = bool(payload.get("approved", False))
    if not engine.pending:
        return jsonify({"ok": False, "error": "No risk-triggered action is waiting for approval."}), 400
    try:
        engine.approve_pending(approved)
        return jsonify({"ok": True, "approved": approved})
    except Exception as exc:
        return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 500


@app.get("/api/audit")
def api_audit():
    """Return recent saved safety and episode-authorization audit records."""
    audit_files = [
        AUDIT_PATH,
        AUDIT_PATH.parent / "flask_episode_approval_audit.csv",
    ]
    files = []

    for path in audit_files:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(
                path,
                low_memory=False,
            ).replace({np.nan: None})
            files.append({
                "name": path.name,
                "rows": df.tail(100).to_dict(
                    orient="records"
                ),
            })
        except Exception as exc:
            files.append({
                "name": path.name,
                "rows": [],
                "error": str(exc),
            })

    stats = audit_record_stats()
    return jsonify({
        "available": bool(files),
        "record_count": stats["record_count"],
        "files": files,
    })


@app.get("/api/evaluation/overall")
def api_overall():
    def load_summary(folder: str, filename: str):
        path = PROJECT_ROOT / "rl" / "results" / folder / filename
        if not path.exists():
            return None
        try:
            rows = pd.read_csv(path, low_memory=False).replace({np.nan: None}).to_dict(orient="records")
            return rows[0] if rows else None
        except Exception:
            return None
    dqn = load_summary("dqn_500k", "dqn_500k_summary.csv")
    proposed = load_summary("ppo_marl_500k_evaluation", "ppo_marl_500k_summary.csv")
    if dqn is None or proposed is None:
        return jsonify({"available": False, "dqn": dqn, "ppo_marl": proposed, "error": "Required DQN 500K and/or PPO-MARL 500K summary CSV is missing."})
    return jsonify({"available": True, "dqn": dqn, "ppo_marl": proposed})

@app.get("/api/evaluation/scenarios")
def api_scenarios():
    def load_scenarios(folder: str, filename: str):
        path = PROJECT_ROOT / "rl" / "results" / folder / filename
        if not path.exists():
            return []
        try:
            return pd.read_csv(path, low_memory=False).replace({np.nan: None}).to_dict(orient="records")
        except Exception:
            return []
    dqn_rows = load_scenarios("dqn_500k", "dqn_500k_scenario_summary.csv")
    proposed_rows = load_scenarios("ppo_marl_500k_evaluation", "ppo_marl_500k_scenario_summary.csv")
    return jsonify({"available": bool(dqn_rows and proposed_rows), "dqn_rows": dqn_rows, "ppo_marl_rows": proposed_rows})

# ============================================================
# EVALUATION EVIDENCE DISCOVERY
# ============================================================
EVIDENCE_ROOTS = [
    PROJECT_ROOT / "rl" / "results",
    PROJECT_ROOT / "rl" / "logs" / "safety",
]

def _discover_evaluation_csvs() -> list[str]:
    """Discover all saved project CSV evidence under approved folders."""
    names: list[str] = []
    for root in EVIDENCE_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.csv"):
            try:
                rel = path.relative_to(PROJECT_ROOT / "rl")
                names.append(rel.as_posix())
            except ValueError:
                continue
    return sorted(set(names), key=lambda x: (x.lower().split('/')[0], x.lower()))

def _csv_path(name: str) -> Path | None:
    """Resolve only CSV files inside the approved evidence directories."""
    candidate = (PROJECT_ROOT / "rl" / name).resolve()
    allowed = any(
        candidate == root.resolve() or root.resolve() in candidate.parents
        for root in EVIDENCE_ROOTS
    )
    if not allowed or candidate.suffix.lower() != ".csv":
        return None
    if not candidate.exists() or not candidate.is_file():
        return None
    return candidate

@app.get("/api/evaluation/files")
def api_evaluation_files():
    files = []
    for name in _discover_evaluation_csvs():
        low = name.lower()
        if not ("dqn_500k" in low or "ppo_marl_500k" in low):
            continue
        path = _csv_path(name)
        row_count = 0
        size = 0
        if path is not None:
            try:
                row_count = int(pd.read_csv(path, low_memory=False).shape[0])
                size = int(path.stat().st_size)
            except Exception:
                row_count = 0
        files.append({"name": name, "available": path is not None, "rows": row_count, "size_bytes": size})
    return jsonify({"files": files})

@app.get("/api/evaluation/download")
def api_evaluation_download():
    name = request.args.get("name", "")
    path = _csv_path(name)
    if path is None:
        return jsonify({"available": False, "error": "CSV not found or outside approved evidence folders."}), 404
    return send_file(
        path,
        as_attachment=True,
        download_name=path.name,
        mimetype="text/csv",
    )

@app.get("/api/evaluation/file")
def api_evaluation_file():
    name = request.args.get("name", "")
    path = _csv_path(name)
    if path is None:
        return jsonify({"available": False, "rows": [], "error": "CSV file is outside the approved project evidence folders or does not exist."}), 400
    try:
        df = pd.read_csv(path, low_memory=False).replace({np.nan: None})
        return jsonify({"available": True, "name": name, "columns": [str(c) for c in df.columns], "rows": df.to_dict(orient="records")})
    except Exception as exc:
        return jsonify({"available": False, "rows": [], "error": str(exc)}), 500

# ============================================================
# COMMON WHITE-CONTROL-CENTER CSS
# ============================================================
COMMON_CSS = r"""<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{
 --bg:#070912;--bg2:#0b1020;--card:#101522;--card2:#151c2b;--line:#28334b;--line2:#394663;
 --text:#f5f7ff;--muted:#aeb9d0;--blue:#58adff;--cyan:#35d8ff;--purple:#9a68ff;--green:#38d49e;--amber:#ffc857;--red:#ff6f7f;
 --shadow:0 14px 40px rgba(0,0,0,.32);
}
*{box-sizing:border-box}html{background:var(--bg)}
body{margin:0;background:radial-gradient(circle at 12% 0%,rgba(108,45,170,.18),transparent 30%),radial-gradient(circle at 90% 15%,rgba(30,80,150,.10),transparent 28%),var(--bg);color:var(--text);font-family:Inter,Segoe UI,Arial,sans-serif;font-size:14px;line-height:1.6}
.topbar{background:rgba(7,9,18,.96);backdrop-filter:blur(14px);border-bottom:1px solid #222a3f;position:sticky;top:0;z-index:50}.nav{max-width:1500px;margin:auto;display:flex;align-items:center;gap:8px;padding:12px 22px}
.brand{font-weight:900;color:#fff;margin-right:auto;font-size:15px}.brand:before{content:'●';color:var(--purple);margin-right:7px}.nav a{color:#aeb9d2;text-decoration:none;padding:9px 13px;border-radius:9px;font-weight:800;font-size:13px}.nav a:hover,.nav a.active{background:#21152e;color:#e7c8ff;box-shadow:inset 0 0 0 1px #42265b}
.container{max-width:1500px;margin:auto;padding:28px 22px 50px}.hero{background:linear-gradient(135deg,#24103f 0%,#70209a 52%,#4927b4 100%);color:#fff;border:1px solid #6b3b93;border-radius:20px;padding:34px;box-shadow:0 18px 55px rgba(78,28,119,.28)}.hero h1{font-size:32px;line-height:1.2;margin:12px 0 10px}.hero p{margin:0;color:#eee8ff;line-height:1.7;max-width:1100px}
.section-title{margin:30px 0 13px;font-size:21px;color:#fff;font-weight:900}.grid{display:grid;gap:16px}.g2{grid-template-columns:repeat(2,minmax(0,1fr))}.g3{grid-template-columns:repeat(3,minmax(0,1fr))}.g4{grid-template-columns:repeat(4,minmax(0,1fr))}.g5{grid-template-columns:repeat(5,minmax(0,1fr))}
.card{background:linear-gradient(180deg,rgba(18,23,36,.98),rgba(13,18,29,.98));border:1px solid var(--line);border-radius:15px;padding:20px;box-shadow:var(--shadow)}.card h3{margin:0 0 8px;color:#fff}.muted{color:#b5c0d7;font-size:13px;line-height:1.65}.card p{color:#c4cbdb;line-height:1.65}.card td{color:#dce4f4}.card th{color:#b8c5df;font-weight:900}
.kpi{padding:18px;position:relative;overflow:hidden}.kpi:before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;background:linear-gradient(180deg,var(--purple),var(--cyan))}.kpi .label{font-size:11px;color:#8fa0c2;font-weight:900;text-transform:uppercase;letter-spacing:.7px}.kpi .value{font-size:29px;font-weight:950;color:#fff;margin-top:4px}.kpi .sub{font-size:11px;color:#9ba8c3;margin-top:3px}
.badge{display:inline-flex;align-items:center;gap:5px;border-radius:999px;padding:5px 9px;font-size:11px;font-weight:900;letter-spacing:.25px}.good{background:rgba(56,212,158,.12);color:#50e5ad;border:1px solid rgba(56,212,158,.35)}.warn{background:rgba(255,200,87,.12);color:#ffd36f;border:1px solid rgba(255,200,87,.35)}.bad{background:rgba(255,111,127,.12);color:#ff8b99;border:1px solid rgba(255,111,127,.35)}.info{background:rgba(88,173,255,.12);color:#69bdff;border:1px solid rgba(88,173,255,.35)}
.flow{display:flex;align-items:center;justify-content:center;gap:9px;flex-wrap:wrap;padding:14px}.flow .node{background:#151c2d;border:1px solid #34415f;border-radius:10px;padding:11px 13px;text-align:center;font-size:12px;font-weight:900;color:#eef3ff}.arrow{color:#7686a8;font-weight:900}
.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:end}.field label{display:block;font-size:11px;color:#8fa0c2;font-weight:900;margin-bottom:6px;letter-spacing:.4px}.field select,.btn{border:1px solid #35415e;background:#121827;border-radius:9px;padding:10px 13px;color:#f4f7ff;font-weight:800}.field select{min-width:230px}.btn{cursor:pointer}.btn.primary{background:linear-gradient(135deg,#6d39d8,#9345c6);color:#fff;border-color:#9b61d8}.btn.danger{background:#2a151c;color:#ff9aa5;border-color:#65303c}.btn:disabled{opacity:.5;cursor:not-allowed}
.alert{padding:14px 16px;border-radius:11px;border:1px solid}.alert.info{background:rgba(53,167,255,.10);border-color:#235274;color:#7dc9ff}.alert.warn{background:rgba(255,200,87,.10);border-color:#654f25;color:#ffd776}.alert.good{background:rgba(53,211,155,.10);border-color:#205b49;color:#55e4af}.alert.bad{background:rgba(255,107,122,.10);border-color:#67303a;color:#ff9ba7}
.hidden{display:none!important}.decision-chain{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}.decision-chain span{padding:6px 9px;border:1px solid #dce3ee;border-radius:9px;background:#f7f9fc;font-size:12px}.decision-chain strong{color:#17243a}table{width:100%;border-collapse:collapse;font-size:12.5px;background:#0f1420}th,td{border-bottom:1px solid #252e44;padding:10px 11px;text-align:left}th{color:#b8c5df;background:#171d2b;font-weight:900;position:sticky;top:0}td.num{text-align:right;font-variant-numeric:tabular-nums}.scroll{overflow:auto}
#map{height:640px}.chart{height:350px}.agent-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.agent{border:1px solid #29334a;border-radius:12px;padding:14px;background:#111725}.agent-head{display:flex;justify-content:space-between;align-items:center}.agent h3{margin:0}.stats{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:10px}.stat{background:#171d2b;border:1px solid #273149;border-radius:8px;padding:9px}.stat small{display:block;color:#8fa0c2;font-size:10px}.stat strong{font-size:13px;color:#f4f7ff}.signals{display:grid;grid-template-columns:repeat(4,1fr);gap:5px;margin-top:10px}.sig{border-radius:7px;padding:6px 3px;text-align:center;font-size:10px;font-weight:900}.sig.green{background:#123c31;color:#58e8b1}.sig.yellow{background:#433515;color:#ffd66f}.sig.red{background:#421d25;color:#ff8e9b}.sig.unknown{background:#252c3b;color:#aab6cd}
.approval{border:2px solid #7b5620;background:#1d1910;border-radius:14px;padding:18px}.approval h3{color:#ffd36f}.decision-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:9px}.decision{border:1px solid #4b3b22;background:#171a22;border-radius:10px;padding:10px}.decision strong{color:#fff}.footer{text-align:center;color:#687895;font-size:12px;padding:24px}.chart-shell{background:#0d121d;border:1px solid #28334a;border-radius:12px;padding:6px}.scenario-panel{background:linear-gradient(180deg,#101624,#0c111c);border:1px solid #313b56}.csv-card{border-top:3px solid #8b50df}.csv-name{font-weight:900;color:#d4b7ff}.eval-callout{border-left:4px solid #8b50df;background:#101625!important}.ablation-grid .stat{background:#171d2b;border:1px solid #2c3650;border-radius:10px;padding:13px}.ablation-grid .stat strong{display:block;color:#fff;font-size:21px;margin-top:3px}
@media(max-width:1000px){.g4,.g5{grid-template-columns:repeat(2,1fr)}.g3,.g2{grid-template-columns:1fr}.decision-grid{grid-template-columns:1fr 1fr}}@media(max-width:650px){.g4,.g5,.agent-grid,.decision-grid{grid-template-columns:1fr}.nav{flex-wrap:wrap}.brand{width:100%}.container{padding:16px 12px}.hero h1{font-size:24px}.field select{min-width:190px}}
.section-head{display:flex;align-items:flex-end;justify-content:space-between;gap:18px;flex-wrap:wrap}
.live-metric-field{min-width:220px}
.live-chart-meta{margin:10px 0 0;color:#8fa4c5;font-size:12px;font-weight:750}
select,.field select{color-scheme:dark}
.csv-card table{min-width:900px}
.csv-card th{position:sticky;top:0;z-index:2}
.reference-note{border:1px solid #35425e;background:#0c1423;color:#c9d5ea;border-radius:12px;padding:14px 16px}
.metric-good{color:var(--green);font-weight:850}
.metric-limit{color:var(--amber);font-weight:850}
.comparison-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px}
@media(max-width:1100px){.comparison-grid{grid-template-columns:repeat(3,1fr)}}
@media(max-width:650px){.comparison-grid{grid-template-columns:1fr 1fr}.section-head{align-items:stretch}}

</style>
<style>
.approval-episode{border:2px solid #4b75d1!important;background:#111b32!important;box-shadow:0 8px 24px rgba(55,110,230,.15)}.approval-risk{border:2px solid #9a6a17!important;background:#201a0d!important}#map{height:640px!important;background:#0b0f17;border-radius:10px}#liveChart{min-height:360px}.map-error{height:100%;min-height:300px;display:flex;align-items:center;justify-content:center;color:#ff8e9b;font-weight:800}@media(max-width:1100px){[id="map"]{height:520px!important}}
</style>
"""
NAV = r'''
<div class="topbar"><div class="nav">
  <div class="brand">🚦 MAI-07 · Traffic Control</div>
  <a href="/" class="{{'active' if page=='project' else ''}}">01 · Project & Model</a>
  <a href="/live" class="{{'active' if page=='live' else ''}}">02 · Live Control</a>
  <a href="/evaluation" class="{{'active' if page=='evaluation' else ''}}">03 · Evaluation</a>
</div></div>
'''

PROJECT_HTML = COMMON_CSS + NAV + r'''
<div class="container">
  <section class="hero">
    <div class="badge" style="background:rgba(255,255,255,.14);color:white">M.Sc. Artificial Intelligence · Semester III · MAI-07</div>
    <h1 style="margin-top:12px">Multi-Agent Reinforcement Learning for Network-Wide Adaptive Traffic Control</h1>
    <p>Research-grade simulation prototype for adaptive signal control across four connected intersections. The live controller is the completed PPO-MARL 500K proposed model, using the environment's 120-D graph-aware observation.</p>
  </section>

  <h2 class="section-title">System at a glance</h2>
  <div class="grid g4">
    <div class="card kpi"><div class="label">Live controller</div><div class="value" style="font-size:20px">PPO-MARL</div><div class="sub">500K proposed model</div></div>
    <div class="card kpi"><div class="label">Observation</div><div class="value">120-D</div><div class="sub">Global graph-aware · 15 local + 15 neighbour × 4 intersections</div></div>
    <div class="card kpi"><div class="label">Agents</div><div class="value">4</div><div class="sub">A · B · C · D</div></div>
    <div class="card kpi"><div class="label">Scenarios</div><div class="value">10</div><div class="sub">distinct demand conditions</div></div>
  </div>

  <h2 class="section-title">What makes this a MARL system?</h2>
  <div class="grid g3">
    <div class="card"><h3>🚦 Adaptive control</h3><div class="muted">The PPO-MARL policy proposes binary North/South or East/West signal actions from the current SUMO traffic state.</div></div>
    <div class="card"><h3>🕸️ Graph coordination</h3><div class="muted">Each agent receives local traffic information together with neighbour and graph-structured information representing network relationships.</div></div>
    <div class="card"><h3>🛡️ Human-governed AI</h3><div class="muted">PPO-MARL proposals pass through safety validation and governed execution. Every new episode requires human authorization before SUMO starts.</div></div>
  </div>

  <h2 class="section-title">End-to-end decision boundary</h2>
  <div class="card">
    <div class="flow">
      <div class="node">Scenario / Demand</div><div class="arrow">→</div>
      <div class="node">SUMO + TraCI</div><div class="arrow">→</div>
      <div class="node">120-D Observation</div><div class="arrow">→</div>
      <div class="node">PPO-MARL 500K</div><div class="arrow">→</div>
      <div class="node">Safety Validator</div><div class="arrow">→</div>
      <div class="node">Human Gate</div><div class="arrow">→</div>
      <div class="node">Traffic Controller</div><div class="arrow">→</div>
      <div class="node">SUMO</div>
    </div>
  </div>

  <h2 class="section-title">Research configuration</h2>
  <div class="grid g2">
    <div class="card">
      <h3>Model & action space</h3>
      <table>
        <tr><th>Algorithm</th><td>PPO · Stable-Baselines3</td></tr>
        <tr><th>Training</th><td>500,000 timesteps</td></tr>
        <tr><th>Observation</th><td>120-D graph-aware model input</td></tr>
        <tr><th>Action</th><td>MultiDiscrete([2,2,2,2])</td></tr>
        <tr><th>Signal actions</th><td>0 = North/South · 1 = East/West</td></tr>
        <tr><th>Episode horizon</th><td>300 simulation steps</td></tr>
      </table>
    </div>
    <div class="card">
      <h3>Network & governance</h3>
      <table>
        <tr><th>Agents</th><td>A, B, C, D</td></tr>
        <tr><th>Graph</th><td>A–B · A–C · B–D · C–D</td></tr>
        <tr><th>Safety</th><td>Minimum-green and yellow-transition protection</td></tr>
        <tr><th>Governance</th><td>Automatic · Risk-Triggered</td></tr>
        <tr><th>Audit</th><td>Proposal → safety decision → human decision → final action</td></tr>
        <tr><th>Simulation</th><td>SUMO / TraCI</td></tr>
      </table>
    </div>
  </div>

  <h2 class="section-title">Evaluation evidence</h2>
  <div class="grid g4">
    <div class="card kpi"><div class="label">Evaluation episodes</div><div class="value">60</div><div class="sub">2 controllers × 30 episodes</div></div>
    <div class="card kpi"><div class="label">Scenarios</div><div class="value">10</div><div class="sub">baseline through stress test</div></div>
    <div class="card kpi"><div class="label">Safety violations</div><div class="value">0*</div><div class="sub">explicit invalid-action records</div></div>
    <div class="card kpi"><div class="label">PPO inference</div><div class="value">~0.27 ms</div><div class="sub">mean measured latency</div></div>
  </div>

  <div class="card" style="margin-top:16px">
    <h3>Important research conclusion</h3>
    <div class="alert info">PPO-MARL is the proposed learned controller. The final study compares it with the DQN baseline across 10 demand scenarios and retains failure cases and metric trade-offs rather than hiding them.</div>
  </div>

  <div class="muted" style="margin-top:10px">*Safety interventions and SUMO environment warnings are tracked separately from explicit invalid-action violations.</div>

  <h2 class="section-title">Configured scenarios</h2>
  <div class="card">
    <div class="grid g5" id="scenarioList"></div>
    <div class="muted" style="margin-top:12px">Morning and evening rush scenarios change the dominant directional demand. The network itself uses SUMO edge directions; geographical names are not assumed by the PPO model.</div>
  </div>

  <div class="card" style="margin-top:16px">
    <h3>Model artifact verification</h3>
    <div class="muted" id="modelPath">Checking model path…</div>
    <div id="modelStatus" style="margin-top:10px"></div>
  </div>

  <div class="footer">MAI-07 · Page 1 of 3 · Verified project, model and research configuration</div>
</div>
<script>
async function boot(){
  try{
    const r=await fetch('/api/health'); const d=await r.json();
    document.getElementById('modelPath').textContent=d.model;
    document.getElementById('modelStatus').innerHTML=d.model_exists
      ? '<span class="badge good">✓ 500K MODEL FOUND · 120-D GRAPH-AWARE PPO-MARL MODEL VALIDATED</span>'
      : '<span class="badge bad">✕ MODEL NOT FOUND</span>';
    document.getElementById('scenarioList').innerHTML=d.scenarios.map((s,i)=>
      `<div class="stat"><small>${String(i+1).padStart(2,'0')}</small><strong>${s.replaceAll('_',' ')}</strong></div>`
    ).join('');
  }catch(e){
    document.getElementById('modelStatus').innerHTML='<span class="badge bad">✕ HEALTH CHECK UNAVAILABLE</span>';
  }
}
boot();
</script>
'''
LIVE_HTML = COMMON_CSS + NAV + r'''
<div class="container">
  <section class="hero" style="padding:22px 26px">
    <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap"><h1 style="font-size:25px;margin:0">Live PPO-MARL Control Center</h1><span id="liveStatus" class="badge" style="background:rgba(255,255,255,.14);color:white">NOT STARTED</span><span id="auditStatus" class="badge good">AUDIT ACTIVE · 0 records</span></div>
    <p style="margin-top:7px">Only the completed PPO-MARL 500K model controls the live SUMO simulation. Fixed-Time and Queue-Based controllers are never used for live control.</p>
  </section>

  <div class="card" style="margin-top:16px">
    <div class="controls">
      <div class="field"><label>TRAFFIC SCENARIO</label><select id="scenario">
        <option value="baseline">Baseline</option><option value="low_demand">Low Demand</option><option value="balanced">Balanced</option><option value="morning_rush">Morning Rush</option><option value="evening_rush">Evening Rush</option><option value="high_demand">High Demand</option><option value="north_south_heavy">North-South Heavy</option><option value="east_west_heavy">East-West Heavy</option><option value="random">Random Demand</option><option value="stress_test">Stress Test</option>
      </select></div>
      <div class="field"><label>CONTROL MODE</label><select id="governance">
        <option value="automatic">Automatic — normal PPO control</option>
        <option value="approval_required">Risk-Triggered — human reviews high-risk states</option>
      </select></div>
      <button class="btn primary" onclick="prepareEpisode()">▶ PREPARE NEW EPISODE</button><button class="btn" onclick="resumeRun()">RESUME</button><button class="btn" onclick="pauseRun()">PAUSE</button><button class="btn" onclick="stepRun()">STEP</button><button class="btn danger" onclick="stopRun()">STOP</button>
    </div>
  </div>

  <div id="episodeGate" class="approval hidden" style="margin-top:16px">
    <h3>👤 HUMAN EPISODE AUTHORIZATION REQUIRED</h3>
    <p><strong>This approval is required for every new episode.</strong> It is not asking the human to approve every PPO action. The operator is authorizing this selected scenario and granting PPO-MARL permission to control SUMO for this episode.</p>
    <div class="grid g3">
      <div class="stat"><small>Episode</small><strong id="gateEpisode">—</strong></div>
      <div class="stat"><small>Scenario</small><strong id="gateScenario">—</strong></div>
      <div class="stat"><small>Control mode</small><strong id="gateMode">—</strong></div>
      <div class="stat"><small>Decision boundary</small><strong>Human → Episode</strong></div>
    </div>
    <div style="margin-top:13px;display:flex;gap:9px"><button class="btn primary" onclick="approveEpisode(true)">✓ APPROVE & START EPISODE</button><button class="btn danger" onclick="approveEpisode(false)">✕ REJECT EPISODE</button></div>
  </div>

  <div class="grid g4" style="margin-top:16px">
    <div class="card kpi"><div class="label">Current Vehicles</div><div class="value" id="vehicles">0</div><div class="sub">in network</div></div>
    <div class="card kpi"><div class="label">Completed Trips</div><div class="value" id="arrivals">0</div><div class="sub">SUMO arrivals</div></div>
    <div class="card kpi"><div class="label">Simulation Step</div><div class="value" id="step">0/300</div><div class="sub">controller step</div></div>
    <div class="card kpi"><div class="label">Safety Overrides</div><div class="value" id="overrides">0</div><div class="sub">PPO proposals modified by safety</div></div>
  </div>

  <div class="grid g2" style="margin-top:16px;align-items:start">
    <div class="card"><div style="display:flex;justify-content:space-between;align-items:center"><h3>🗺️ Live Network</h3><span class="badge info">REAL SUMO / TRACI</span></div><div class="muted">White control surface · dark lines = real SUMO lane geometry · vehicle markers = live SUMO vehicles · signal heads = current states · dashed links = intersection relationships</div><div id="map"></div></div>
    <div>
      <div class="card"><h3>🤖 Four Controlled Intersections</h3><div class="agent-grid" id="agents"></div></div>
      <div class="card" style="margin-top:16px"><h3>🛡️ Safety & Governance</h3><div id="safetyText" class="alert info">Waiting for an episode.</div><div class="grid g2" style="margin-top:10px"><div class="stat"><small>Violations</small><strong id="violations">0</strong></div><div class="stat"><small>Mode</small><strong id="mode">—</strong></div></div></div>
    </div>
  </div>

  <div id="actionGate" class="approval hidden" style="margin-top:16px"><h3>🟠 RISK-TRIGGERED HUMAN REVIEW</h3><p><strong>Why is approval required now?</strong> The episode is already authorized. The system paused because the selected Risk-Triggered mode detected a high-risk traffic state. The human is reviewing the AI's proposed signal actions, not authorizing the whole project.</p><div id="riskReason" class="alert warn"></div><div class="decision-grid" id="pendingDecisions" style="margin-top:10px"></div><div style="margin-top:13px;display:flex;gap:9px"><button class="btn primary" onclick="approveAction(true)">✓ APPROVE SAFE ACTIONS</button><button class="btn danger" onclick="approveAction(false)">✕ REJECT / KEEP SAFETY-AUTHORIZED ACTIONS</button></div></div>

  <div class="card live-performance-card" style="margin-top:16px">
    <div class="section-head">
      <div>
        <h3 style="margin:0">📈 Live Traffic Performance</h3>
        <div class="muted">One metric at a time — no incompatible dual/triple axes.</div>
      </div>
      <div class="field live-metric-field">
        <label>LIVE METRIC</label>
        <select id="liveMetric">
          <option value="queue">Queue length</option>
          <option value="waiting">Waiting time</option>
          <option value="speed">Average speed</option>
          <option value="arrivals">Completed trips</option>
          <option value="reward">Cumulative reward</option>
        </select>
      </div>
    </div>
    <div class="live-chart-meta" id="liveChartMeta">Queue length · lower is better · vehicles</div>
    <div id="liveChart" class="chart"></div>
  </div>
  <div class="card" style="margin-top:16px"><h3>✅ Command Verification</h3><div class="muted">PPO proposal → safety-approved action → final action → current SUMO signal state.</div><div class="scroll" style="margin-top:10px"><table><thead><tr><th>Agent</th><th>PPO Proposal</th><th>Safety Action</th><th>Final Action</th><th>SUMO State</th><th>Status</th></tr></thead><tbody id="commands"></tbody></table></div></div>
  <div class="grid g2" style="margin-top:16px;align-items:start">
    <div class="card">
      <h3>🧰 Governance & Tool Registry</h3>
      <div class="muted">Only explicitly allowed control tools are exposed to the live controller.</div>
      <div id="toolRegistry" class="grid g2" style="margin-top:10px"></div>
    </div>
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <h3>🧾 Audit Trail</h3>
        <span class="badge good">LIVE AUDIT · AUTO REFRESH</span>
      </div>
      <div class="muted" style="margin-top:4px">Latest 20 audit records. dispaly<br>Audit data refreshes every 1 second.</div>
      <div class="muted" style="margin-top:8px;font-size:11.5px;color:#8fa0c2;">
        <strong>Browser real-time clock:</strong> <span id="auditClockBrowser">—</span> |
        <strong>Last server audit refresh:</strong> <span id="auditClockServer">—</span> |
        <strong>Latest audit timestamp:</strong> <span id="auditClockLatest">—</span>
      </div>
      <div id="auditTrailStatus" style="margin-top:10px"></div>
      <div class="scroll" style="margin-top:10px;max-height:240px">
        <table><thead><tr><th>Time</th><th>Subject</th><th>Status</th><th>Reason</th></tr></thead>
        <tbody id="auditRows"></tbody></table>
      </div>
    </div>
  </div>

  <div class="card" style="margin-top:16px"><h3>🔎 Recent Decision Trace</h3><div class="scroll"><table><thead><tr><th>Step</th><th>Queue</th><th>Waiting</th><th>Speed</th><th>Completed Trips</th></tr></thead><tbody id="trace"></tbody></table></div></div>
  <div class="footer">MAI-07 · Page 2 of 3 · Live control = PPO-MARL 500K only</div>
</div>
<script>
let timer=null;
function fmt(x,d=1){const n=Number(x);return Number.isFinite(n)?n.toFixed(d):'0'}
async function post(url,body={}){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});return r.json()}
async function prepareEpisode(){const scenario=document.getElementById('scenario').value, governance=document.getElementById('governance').value;const d=await post('/api/start',{scenario,governance});if(!d.ok){alert(d.error);return}document.getElementById('gateScenario').textContent=scenario.replaceAll('_',' ');document.getElementById('gateMode').textContent=document.getElementById('governance').selectedOptions[0].text;document.getElementById('episodeGate').classList.remove('hidden');refresh()}
async function approveEpisode(v){const d=await post('/api/approve-episode',{approved:v});if(!d.ok){alert(d.error);return}document.getElementById('episodeGate').classList.add('hidden');refresh()}
async function approveAction(v){
 const buttons=document.querySelectorAll('#actionGate button');
 buttons.forEach(b=>b.disabled=true);
 try{
   const d=await post('/api/approve-action',{approved:v});
   if(!d.ok){alert(d.error||'Could not resolve the human review.');return}
   await refresh();
 }finally{
   buttons.forEach(b=>b.disabled=false);
 }
}
async function resumeRun(){await post('/api/resume');refresh()} async function pauseRun(){await post('/api/pause');refresh()} async function stepRun(){await post('/api/step');refresh()} async function stopRun(){await post('/api/stop');refresh()}
function signalClass(v){return v==='GREEN'?'green':v==='YELLOW'?'yellow':v==='RED'?'red':'unknown'}
function actionText(v){return v==null?'—':Number(v)===0?'North / South':'East / West'}
function renderAgents(s){let root=document.getElementById('agents');root.innerHTML='';for(const a of ['A','B','C','D']){const d=(s.intersections||{})[a]||{},sig=(s.signals||{})[a]||{},act=(s.actions||{})[a]||{},ap=sig.approaches||{};root.innerHTML+=`<div class="agent"><div class="agent-head"><h3>Intersection ${a}</h3><span class="badge info">Phase ${sig.phase??'—'}</span></div><div class="decision-chain"><span>PPO: <strong>${actionText(act.proposed)}</strong></span><span>Safety: <strong>${actionText(act.safety)}</strong></span><span>Final: <strong>${actionText(act.final)}</strong></span></div><div class="muted" style="margin-top:5px">${act.override?'Safety validator modified the PPO proposal.':'PPO proposal passed safety validation.'}</div><div class="stats"><div class="stat"><small>Vehicles</small><strong>${fmt(d.vehicles,0)}</strong></div><div class="stat"><small>Queue</small><strong>${fmt(d.queue,0)}</strong></div><div class="stat"><small>Waiting</small><strong>${fmt(d.waiting,0)}s</strong><span class="muted">accumulated</span></div><div class="stat"><small>Speed</small><strong>${fmt(d.speed,2)} m/s</strong></div></div><div class="signals"><div class="sig ${signalClass(ap.NORTH)}">N · ${ap.NORTH||'?'}</div><div class="sig ${signalClass(ap.EAST)}">E · ${ap.EAST||'?'}</div><div class="sig ${signalClass(ap.SOUTH)}">S · ${ap.SOUTH||'?'}</div><div class="sig ${signalClass(ap.WEST)}">W · ${ap.WEST||'?'}</div></div></div>`}}
function renderMap(s){
 if(typeof Plotly==='undefined'){
   document.getElementById('map').innerHTML='<div class="map-error">Plotly is unavailable. Reload the dashboard.</div>';
   return;
 }
 const g=s.geometry||{},lanes=g.lanes||{},tls=g.tls||{},edges=g.edges||[],vehicles=g.vehicles||[];
 let traces=[];

 for(const shape of Object.values(lanes)){
   traces.push({
     x:shape.map(p=>p[0]),y:shape.map(p=>p[1]),
     mode:'lines',line:{width:18},hoverinfo:'skip',showlegend:false
   });
 }

 for(const [a,b] of edges){
   if(!tls[a]||!tls[b])continue;
   traces.push({
     x:[tls[a][0],tls[b][0]],y:[tls[a][1],tls[b][1]],
     mode:'lines',line:{width:2,dash:'dot'},
     hovertemplate:`MARL link ${a} ↔ ${b}<extra></extra>`,
     showlegend:false
   });
 }

 // Render each real SUMO vehicle class as a separate visual layer.
 const groups=[
   {type:'car',name:'Cars',color:'#2563eb',symbol:'circle',size:9},
   {type:'bus',name:'Buses',color:'#d64545',symbol:'square',size:13},
   {type:'truck',name:'Trucks',color:'#16a36a',symbol:'triangle-up',size:13},
   {type:'unknown',name:'Other',color:'#64748b',symbol:'diamond',size:10}
 ];

 for(const group of groups){
   const items=vehicles.filter(
     v=>String(v.type||'unknown').toLowerCase()===group.type
   );
   if(!items.length)continue;

   traces.push({
     x:items.map(v=>v.x),
     y:items.map(v=>v.y),
     mode:'markers',
     marker:{
       size:group.size,
       color:group.color,
       symbol:group.symbol,
       line:{width:1,color:'#ffffff'}
     },
     text:items.map(v=>v.id),
     customdata:items.map(v=>[
       v.speed,v.waiting,
       String(v.type||'unknown').toUpperCase(),
       v.type_id||''
     ]),
     hovertemplate:
       '<b>%{customdata[2]}</b> · Vehicle %{text}' +
       '<br>Speed: %{customdata[0]:.2f} m/s' +
       '<br>Waiting: %{customdata[1]:.1f}s' +
       '<br>SUMO vType: %{customdata[3]}' +
       '<extra></extra>',
     name:group.name,
     showlegend:true
   });
 }

 for(const a of ['A','B','C','D']){
   if(!tls[a])continue;
   const p=tls[a];
   traces.push({
     x:[p[0]],y:[p[1]],mode:'markers+text',text:[a],
     textposition:'middle center',marker:{size:28},
     hovertemplate:`Intersection ${a}<extra></extra>`,
     showlegend:false
   });
 }

 Plotly.react('map',traces,{
   paper_bgcolor:'#fff',plot_bgcolor:'#f7f9fc',
   font:{color:'#25364a'},margin:{l:10,r:10,t:10,b:10},
   xaxis:{visible:false,scaleanchor:'y',scaleratio:1},
   yaxis:{visible:false},
   legend:{
     orientation:'h',y:1.02,x:0,
     bgcolor:'rgba(10,14,24,.94)',font:{color:'#dce5f8'}
   }
 },{responsive:true,displaylogo:false});
}
function renderCommands(s){let root=document.getElementById('commands');root.innerHTML='';for(const a of ['A','B','C','D']){const c=(s.commands||{})[a]||{},act=(s.actions||{})[a]||{};const status=String(c.status||'WAITING');const cls=status.includes('REACHED')?'good':status.includes('TRANSITION')?'warn':'bad';root.innerHTML+=`<tr><td><strong>${a}</strong></td><td>${actionText(act.proposed)}</td><td>${actionText(act.safety)}</td><td>${actionText(act.final)}</td><td>${c.actual||'—'}</td><td><span class="badge ${cls}">${status}</span></td></tr>`}}
function renderApproval(s){
 const eg=document.getElementById('episodeGate'),ag=document.getElementById('actionGate');
 const pe=s.pending_episode,p=s.pending_detail,auth=s.episode_authorization||{};
 if(pe){
   eg.classList.remove('hidden');
   document.getElementById('gateEpisode').textContent=s.episode_number||'—';
   document.getElementById('gateScenario').textContent=String(pe.scenario||'').replaceAll('_',' ');
   document.getElementById('gateMode').textContent=String(pe.governance||'').replaceAll('_',' ');
 }else{
   eg.classList.add('hidden');
 }
 if(s.pending){
   ag.classList.remove('hidden');
   document.getElementById('riskReason').textContent=(p&&p.reason)||'High-risk traffic state detected.';
   document.getElementById('pendingDecisions').innerHTML=['A','B','C','D'].map(a=>{
     const d=(p&&p[a])||{};
     return `<div class="decision"><strong>Intersection ${a}</strong><br>AI proposal: <b>${actionText(d.proposed)}</b><br>Safety result: <b>${actionText(d.approved)}</b><br><span class="badge ${d.override?'warn':'good'}">${d.override?'SAFETY OVERRIDE':'SAFE'}</span></div>`;
   }).join('');
 }else{
   ag.classList.add('hidden');
 }
 const status=document.getElementById('authStatus');
 if(status){
   status.textContent=pe?'PENDING — HUMAN DECISION REQUIRED':
     auth.status==='AUTHORIZED'?(`AUTHORIZED · EPISODE ${auth.episode}`):
     auth.status==='REJECTED'?'REJECTED — PREPARE A NEW EPISODE':
     s.connected?(`AUTHORIZED · EPISODE ${s.episode_number}`):'NOT AUTHORIZED';
 }
}
function renderTools(s){
  const root=document.getElementById('toolRegistry');
  if(!root)return;
  const tools=(s.governance_tools&&s.governance_tools.allowed)||s.tool_allow_list||[];
  root.innerHTML=tools.map(t=>
    `<div class="stat"><small>ALLOWED TOOL</small><strong>${String(t)}</strong></div>`
  ).join('') || '<div class="muted">No tool registry data available.</div>';
}

async function refreshAudit(){
  try{
    const r=await fetch('/api/audit');
    const d=await r.json();
    const status=document.getElementById('auditTrailStatus');
    const root=document.getElementById('auditRows');
    const now = new Date().toLocaleTimeString();

    document.getElementById('auditClockBrowser').textContent = now;
    document.getElementById('auditClockServer').textContent = now;

    if(!d.available){
      if(status) status.innerHTML='<span class="badge warn">No audit records yet.</span>';
      if(root) root.innerHTML='';
      document.getElementById('auditClockLatest').textContent = '—';
      return;
    }

    const rows=[];
    for(const file of (d.files||[])){
      for(const item of (file.rows||[])){
        rows.push({
          time:item.timestamp||'—',
          subject:item.intersection||
            (item.episode!==undefined?`Episode ${item.episode}`:file.name),
          status:item.status||
            (item.human_approved===true?'APPROVED':
             item.human_approved===false?'REJECTED':'RECORDED'),
          reason:item.reason||''
        });
      }
    }

    rows.sort((a,b)=>String(b.time).localeCompare(String(a.time)));
    if(root){
      root.innerHTML=rows.slice(0,20).map(x=>
        `<tr><td>${x.time}</td><td>${x.subject}</td><td>${x.status}</td><td>${x.reason}</td></tr>`
      ).join('');
    }

    document.getElementById('auditClockLatest').textContent = rows.length > 0 ? rows[0].time : '—';
    if(status) {
      status.innerHTML= `<span class="badge good">✓ AUDIT ACTIVE · ${rows.length} recent records</span>`;
    }
  }catch(e){
    const status=document.getElementById('auditTrailStatus');
    if(status) status.innerHTML= '<span class="badge bad">Audit endpoint unavailable.</span>';
  }
}

function renderTrace(s){document.getElementById('trace').innerHTML=(s.history||[]).slice(-12).reverse().map(x=>`<tr><td>${x.step}</td><td>${fmt(x.queue,0)}</td><td>${fmt(x.waiting,0)}s</td><td>${fmt(x.speed,2)}</td><td>${fmt(x.arrivals,0)}</td></tr>`).join('')}
function renderChart(s){
  if(typeof Plotly==='undefined')return;
  const h=s.history||[];
  const metric=document.getElementById('liveMetric')?.value||'queue';
  const cfg={
    queue:{label:'Queue length',unit:'vehicles',direction:'lower is better',key:'queue',digits:0},
    waiting:{label:'Waiting time',unit:'seconds',direction:'lower is better',key:'waiting',digits:0},
    speed:{label:'Average speed',unit:'m/s',direction:'higher is better',key:'speed',digits:2},
    arrivals:{label:'Completed trips',unit:'vehicles',direction:'higher is better',key:'arrivals',digits:0},
    reward:{label:'Cumulative reward',unit:'reward',direction:'higher is better',key:'reward',digits:1}
  }[metric];
  if(!cfg)return;
  const x=h.map(v=>v.step), y=h.map(v=>Number(v[cfg.key]));
  document.getElementById('liveChartMeta').textContent=
    `${cfg.label} · ${cfg.direction} · ${cfg.unit}`;
  Plotly.react('liveChart',[{
    x:x,y:y,type:'scatter',mode:'lines+markers',
    name:cfg.label,line:{width:3},marker:{size:5},
    hovertemplate:`Step %{x}<br>${cfg.label}: %{y:.${cfg.digits}f} ${cfg.unit}<extra></extra>`
  }],{
    paper_bgcolor:'#0d1320',plot_bgcolor:'#0d1320',
    font:{color:'#eaf0ff',family:'Inter,Segoe UI,Arial,sans-serif'},
    margin:{l:70,r:30,t:45,b:58},
    xaxis:{title:'Simulation step',color:'#aeb9d0',gridcolor:'#26324a',zerolinecolor:'#3b4964'},
    yaxis:{title:cfg.label,color:'#aeb9d0',gridcolor:'#26324a',zerolinecolor:'#3b4964',rangemode:'tozero'},
    title:{text:`PPO-MARL 500K · ${cfg.label}`,x:0.02,font:{size:15,color:'#ffffff'}},
    legend:{font:{color:'#eaf0ff'},bgcolor:'rgba(0,0,0,0)'}
  },{responsive:true,displaylogo:false});
}
async function refresh(){try{const r=await fetch('/api/state');const s=await r.json();const m=s.metrics||{};document.getElementById('vehicles').textContent=fmt(m.vehicles,0);document.getElementById('arrivals').textContent=fmt(m.arrivals,0);document.getElementById('step').textContent=`${s.step}/${s.max_steps}`;document.getElementById('overrides').textContent=s.overrides;document.getElementById('violations').textContent=s.violations;renderTools(s);document.getElementById('mode').textContent=(s.governance||'—').replaceAll('_',' ');const ls=document.getElementById('liveStatus');ls.textContent=s.pending_episode?'WAITING FOR EPISODE APPROVAL':s.pending?'WAITING FOR RISK REVIEW':s.running?'LIVE · PPO-MARL':s.finished?'EPISODE FINISHED':'NOT STARTED';const auditCount=Number((s.audit||{}).record_count||0);document.getElementById('auditStatus').textContent=`AUDIT ACTIVE · ${auditCount} records`;renderAgents(s);renderMap(s);renderCommands(s);renderApproval(s);renderTrace(s);renderChart(s);document.getElementById('safetyText').className='alert '+(s.violations>0?'bad':'good');document.getElementById('safetyText').textContent=s.violations>0?'Safety violations recorded. Review the trace.':`Safety layer active. ${s.overrides} PPO proposals modified by the safety layer; 0 violations.`}catch(e){const badge=document.getElementById('auditStatus');if(badge)badge.textContent='AUDIT STATUS UNAVAILABLE';}}
document.getElementById('liveMetric')?.addEventListener('change',()=>{
  fetch('/api/state').then(r=>r.json()).then(renderChart).catch(()=>{});
});
refresh();
refreshAudit();
timer=setInterval(refresh,800);
setInterval(refreshAudit,1000);
</script>
'''

EVALUATION_HTML = COMMON_CSS + NAV + r'''
<div class="container">
<section class="hero">
  <div class="badge" style="background:rgba(255,255,255,.14);color:white">FINAL VALIDATION · BASELINE VS PROPOSED</div>
  <h1 style="margin-top:12px">Final Evaluation & Failure Analysis</h1>
  <p>Evidence-based comparison of the DQN 500K baseline and PPO-MARL 500K proposed controller. Values are read from the saved evaluation CSV files; the dashboard does not generate or modify experimental results.</p>
</section>
<h2 class="section-title">Evaluation configuration</h2>
<div class="grid g4">
  <div class="card kpi"><div class="label">Baseline</div><div class="value" style="font-size:20px">DQN 500K</div><div class="sub">Discrete(16) · Global 120-D</div></div>
  <div class="card kpi"><div class="label">Proposed</div><div class="value" style="font-size:20px">PPO-MARL 500K</div><div class="sub">MultiDiscrete · Global 120-D graph-aware</div></div>
  <div class="card kpi"><div class="label">Scenarios</div><div class="value">10</div><div class="sub">baseline → stress test</div></div>
  <div class="card kpi"><div class="label">Episodes</div><div class="value">30</div><div class="sub">3 per scenario · seeds 42–44</div></div>
</div>
<div class="card" style="margin-top:16px"><h3>Evaluation protocol</h3><table>
<tr><th>Environment</th><td>SUMO / TraCI · same TrafficEnvironment</td></tr>
<tr><th>Episode horizon</th><td>300 simulation steps</td></tr>
<tr><th>Scenarios</th><td>10 demand scenarios</td></tr>
<tr><th>Episodes / scenario</th><td>3</td></tr>
<tr><th>Seeds</th><td>42, 43, 44</td></tr>
<tr><th>Inference</th><td>Deterministic</td></tr>
<tr><th>Metrics</th><td>Reward, waiting time, queue length, average speed, throughput</td></tr>
</table></div>
<h2 class="section-title">Overall baseline vs proposed model</h2>
<div id="overallStatus" class="card"><span class="muted">Loading saved DQN and PPO-MARL summary CSVs…</span></div>
<div class="card" style="margin-top:16px"><div class="scroll"><table><thead><tr><th>Metric</th><th>DQN 500K · Baseline</th><th>PPO-MARL 500K · Proposed</th><th>Direction</th><th>Difference</th></tr></thead><tbody id="overallTable"></tbody></table></div></div>
<div class="card" style="margin-top:16px">
  <div class="controls">
    <div class="field" style="min-width:260px">
      <label>OVERALL GRAPH METRIC</label>
      <select id="overallMetricSelect">
        <option value="reward">Mean total reward ↑</option>
        <option value="waiting">Mean waiting time ↓</option>
        <option value="queue">Mean queue length ↓</option>
        <option value="speed">Mean average speed ↑</option>
        <option value="throughput">Mean throughput ↑</option>
      </select>
    </div>
  </div>
  <div class="muted" style="margin-top:8px">One metric is plotted at a time so incompatible units are not placed on a common axis.</div>
  <div id="overallChart" class="chart"></div>
</div>
<h2 class="section-title">Scenario-by-scenario robustness</h2>
<div class="card scenario-panel"><div class="controls">
<div class="field" style="min-width:260px"><label>SELECT SCENARIO</label><select id="scenarioSelect"></select></div>
<div class="field" style="min-width:260px"><label>SELECT METRIC</label><select id="metricSelect">
<option value="reward">Mean total reward ↑</option><option value="waiting">Mean waiting time ↓</option><option value="queue">Mean queue length ↓</option><option value="speed">Mean average speed ↑</option><option value="throughput">Mean throughput ↑</option>
</select></div></div>
<div id="scenarioDetail" style="margin-top:15px"></div><div class="chart-shell" style="margin-top:12px"><div id="scenarioChart" class="chart"></div></div>
<div class="alert info" style="margin-top:12px"><strong>Reading the graph:</strong> only the selected scenario and selected metric are plotted.</div></div>
<h2 class="section-title">Selected-scenario evidence table</h2>
<div class="card"><p class="muted">The table below is populated directly from the saved DQN and PPO-MARL scenario CSV evidence.</p><div class="scroll"><table><thead><tr><th>Metric</th><th>DQN 500K</th><th>PPO-MARL 500K</th><th>Direction</th></tr></thead><tbody id="selectedTable"></tbody></table></div></div>
<h2 class="section-title">Saved evaluation CSV evidence</h2>
<div class="card csv-card"><div class="controls"><div class="field" style="min-width:520px;max-width:100%"><label>SELECT DQN / PPO-MARL CSV</label><select id="csvSelect"></select></div><button class="btn primary" onclick="downloadSelectedCsv()">⬇ DOWNLOAD SELECTED CSV</button></div>
<div class="muted" style="margin-top:9px">Only DQN 500K and PPO-MARL 500K evaluation evidence is exposed on the final page.</div><div id="csvStatus" style="margin-top:10px"></div>
<div class="scroll" style="margin-top:12px;max-height:560px"><table><thead id="csvHead"></thead><tbody id="csvBody"></tbody></table></div></div>
<div class="footer">MAI-07 · Page 3 of 3 · DQN baseline vs PPO-MARL proposed model</div></div>
<script>
const SC=['baseline','low_demand','balanced','morning_rush','evening_rush','high_demand','north_south_heavy','east_west_heavy','random','stress_test'];
const MET={reward:{label:'Mean total reward',keys:['reward_mean','mean_total_reward','reward','total_reward'],higher:true},waiting:{label:'Mean waiting time',keys:['waiting_time_mean','mean_waiting_time','waiting_time','waiting'],higher:false},queue:{label:'Mean queue length',keys:['queue_length_mean','mean_queue_length','queue_length','queue'],higher:false},speed:{label:'Mean average speed',keys:['average_speed_mean','mean_average_speed','average_speed','speed'],higher:true},throughput:{label:'Mean throughput',keys:['throughput_mean','mean_throughput','throughput','arrivals'],higher:true}};
const SUMMARY_KEYS={reward:['reward_mean','mean_reward'],waiting:['waiting_time_mean','mean_waiting_time'],queue:['queue_length_mean','mean_queue_length'],speed:['average_speed_mean','mean_average_speed'],throughput:['throughput_mean','mean_throughput']};
let dqnRows=[],marlRows=[],dqnOverall=null,marlOverall=null;
function num(v){const x=Number(v);return Number.isFinite(x)?x:null} function fmt(v,d=2){const x=num(v);return x===null?'—':x.toFixed(d)} function lab(v){return String(v||'').replaceAll('_',' ').replace(/\b\w/g,c=>c.toUpperCase())} function pick(o,ks){for(const k of ks)if(o&&o[k]!==undefined&&o[k]!==null&&o[k]!=='')return o[k];return null}
function scenarioOf(r){return String(pick(r,['scenario','Scenario','scenario_name'])||'').trim().toLowerCase()} function metricVal(r,k){return num(pick(r,MET[k].keys))} function summaryVal(r,k){return num(pick(r,SUMMARY_KEYS[k]||[]))} function rowFor(rows,s){return rows.find(r=>scenarioOf(r)===s)||null} function overallPair(k){return {dqn:summaryVal(dqnOverall,k),local:summaryVal(marlOverall,k)}}
function renderOverallTable(){
  const root=document.getElementById('overallTable');
  root.innerHTML=Object.keys(MET).map(k=>{
    const p=overallPair(k);
    const d=(p.local!==null&&p.dqn!==null)?p.local-p.dqn:null;
    return `<tr><td><strong>${MET[k].label}</strong></td><td class="num">${fmt(p.dqn)}</td><td class="num">${fmt(p.local)}</td><td>${MET[k].higher?'Higher is better':'Lower is better'}</td><td class="num">${fmt(d)}</td></tr>`;
  }).join('');
}

function renderOverallChart(){
  const metric=document.getElementById('overallMetricSelect').value;
  const p=overallPair(metric);
  if(typeof Plotly==='undefined')return;

  Plotly.react('overallChart',[
    {
      x:['DQN 500K','PPO-MARL 500K'],
      y:[p.dqn,p.local],
      type:'bar',
      text:[fmt(p.dqn),fmt(p.local)],
      textposition:'auto'
    }
  ],{
    paper_bgcolor:'#fff',
    plot_bgcolor:'#f7f9fc',
    font:{color:'#243247'},
    margin:{l:75,r:25,t:55,b:80},
    xaxis:{title:'Controller'},
    yaxis:{title:MET[metric].label,rangemode:'tozero'},
    title:{text:`Overall · ${MET[metric].label}`,x:0.02,font:{size:16}}
  },{responsive:true,displaylogo:false});
}

function renderOverall(){
  renderOverallTable();
  renderOverallChart();
  document.getElementById('overallStatus').innerHTML=
    '<span class="badge good">✓ SAVED DQN + PPO-MARL SUMMARY EVIDENCE LOADED</span>';
}

function scenarioPair(s,k){const a=rowFor(dqnRows,s),b=rowFor(marlRows,s);return {dqn:a?metricVal(a,k):null,local:b?metricVal(b,k):null}}
function renderScenario(){const s=document.getElementById('scenarioSelect').value,all={};Object.keys(MET).forEach(k=>all[k]=scenarioPair(s,k));document.getElementById('scenarioDetail').innerHTML='<div class="grid g5">'+Object.keys(MET).map(k=>`<div class="stat"><small>${MET[k].label}</small><strong>${fmt(all[k].dqn)} / ${fmt(all[k].local)}</strong><span class="muted">DQN / PPO-MARL</span></div>`).join('')+'</div>';document.getElementById('selectedTable').innerHTML=Object.keys(MET).map(k=>`<tr><td><strong>${MET[k].label}</strong></td><td class="num">${fmt(all[k].dqn)}</td><td class="num">${fmt(all[k].local)}</td><td>${MET[k].higher?'Higher is better':'Lower is better'}</td></tr>`).join('');const k=document.getElementById('metricSelect').value,p=all[k];Plotly.react('scenarioChart',[{x:['DQN 500K','PPO-MARL 500K'],y:[p.dqn,p.local],type:'bar',text:[fmt(p.dqn),fmt(p.local)],textposition:'auto'}],{paper_bgcolor:'#fff',plot_bgcolor:'#f7f9fc',font:{color:'#243247'},margin:{l:75,r:25,t:55,b:80},xaxis:{title:'Controller'},yaxis:{title:MET[k].label,rangemode:'tozero'},title:{text:lab(s)+' · '+MET[k].label,x:0.02,font:{size:16}}},{responsive:true,displaylogo:false})}
async function loadScenario(){const d=await (await fetch('/api/evaluation/scenarios')).json();if(!d.available){document.getElementById('scenarioDetail').innerHTML='<div class="alert bad">Scenario evidence unavailable.</div>';return}dqnRows=d.dqn_rows||[];marlRows=d.ppo_marl_rows||[];const select=document.getElementById('scenarioSelect');select.innerHTML=SC.filter(s=>rowFor(dqnRows,s)||rowFor(marlRows,s)).map(s=>`<option value="${s}">${lab(s)}</option>`).join('');if(select.options.length){select.onchange=renderScenario;document.getElementById('metricSelect').onchange=renderScenario;renderScenario()}}
document.getElementById('overallMetricSelect')?.addEventListener('change',renderOverallChart);

async function loadOverall(){const d=await (await fetch('/api/evaluation/overall')).json();if(!d.available){document.getElementById('overallStatus').innerHTML='<span class="badge bad">Saved summary evidence unavailable.</span>';return}dqnOverall=d.dqn;marlOverall=d.ppo_marl;renderOverall()}
async function loadCsvList(){const d=await (await fetch('/api/evaluation/files')).json(),f=(d.files||[]).filter(x=>x.available),s=document.getElementById('csvSelect');s.innerHTML=f.map(x=>`<option value="${x.name}">${x.name} · ${x.rows} rows</option>`).join('');if(f.length){s.onchange=loadCsv;loadCsv()}else document.getElementById('csvStatus').innerHTML='<span class="badge warn">No DQN/PPO-MARL CSV files available.</span>'}
function downloadSelectedCsv(){const name=document.getElementById('csvSelect').value;if(name)window.open('/api/evaluation/download?name='+encodeURIComponent(name),'_blank')}
async function loadCsv(){const name=document.getElementById('csvSelect').value;if(!name)return;const d=await (await fetch('/api/evaluation/file?name='+encodeURIComponent(name))).json();if(!d.available){document.getElementById('csvStatus').innerHTML='<span class="badge bad">CSV unavailable</span>';return}const c=d.columns||[];document.getElementById('csvHead').innerHTML='<tr>'+c.map(x=>`<th>${x}</th>`).join('')+'</tr>';document.getElementById('csvBody').innerHTML=(d.rows||[]).map(r=>'<tr>'+c.map(x=>`<td>${r[x]===null||r[x]===undefined?'—':String(r[x])}</td>`).join('')+'</tr>').join('');document.getElementById('csvStatus').innerHTML=`<span class="badge good">✓ ${d.rows.length} rows loaded</span> <span class="csv-name">${name}</span>`}
loadOverall();loadScenario();loadCsvList();
</script>
'''

# render_template_string needs page variable in templates; easier inject page markers at routes.

# ============================================================
# APPLICATION ENTRY POINT
# ============================================================
if __name__ == "__main__":
    print("=" * 72, flush=True)
    print("MAI-07 | Traffic Control Center", flush=True)
    print("=" * 72, flush=True)
    print(f"Project root : {PROJECT_ROOT}", flush=True)
    print(f"Model        : {MODEL_PATH}", flush=True)
    print(f"Model found  : {MODEL_PATH.exists()}", flush=True)
    print(f"SUMO binary  : {find_sumo()}", flush=True)
    print(f"SUMO-GUI     : {find_sumo_gui() or 'not found (headless fallback)'}", flush=True)
    print("Dashboard    : http://127.0.0.1:8501/", flush=True)
    print("Live page    : http://127.0.0.1:8501/live", flush=True)
    print("Evaluation   : http://127.0.0.1:8501/evaluation", flush=True)
    print("=" * 72, flush=True)
    app.run(
        host=os.environ.get("TRAFFIC_DASHBOARD_HOST", "127.0.0.1"),
        port=int(os.environ.get("TRAFFIC_DASHBOARD_PORT", "8501")),
        debug=False,
        threaded=True,
        use_reloader=False,
    )