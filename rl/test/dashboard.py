from __future__ import annotations

import json
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
from flask import Flask, jsonify, render_template_string, request
from stable_baselines3 import PPO

# ============================================================
# PROJECT BOOTSTRAP
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.environments.config import EnvironmentConfig
from rl.environments.traffic_env import TrafficEnvironment
from rl.safety.audit_logger import AuditLogger
from rl.safety.human_approval import HumanApprovalGate
from rl.safety.safety_validator import SafetyValidator


# ============================================================
# CONFIGURATION
# ============================================================
APP_TITLE = "MAI-07 | Traffic MARL Control Center"

# IMPORTANT: this is the completed 500K native 120-D PPO-MARL model.
MODEL_PATH = (
    PROJECT_ROOT
    / "rl"
    / "checkpoints"
    / "ppo_marl_500k"
    / "ppo_marl_500k_final.zip"
)

RESULTS_DIR = PROJECT_ROOT / "rl" / "results" / "final_evaluation"
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
    values = np.asarray(action).reshape(-1)
    if values.size != 4:
        raise ValueError(f"Expected 4 PPO actions, got {values.size}.")
    return np.clip(np.rint(values), 0, 1).astype(np.int64)


def find_sumo() -> Path:
    candidates = [
        PROJECT_ROOT / "envir" / "Scripts" / "sumo.exe",
        PROJECT_ROOT / "envir" / "bin" / "sumo.exe",
        PROJECT_ROOT / "sumo" / "bin" / "sumo.exe",
        Path(r"C:\Program Files\Eclipse\Sumo\bin\sumo.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path("sumo.exe")


def build_safety_states(observation: np.ndarray) -> dict[str, dict[str, Any]]:
    observation = np.asarray(observation, dtype=np.float32).reshape(-1)
    if observation.shape != (120,):
        raise ValueError(
            f"Expected 120-D observation, received {observation.shape}."
        )

    states = {}
    for index, agent in enumerate(AGENTS):
        start = index * 30
        local = observation[start:start + 15]
        states[agent] = {
            "green_timer": safe_float(local[10]),
            "in_yellow": bool(local[13] > 0.5),
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


# ============================================================
# LIVE PPO-MARL ENGINE
# ============================================================
class LiveTrafficEngine:
    """
    One authoritative SUMO/TraCI session.

    LIVE CONTROL:
        TrafficEnvironment
            -> 120-D graph-aware observation
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
        self.last_error = ""

        self.step = 0
        self.total_reward = 0.0
        self.total_arrivals = 0
        self.safety_overrides = 0
        self.safety_violations = 0

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
        sumo = find_sumo()
        if sumo.name.lower() == "sumo-gui.exe":
            raise RuntimeError(
                "This Flask dashboard requires headless sumo.exe, "
                "not sumo-gui.exe."
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

        shape = tuple(self.model.observation_space.shape)
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
        with self.lock:
            self.stop(clear_error=False)

            if scenario not in SCENARIOS:
                raise ValueError(f"Unsupported scenario: {scenario}")

            if governance not in {
                "automatic",
                "approval_required",
                "emergency",
            }:
                raise ValueError(f"Unsupported governance mode: {governance}")

            self.scenario = scenario
            self.governance = governance

            self.last_error = ""
            self.finished = False
            self.running = True
            self.pending = None

            self.step = 0
            self.total_reward = 0.0
            self.total_arrivals = 0
            self.safety_overrides = 0
            self.safety_violations = 0

            self.decisions = {}
            self.final_actions = {}
            self.history.clear()

            self._configure_sumo()
            self._load_model()

            self.gate = HumanApprovalGate(mode=governance)

            self.env = TrafficEnvironment(
                scenario_name=scenario,
                render_mode=None,
            )

            self.observation, _ = self.env.reset(seed=42)
            self.observation = np.asarray(
                self.observation,
                dtype=np.float32,
            )

            if self.observation.shape != (120,):
                raise ValueError(
                    f"Environment returned {self.observation.shape}; "
                    "PPO-MARL requires 120-D."
                )

            if not self.connected:
                raise RuntimeError(
                    "SUMO/TraCI did not remain connected after reset."
                )

            self._load_network_geometry()

            self.running = True
            self.last_step_at = 0.0

    def stop(self, clear_error=True):
        with self.lock:
            self.running = False
            self.pending = None

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
                        traci.vehicle.getWaitingTime(vehicle_id)
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
                    traci.vehicle.getWaitingTime(vehicle_id)
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

        action, _ = self.model.predict(
            self.observation,
            deterministic=True,
        )
        return normalize_action(action)

    def make_pending(self, proposed, decisions):
        self.pending = {
            "proposed": proposed.tolist(),
            "decisions": decisions,
        }

    def approve_pending(self, approved: bool):
        with self.lock:
            if not self.pending:
                return

            final_actions = []

            for agent in AGENTS:
                decision = self.pending["decisions"][agent]

                if decision.overridden:
                    self.safety_overrides += 1

                if not decision.valid_action:
                    self.safety_violations += 1

                approval = process_approval(
                    self.gate,
                    decision,
                    human_approved=approved,
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

                write_audit(self.audit, approval)

            self.pending = None

            self._environment_step(
                np.asarray(final_actions, dtype=np.int64)
            )

    def step_once(self):
        with self.lock:
            if not self.connected or not self.running:
                return

            if self.pending:
                return

            try:
                proposed = self.proposed_actions()

                safety_states = build_safety_states(self.observation)

                decisions = self.validator.validate_network_actions(
                    proposed,
                    safety_states,
                )

                self.decisions = decisions

                if self.governance == "approval_required":
                    self.make_pending(proposed, decisions)
                    self.running = False
                    return

                final_actions = []

                for agent in AGENTS:
                    decision = decisions[agent]

                    if decision.overridden:
                        self.safety_overrides += 1

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

                    write_audit(self.audit, approval)

                self._environment_step(
                    np.asarray(final_actions, dtype=np.int64)
                )

            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.running = False

    def _environment_step(self, final_actions):
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

            actual_values = [
                approaches.get(d, "UNKNOWN")
                for d in expected_dirs
            ]

            ok = any(
                value in {"GREEN", "YELLOW"}
                for value in actual_values
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
                    if ok
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
                "scenario": self.scenario,
                "governance": self.governance,
                "controller": "PPO-MARL",
                "model": str(MODEL_PATH),
                "model_exists": MODEL_PATH.exists(),
                "step": self.step,
                "max_steps": MAX_STEPS,
                "reward": self.total_reward,
                "overrides": self.safety_overrides,
                "violations": self.safety_violations,
                "metrics": metrics,
                "health": (
                    self.network_health(metrics)
                    if self.step > 0
                    else None
                ),
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
                },
            }

    def _pending_detail(self):
        if not self.pending:
            return None

        data = {}

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


def live_worker():
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

        time.sleep(0.05)


threading.Thread(
    target=live_worker,
    daemon=True,
    name="traffic-live-loop",
).start()


# ============================================================
# SAVED EVALUATION DATA
# ============================================================
def read_csv(name: str):
    path = RESULTS_DIR / name

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        return None


def overall_summary():
    df = read_csv("final_overall_summary.csv")
    if df is None:
        return {
            "available": False,
            "rows": [],
        }

    return {
        "available": True,
        "rows": df.replace({np.nan: None}).to_dict(
            orient="records"
        ),
    }


def scenario_summary():
    df = read_csv("final_scenario_summary.csv")
    if df is None:
        return {
            "available": False,
            "rows": [],
        }

    return {
        "available": True,
        "rows": df.replace({np.nan: None}).to_dict(
            orient="records"
        ),
    }


def ablation_summary():
    df = read_csv("ppo_local_vs_ppo_marl.csv")
    if df is None:
        return {
            "available": False,
            "rows": [],
        }

    return {
        "available": True,
        "rows": df.replace({np.nan: None}).to_dict(
            orient="records"
        ),
    }


# ============================================================
# FLASK APPLICATION
# ============================================================
app = Flask(__name__)


@app.get("/")
def index():
    return render_template_string(HTML)


@app.get("/api/state")
def api_state():
    return jsonify(engine.snapshot())


@app.get("/api/evaluation/overall")
def api_overall():
    return jsonify(overall_summary())


@app.get("/api/evaluation/scenarios")
def api_scenarios():
    return jsonify(scenario_summary())


@app.get("/api/evaluation/ablation")
def api_ablation():
    return jsonify(ablation_summary())


@app.post("/api/start")
def api_start():
    payload = request.get_json(silent=True) or {}

    scenario = payload.get("scenario", "baseline")
    governance = payload.get(
        "governance",
        "automatic",
    )

    try:
        engine.start(scenario, governance)
        return jsonify(
            {
                "ok": True,
                "message": (
                    f"PPO-MARL started on "
                    f"{scenario} scenario."
                ),
            }
        )
    except Exception as exc:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                }
            ),
            500,
        )


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
    if engine.connected and not engine.pending:
        was_running = engine.running
        engine.running = True
        engine.step_once()
        if not was_running and not engine.pending:
            engine.running = False

    return jsonify({"ok": True})


@app.post("/api/stop")
def api_stop():
    engine.stop()
    return jsonify({"ok": True})


@app.post("/api/approve")
def api_approve():
    payload = request.get_json(silent=True) or {}
    approved = bool(payload.get("approved", False))

    if not engine.pending:
        return jsonify(
            {
                "ok": False,
                "error": "There is no pending human approval.",
            }
        ), 400

    try:
        engine.approve_pending(approved)
        return jsonify(
            {
                "ok": True,
                "approved": approved,
            }
        )
    except Exception as exc:
        return jsonify(
            {
                "ok": False,
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
            }
        ), 500


@app.get("/api/health")
def api_health():
    return jsonify(
        {
            "app": APP_TITLE,
            "model": str(MODEL_PATH),
            "model_exists": MODEL_PATH.exists(),
            "sumo": str(find_sumo()),
            "sumo_exists": find_sumo().exists(),
            "traci_connected": engine.connected,
            "observation_dimension": 120,
            "action_space": "[2,2,2,2]",
            "agents": AGENTS,
            "scenarios": SCENARIOS,
        }
    )


# ============================================================
# COMPLETE BROWSER UI
# ============================================================
HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MAI-07 Traffic MARL Control Center</title>

<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>

<style>
:root{
  --bg:#07111d;
  --panel:#0d1927;
  --panel2:#101f30;
  --border:#263b52;
  --text:#e8eef6;
  --muted:#8ea3b8;
  --cyan:#22d3ee;
  --green:#35d07f;
  --yellow:#f4c95d;
  --red:#ef6461;
}

*{box-sizing:border-box}

body{
  margin:0;
  background:var(--bg);
  color:var(--text);
  font-family:Inter,Segoe UI,Arial,sans-serif;
}

header{
  padding:22px 30px 18px;
  border-bottom:1px solid var(--border);
  background:linear-gradient(135deg,#07111d,#10263b);
}

h1{
  margin:0;
  font-size:28px;
}

.subtitle{
  color:var(--muted);
  margin-top:7px;
}

.wrapper{
  max-width:1700px;
  margin:auto;
  padding:18px;
}

.toolbar{
  display:flex;
  gap:10px;
  flex-wrap:wrap;
  align-items:end;
  background:var(--panel);
  border:1px solid var(--border);
  border-radius:14px;
  padding:14px;
  margin-bottom:14px;
}

label{
  font-size:12px;
  color:var(--muted);
  display:block;
  margin-bottom:5px;
}

select,button{
  border:1px solid var(--border);
  border-radius:9px;
  padding:10px 13px;
  background:#122336;
  color:var(--text);
  font-weight:650;
}

button{
  cursor:pointer;
}

button:hover{
  border-color:var(--cyan);
}

.primary{
  background:#12394a;
  border-color:#1a7387;
}

.danger{
  background:#3b1b20;
}

.kpis{
  display:grid;
  grid-template-columns:repeat(7,minmax(120px,1fr));
  gap:10px;
  margin-bottom:14px;
}

.kpi{
  background:var(--panel);
  border:1px solid var(--border);
  border-radius:13px;
  padding:13px;
}

.kpi .label{
  color:var(--muted);
  font-size:11px;
  text-transform:uppercase;
}

.kpi .value{
  font-size:24px;
  font-weight:800;
  margin-top:5px;
}

.grid{
  display:grid;
  grid-template-columns:2fr 1fr;
  gap:14px;
}

.panel{
  background:var(--panel);
  border:1px solid var(--border);
  border-radius:14px;
  padding:14px;
  margin-bottom:14px;
}

.panel h2{
  margin:0 0 10px;
  font-size:17px;
}

.map{
  min-height:610px;
}

.agents{
  display:grid;
  grid-template-columns:1fr 1fr;
  gap:10px;
}

.agent{
  background:var(--panel2);
  border:1px solid var(--border);
  border-radius:12px;
  padding:12px;
}

.agent h3{
  margin:0 0 7px;
}

.row{
  display:flex;
  justify-content:space-between;
  border-bottom:1px solid #203347;
  padding:5px 0;
  font-size:13px;
}

.badge{
  display:inline-block;
  border-radius:999px;
  padding:4px 8px;
  font-size:11px;
  font-weight:800;
}

.good{background:#123c2a;color:#70e3a0}
.warn{background:#403617;color:#f5d46e}
.bad{background:#451d22;color:#ff8b88}
.neutral{background:#203247;color:#b7c7d8}

table{
  width:100%;
  border-collapse:collapse;
  font-size:12px;
}

th,td{
  border-bottom:1px solid #203347;
  padding:8px;
  text-align:left;
}

th{color:#9eb2c7}

.chart{
  height:310px;
}

.status{
  padding:10px;
  border-radius:10px;
  margin-top:8px;
  background:#102337;
}

.pending{
  border:1px solid var(--yellow);
  background:#302b16;
  padding:12px;
  border-radius:12px;
}

.hidden{display:none}

.footer{
  color:#70869c;
  text-align:center;
  padding:18px;
  font-size:12px;
}

@media(max-width:1100px){
  .kpis{grid-template-columns:repeat(3,1fr)}
  .grid{grid-template-columns:1fr}
}

@media(max-width:650px){
  .kpis{grid-template-columns:repeat(2,1fr)}
  .agents{grid-template-columns:1fr}
}
</style>
</head>

<body>

<header>
  <h1>🚦 MAI-07 | Multi-Agent Reinforcement Learning for Network-Wide Adaptive Traffic Control</h1>
  <div class="subtitle">
    Live PPO-MARL Control Center • 500K trained model • 120-D graph-aware observation • 4 coordinated agents
  </div>
</header>

<div class="wrapper">

  <div class="toolbar">

    <div>
      <label>Traffic Scenario</label>
      <select id="scenario">
        <option value="baseline">Baseline</option>
        <option value="low_demand">Low Demand</option>
        <option value="balanced">Balanced</option>
        <option value="morning_rush">Morning Rush</option>
        <option value="evening_rush">Evening Rush</option>
        <option value="high_demand">High Demand</option>
        <option value="north_south_heavy">North-South Heavy</option>
        <option value="east_west_heavy">East-West Heavy</option>
        <option value="random">Random Demand</option>
        <option value="stress_test">Stress Test</option>
      </select>
    </div>

    <div>
      <label>Human Governance</label>
      <select id="governance">
        <option value="automatic">Automatic</option>
        <option value="approval_required">Human Approval Required</option>
        <option value="emergency">Emergency / Safety Priority</option>
      </select>
    </div>

    <button class="primary" onclick="startRun()">▶ START PPO-MARL</button>
    <button onclick="resumeRun()">▶ RESUME</button>
    <button onclick="pauseRun()">⏸ PAUSE</button>
    <button onclick="stepRun()">⏭ STEP</button>
    <button class="danger" onclick="stopRun()">⏹ STOP</button>

    <div style="margin-left:auto">
      <label>System</label>
      <span id="connection" class="badge neutral">DISCONNECTED</span>
    </div>
  </div>


  <div class="kpis">
    <div class="kpi">
      <div class="label">Health</div>
      <div class="value" id="health">—</div>
    </div>
    <div class="kpi">
      <div class="label">Queue</div>
      <div class="value" id="queue">0</div>
    </div>
    <div class="kpi">
      <div class="label">Waiting</div>
      <div class="value" id="waiting">0</div>
    </div>
    <div class="kpi">
      <div class="label">Avg Speed</div>
      <div class="value" id="speed">0</div>
    </div>
    <div class="kpi">
      <div class="label">Vehicles</div>
      <div class="value" id="vehicles">0</div>
    </div>
    <div class="kpi">
      <div class="label">Arrivals</div>
      <div class="value" id="arrivals">0</div>
    </div>
    <div class="kpi">
      <div class="label">Step</div>
      <div class="value" id="step">0/300</div>
    </div>
  </div>


  <div class="grid">

    <main>

      <div class="panel map">
        <h2>🗺️ Live Network — Real SUMO / TraCI State</h2>
        <div id="network" style="height:550px"></div>
      </div>

      <div class="panel">
        <h2>📈 Live Performance</h2>
        <div class="chart" id="liveChart"></div>
      </div>

      <div class="panel">
        <h2>🧪 Final Evaluation — Controller Comparison</h2>
        <div class="chart" id="overallChart"></div>
        <div id="overallTable"></div>
      </div>

      <div class="panel">
        <h2>🔬 PPO-Local vs PPO-MARL — Ablation</h2>
        <div id="ablationTable"></div>
      </div>

      <div class="panel">
        <h2>🌍 Scenario Performance</h2>
        <div class="chart" id="scenarioChart"></div>
      </div>

    </main>


    <aside>

      <div class="panel">
        <h2>🤖 MARL Agents</h2>
        <div class="agents" id="agents"></div>
      </div>

      <div class="panel">
        <h2>🛡️ Safety Governance</h2>
        <div class="row"><span>Overrides</span><strong id="overrides">0</strong></div>
        <div class="row"><span>Violations</span><strong id="violations">0</strong></div>
        <div class="row"><span>Mode</span><strong id="mode">Automatic</strong></div>
        <div id="safetyStatus" class="status"></div>
      </div>

      <div class="panel">
        <h2>👤 Human Approval Gate</h2>
        <div id="approval" class="status">
          No approval is currently pending.
        </div>
        <div id="approvalButtons" class="hidden" style="margin-top:10px">
          <button class="primary" onclick="approve(true)">✓ APPROVE</button>
          <button class="danger" onclick="approve(false)">✕ REJECT</button>
        </div>
      </div>

      <div class="panel">
        <h2>✅ Command Verification</h2>
        <div id="commands"></div>
      </div>

      <div class="panel">
        <h2>⚙️ System Telemetry</h2>
        <div class="row"><span>CPU</span><strong id="cpu">0%</strong></div>
        <div class="row"><span>Memory</span><strong id="memory">0 MB</strong></div>
        <div class="row"><span>SUMO</span><strong id="sumo">—</strong></div>
        <div class="row"><span>Observation</span><strong>120-D</strong></div>
        <div class="row"><span>Action</span><strong>[2,2,2,2]</strong></div>
      </div>

      <div class="panel">
        <h2>🧠 Architecture</h2>
        <pre style="white-space:pre-wrap;color:#a9bfd3;font-size:12px">Traffic Scenario
      ↓
Real SUMO / TraCI
      ↓
120-D Graph-Aware Observation
      ↓
PPO-MARL (500K)
      ↓
A / B / C / D Actions
      ↓
Safety Validator
      ↓
Human Approval Gate
      ↓
TrafficEnvironment.step()
      ↓
Real SUMO State
      ↓
Telemetry + Audit Trail</pre>
      </div>

    </aside>
  </div>


  <div class="footer">
    MAI-07 • Multi-Agent Reinforcement Learning for Network-Wide Adaptive Traffic Control
    • Live control is PPO-MARL only • Baselines are shown in saved evaluation pages
  </div>

</div>


<script>
let latestState = null;

function fmt(x, digits=2){
  const n = Number(x);
  if(!Number.isFinite(n)) return "0";
  return n.toFixed(digits);
}

async function post(url, body={}){
  const r = await fetch(url,{
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify(body)
  });
  return await r.json();
}

async function startRun(){
  const scenario = document.getElementById("scenario").value;
  const governance = document.getElementById("governance").value;

  const result = await post("/api/start",{scenario,governance});

  if(!result.ok){
    alert(result.error || "Start failed");
  }
}

async function resumeRun(){ await post("/api/resume"); }
async function pauseRun(){ await post("/api/pause"); }
async function stepRun(){ await post("/api/step"); }
async function stopRun(){ await post("/api/stop"); }

async function approve(value){
  const result = await post("/api/approve",{approved:value});
  if(!result.ok){
    alert(result.error || "Approval failed");
  }
}

async function refreshState(){
  try{
    const r = await fetch("/api/state");
    latestState = await r.json();
    renderState(latestState);
  }catch(e){
    document.getElementById("connection").textContent="SERVER ERROR";
  }
}

function renderState(s){

  const connected = !!s.connected;
  const connection = document.getElementById("connection");

  connection.textContent = connected
    ? (s.running ? "RUNNING" : "CONNECTED")
    : "DISCONNECTED";

  connection.className = "badge " +
    (connected ? "good" : "bad");

  document.getElementById("health").textContent =
    s.health == null ? "—" : fmt(s.health,1);

  const m = s.metrics || {};

  document.getElementById("queue").textContent =
    fmt(m.queue,0);

  document.getElementById("waiting").textContent =
    fmt(m.waiting,0) + "s";

  document.getElementById("speed").textContent =
    fmt(m.speed,2) + " m/s";

  document.getElementById("vehicles").textContent =
    fmt(m.vehicles,0);

  document.getElementById("arrivals").textContent =
    fmt(m.arrivals,0);

  document.getElementById("step").textContent =
    `${s.step}/${s.max_steps}`;

  document.getElementById("overrides").textContent =
    s.overrides;

  document.getElementById("violations").textContent =
    s.violations;

  document.getElementById("mode").textContent =
    String(s.governance || "").replaceAll("_"," ");

  const safety = document.getElementById("safetyStatus");

  if(s.violations === 0){
    safety.innerHTML =
      '<span class="badge good">NO SAFETY VIOLATIONS</span>';
  }else{
    safety.innerHTML =
      '<span class="badge bad">SAFETY VIOLATIONS RECORDED</span>';
  }

  document.getElementById("cpu").textContent =
    fmt(s.telemetry.cpu_percent,1) + "%";

  document.getElementById("memory").textContent =
    fmt(s.telemetry.memory_mb,0) + " MB";

  document.getElementById("sumo").textContent =
    connected ? "TraCI CONNECTED" : "Not connected";

  renderAgents(s);
  renderCommands(s);
  renderApproval(s);
  renderNetwork(s);
  renderLiveChart(s);
}

function renderAgents(s){

  const root = document.getElementById("agents");
  root.innerHTML = "";

  for(const a of ["A","B","C","D"]){

    const d = (s.intersections || {})[a] || {};
    const sig = (s.signals || {})[a] || {};
    const act = (s.actions || {})[a] || {};

    const finalAction =
      act.final == null ? "—" :
      Number(act.final) === 0 ? "N/S" : "E/W";

    const phase =
      sig.phase == null ? "—" : sig.phase;

    const approach = sig.approaches || {};

    const card = document.createElement("div");
    card.className="agent";

    card.innerHTML = `
      <h3>Intersection ${a}</h3>
      <div class="row"><span>Vehicles</span><strong>${fmt(d.vehicles,0)}</strong></div>
      <div class="row"><span>Queue</span><strong>${fmt(d.queue,0)}</strong></div>
      <div class="row"><span>Waiting</span><strong>${fmt(d.waiting,0)}s</strong></div>
      <div class="row"><span>Speed</span><strong>${fmt(d.speed,2)} m/s</strong></div>
      <div class="row"><span>SUMO Phase</span><strong>${phase}</strong></div>
      <div class="row"><span>PPO Final</span><strong>${finalAction}</strong></div>
      <div class="row"><span>N</span><strong>${approach.NORTH || "—"}</strong></div>
      <div class="row"><span>E</span><strong>${approach.EAST || "—"}</strong></div>
      <div class="row"><span>S</span><strong>${approach.SOUTH || "—"}</strong></div>
      <div class="row"><span>W</span><strong>${approach.WEST || "—"}</strong></div>
    `;

    root.appendChild(card);
  }
}

function renderCommands(s){

  const root = document.getElementById("commands");
  root.innerHTML = "";

  for(const a of ["A","B","C","D"]){

    const c = (s.commands || {})[a] || {};

    let cls="neutral";

    if(String(c.status).includes("REACHED")){
      cls="good";
    }else if(String(c.status).includes("NOT")){
      cls="bad";
    }

    root.innerHTML += `
      <div style="margin-bottom:9px">
        <strong>${a}</strong>
        <span class="badge ${cls}">${c.status || "WAITING"}</span>
        <div style="font-size:11px;color:#8ea3b8;margin-top:4px">
          Expected: ${c.expected || "—"}<br>
          Actual: ${c.actual || "—"}
        </div>
      </div>
    `;
  }
}

function renderApproval(s){

  const box = document.getElementById("approval");
  const buttons = document.getElementById("approvalButtons");

  if(!s.pending){

    box.className="status";
    box.textContent =
      "No approval is currently pending.";

    buttons.classList.add("hidden");
    return;
  }

  box.className="pending";

  let html =
    "<strong>HUMAN DECISION REQUIRED</strong><br>" +
    "SUMO is paused until the operator decides.<br><br>";

  for(const a of ["A","B","C","D"]){

    const d = (s.pending_detail || {})[a] || {};

    html += `
      <div style="margin:4px 0">
        <strong>${a}</strong>:
        proposed=${d.proposed},
        safety=${d.approved}
        ${d.override ? " • SAFETY OVERRIDE" : ""}
        <br>
        <span style="font-size:11px">${d.reason || ""}</span>
      </div>
    `;
  }

  box.innerHTML = html;
  buttons.classList.remove("hidden");
}

function renderNetwork(s){

  const geometry = s.geometry || {};
  const lanes = geometry.lanes || {};
  const tls = geometry.tls || {};
  const edges = geometry.edges || [];

  const traces=[];

  for(const [laneId,shape] of Object.entries(lanes)){

    const x=shape.map(p=>p[0]);
    const y=shape.map(p=>p[1]);

    traces.push({
      x,y,
      mode:"lines",
      line:{width:10},
      hoverinfo:"skip",
      showlegend:false
    });
  }

  for(const [a,b] of edges){

    if(!tls[a] || !tls[b]) continue;

    traces.push({
      x:[tls[a][0],tls[b][0]],
      y:[tls[a][1],tls[b][1]],
      mode:"lines",
      line:{width:2,dash:"dot"},
      hovertemplate:`MARL communication ${a} ↔ ${b}<extra></extra>`,
      showlegend:false
    });
  }

  const vx=[],vy=[],vt=[];

  if(s.connected){

    // Vehicle positions are intentionally retrieved by a small endpoint
    // below rather than invented by the browser.
    // If available, they are added by renderVehicles().
  }

  for(const [a,p] of Object.entries(tls)){

    traces.push({
      x:[p[0]],
      y:[p[1]],
      mode:"markers+text",
      text:[a],
      textposition:"middle center",
      marker:{size:30},
      hovertemplate:`Intersection ${a}<extra></extra>`,
      showlegend:false
    });
  }

  // Draw signal state markers around each real SUMO junction.
  const offsets={
    NORTH:[0,12],
    EAST:[12,0],
    SOUTH:[0,-12],
    WEST:[-12,0]
  };

  const colors={
    GREEN:"#35d07f",
    YELLOW:"#f4c95d",
    RED:"#ef6461",
    UNKNOWN:"#94a3b8"
  };

  for(const a of ["A","B","C","D"]){

    if(!tls[a]) continue;

    const sig=(s.signals||{})[a]||{};
    const approaches=sig.approaches||{};

    for(const [dir,off] of Object.entries(offsets)){

      const state=approaches[dir]||"UNKNOWN";

      traces.push({
        x:[tls[a][0]+off[0]],
        y:[tls[a][1]+off[1]],
        mode:"markers+text",
        text:[dir[0]],
        textposition:"middle center",
        marker:{
          size:13,
          color:colors[state] || colors.UNKNOWN
        },
        hovertemplate:
          `${a} ${dir}<br>${state}<extra></extra>`,
        showlegend:false
      });
    }
  }

  const layout={
    paper_bgcolor:"#07111d",
    plot_bgcolor:"#07111d",
    font:{color:"#dce8f2"},
    margin:{l:10,r:10,t:10,b:10},
    xaxis:{
      visible:false,
      scaleanchor:"y",
      scaleratio:1
    },
    yaxis:{visible:false},
    showlegend:false
  };

  Plotly.react(
    "network",
    traces,
    layout,
    {responsive:true,displaylogo:false}
  );
}

function renderLiveChart(s){

  const h=s.history||[];

  const steps=h.map(x=>x.step);

  const data=[
    {
      x:steps,
      y:h.map(x=>x.queue),
      name:"Queue",
      mode:"lines"
    },
    {
      x:steps,
      y:h.map(x=>x.waiting),
      name:"Waiting",
      mode:"lines",
      yaxis:"y2"
    },
    {
      x:steps,
      y:h.map(x=>x.speed),
      name:"Speed",
      mode:"lines",
      yaxis:"y3"
    }
  ];

  const layout={
    paper_bgcolor:"#0d1927",
    plot_bgcolor:"#0d1927",
    font:{color:"#dce8f2"},
    margin:{l:45,r:60,t:20,b:35},
    legend:{orientation:"h"},
    xaxis:{title:"Step"},
    yaxis:{title:"Queue"},
    yaxis2:{
      title:"Waiting",
      overlaying:"y",
      side:"right"
    }
  };

  Plotly.react(
    "liveChart",
    data,
    layout,
    {responsive:true,displaylogo:false}
  );
}

async function loadOverall(){

  const r=await fetch("/api/evaluation/overall");
  const data=await r.json();

  if(!data.available) return;

  const rows=data.rows;

  const names=rows.map(x=>x.controller);

  const waiting=rows.map(x=>x.mean_waiting_time);
  const queue=rows.map(x=>x.mean_queue_length);
  const speed=rows.map(x=>x.mean_average_speed);

  Plotly.react(
    "overallChart",
    [
      {x:names,y:waiting,name:"Waiting",type:"bar"},
      {x:names,y:queue,name:"Queue",type:"bar"},
      {x:names,y:speed,name:"Speed",type:"bar"}
    ],
    {
      barmode:"group",
      paper_bgcolor:"#0d1927",
      plot_bgcolor:"#0d1927",
      font:{color:"#dce8f2"},
      margin:{l:45,r:20,t:20,b:50}
    },
    {responsive:true,displaylogo:false}
  );

  let html="<table><tr>";

  const keys=[
    "controller",
    "mean_total_reward",
    "mean_waiting_time",
    "mean_queue_length",
    "mean_average_speed",
    "mean_throughput"
  ];

  for(const k of keys){
    html += `<th>${k.replaceAll("_"," ")}</th>`;
  }

  html+="</tr>";

  for(const row of rows){

    html+="<tr>";

    for(const k of keys){
      const value=row[k];
      html += `<td>${
        typeof value==="number" ? fmt(value,2) : (value ?? "—")
      }</td>`;
    }

    html+="</tr>";
  }

  html+="</table>";

  document.getElementById("overallTable").innerHTML=html;
}

async function loadAblation(){

  const r=await fetch("/api/evaluation/ablation");
  const data=await r.json();

  if(!data.available) return;

  const rows=data.rows;

  let html="<table><tr>";

  if(rows.length){

    for(const key of Object.keys(rows[0])){
      html += `<th>${key.replaceAll("_"," ")}</th>`;
    }

    html+="</tr>";

    for(const row of rows){

      html+="<tr>";

      for(const value of Object.values(row)){
        html += `<td>${
          typeof value==="number" ? fmt(value,2) : (value ?? "—")
        }</td>`;
      }

      html+="</tr>";
    }
  }

  html+="</table>";

  document.getElementById("ablationTable").innerHTML=html;
}

async function loadScenarios(){

  const r=await fetch("/api/evaluation/scenarios");
  const data=await r.json();

  if(!data.available) return;

  const rows=data.rows;

  // Prefer PPO-Local and PPO-MARL if the expected columns exist.
  const policies=[...new Set(rows.map(x=>x.policy))];

  const marl=rows.filter(x=>String(x.policy).toLowerCase().includes("marl"));
  const local=rows.filter(x=>String(x.policy).toLowerCase().includes("local"));

  const scenarios=[...new Set(rows.map(x=>x.scenario))];

  const traces=[];

  if(marl.length){
    traces.push({
      x:marl.map(x=>x.scenario),
      y:marl.map(x=>x.mean_waiting_time),
      name:"PPO-MARL",
      mode:"lines+markers"
    });
  }

  if(local.length){
    traces.push({
      x:local.map(x=>x.scenario),
      y:local.map(x=>x.mean_waiting_time),
      name:"PPO-Local",
      mode:"lines+markers"
    });
  }

  if(!traces.length){

    for(const p of policies){

      const subset=rows.filter(x=>x.policy===p);

      traces.push({
        x:subset.map(x=>x.scenario),
        y:subset.map(x=>x.mean_waiting_time),
        name:p,
        mode:"lines+markers"
      });
    }
  }

  Plotly.react(
    "scenarioChart",
    traces,
    {
      paper_bgcolor:"#0d1927",
      plot_bgcolor:"#0d1927",
      font:{color:"#dce8f2"},
      margin:{l:45,r:20,t:20,b:80},
      xaxis:{title:"Scenario"},
      yaxis:{title:"Mean waiting time"},
      legend:{orientation:"h"}
    },
    {responsive:true,displaylogo:false}
  );
}

async function boot(){

  await loadOverall();
  await loadAblation();
  await loadScenarios();

  await refreshState();

  setInterval(refreshState,800);
}

boot();
</script>

</body>
</html>
"""


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 72)
    print("MAI-07 FLASK TRAFFIC MARL CONTROL CENTER")
    print("=" * 72)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Model        : {MODEL_PATH}")
    print(f"Model exists : {MODEL_PATH.exists()}")
    print(f"SUMO         : {find_sumo()}")
    print(f"SUMO exists  : {find_sumo().exists()}")
    print("Controller   : PPO-MARL only")
    print("Observation  : 120-D")
    print("Action space : MultiDiscrete([2,2,2,2])")
    print("Scenarios    : 10")
    print("URL          : http://127.0.0.1:5000")
    print("=" * 72)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False,
    )
