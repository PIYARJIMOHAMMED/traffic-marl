"""
=========================================================
Traffic MARL Project

Module:
Queue Balance Analysis

Description:
Provides interpretable network-level queue balance analysis
for the four-intersection SUMO traffic network.

Purpose:
    - Analyse queue distribution across A, B, C and D
    - Measure whether congestion is balanced across agents
    - Identify the most congested intersection
    - Support graph/network interpretability
    - Produce evaluation-ready results
    - Work with live SUMO measurements or evaluator results

This module does NOT:
    - retrain PPO-MARL
    - modify observations
    - modify actions
    - modify safety rules
    - modify human governance
    - modify the traffic environment

Network:
    A <-> B
    A <-> C
    B <-> D
    C <-> D

Author:
    Mohammed Piyarji
=========================================================
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, Mapping, Optional

import math
import numpy as np


# =========================================================
# NETWORK CONFIGURATION
# =========================================================

DEFAULT_INTERSECTIONS = (
    "A",
    "B",
    "C",
    "D",
)

DEFAULT_NEIGHBOURS = {
    "A": ["B", "C"],
    "B": ["A", "D"],
    "C": ["A", "D"],
    "D": ["B", "C"],
}


# =========================================================
# RESULT DATA CLASS
# =========================================================

@dataclass
class QueueBalanceResult:
    """
    Complete queue-balance analysis for one network state.
    """

    intersection_queues: Dict[str, float]

    mean_queue: float
    max_queue: float
    min_queue: float

    queue_std: float
    queue_range: float

    imbalance_index: float
    max_min_ratio: float

    most_congested_intersection: str
    least_congested_intersection: str

    balanced: bool

    balance_threshold: float

    neighbour_queue_difference: Dict[str, float]

    network_neighbour_consistency: float

    def to_dict(self) -> Dict:
        """
        Convert the result into a JSON/CSV-friendly dictionary.
        """
        return asdict(self)


# =========================================================
# QUEUE BALANCE ANALYSER
# =========================================================

class QueueBalanceAnalyzer:
    """
    Analyse how evenly queues are distributed across
    the traffic-light network.

    The analyser is intentionally independent from SUMO,
    PPO and the environment so it can be reused for:

        - live dashboard analysis
        - PPO-MARL evaluation
        - DQN evaluation
        - PPO-Local evaluation
        - stress testing
        - comparison experiments
    """

    def __init__(
        self,
        intersections: Optional[Iterable[str]] = None,
        neighbours: Optional[Mapping[str, Iterable[str]]] = None,
        balance_threshold: float = 0.25,
    ):
        self.intersections = tuple(
            intersections
            if intersections is not None
            else DEFAULT_INTERSECTIONS
        )

        self.neighbours = {
            str(agent): list(values)
            for agent, values in (
                neighbours.items()
                if neighbours is not None
                else DEFAULT_NEIGHBOURS.items()
            )
        }

        self.balance_threshold = float(
            balance_threshold
        )

        if self.balance_threshold < 0:
            raise ValueError(
                "balance_threshold must be >= 0"
            )

    # =====================================================
    # SAFE VALUE CONVERSION
    # =====================================================

    @staticmethod
    def _safe_queue(value) -> float:
        """
        Convert a queue value into a finite non-negative float.
        """

        try:
            value = float(value)
        except (
            TypeError,
            ValueError,
        ):
            return 0.0

        if not math.isfinite(value):
            return 0.0

        return max(
            0.0,
            value,
        )

    # =====================================================
    # NORMALISE QUEUE INPUT
    # =====================================================

    def _normalise_queues(
        self,
        queues: Mapping[str, float],
    ) -> Dict[str, float]:
        """
        Ensure every configured intersection has a queue value.
        """

        if not isinstance(queues, Mapping):
            raise TypeError(
                "queues must be a mapping such as "
                "{'A': 10, 'B': 15, 'C': 8, 'D': 12}"
            )

        result = {}

        for intersection in self.intersections:
            result[intersection] = self._safe_queue(
                queues.get(
                    intersection,
                    0.0,
                )
            )

        return result

    # =====================================================
    # MEAN QUEUE
    # =====================================================

    @staticmethod
    def _mean(values) -> float:
        if not values:
            return 0.0

        return float(
            np.mean(values)
        )

    # =====================================================
    # STANDARD DEVIATION
    # =====================================================

    @staticmethod
    def _std(values) -> float:
        if not values:
            return 0.0

        return float(
            np.std(values)
        )

    # =====================================================
    # IMBALANCE INDEX
    # =====================================================

    @staticmethod
    def _imbalance_index(
        values,
        mean_queue: float,
    ) -> float:
        """
        Normalised queue imbalance.

        Formula:

            standard_deviation / mean_queue

        Lower values indicate a more evenly distributed
        queue state.

        When mean queue is zero, imbalance is zero.
        """

        if mean_queue <= 0:
            return 0.0

        standard_deviation = float(
            np.std(values)
        )

        return float(
            standard_deviation
            / mean_queue
        )

    # =====================================================
    # MAX/MIN RATIO
    # =====================================================

    @staticmethod
    def _max_min_ratio(
        maximum: float,
        minimum: float,
    ) -> float:
        """
        Ratio between the largest and smallest queue.

        If the minimum is zero and maximum is positive,
        the ratio is reported as infinity.
        """

        if minimum <= 0:
            if maximum > 0:
                return float("inf")

            return 1.0

        return float(
            maximum / minimum
        )

    # =====================================================
    # NEIGHBOUR QUEUE DIFFERENCE
    # =====================================================

    def _neighbour_queue_difference(
        self,
        queues: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Calculate the average absolute queue difference
        between each intersection and its neighbours.

        Example:

            A neighbours = B, C

            A difference =
                mean(
                    |queue_A - queue_B|,
                    |queue_A - queue_C|
                )

        This provides an interpretable measure of local
        congestion consistency across the graph.
        """

        differences = {}

        for intersection in self.intersections:

            neighbour_ids = self.neighbours.get(
                intersection,
                [],
            )

            valid_neighbours = [
                neighbour
                for neighbour in neighbour_ids
                if neighbour in queues
            ]

            if not valid_neighbours:
                differences[intersection] = 0.0
                continue

            local_queue = queues[
                intersection
            ]

            values = [
                abs(
                    local_queue
                    - queues[neighbour]
                )
                for neighbour in valid_neighbours
            ]

            differences[intersection] = float(
                np.mean(values)
            )

        return differences

    # =====================================================
    # NETWORK NEIGHBOUR CONSISTENCY
    # =====================================================

    def _network_neighbour_consistency(
        self,
        queues: Dict[str, float],
    ) -> float:
        """
        Calculate the average absolute queue difference
        over unique graph edges.

        Lower value:
            neighbouring intersections have similar queues.

        Higher value:
            congestion is unevenly distributed between
            connected intersections.
        """

        edge_pairs = set()

        for intersection in self.intersections:

            for neighbour in self.neighbours.get(
                intersection,
                [],
            ):

                if neighbour not in queues:
                    continue

                edge = tuple(
                    sorted(
                        (
                            intersection,
                            neighbour,
                        )
                    )
                )

                edge_pairs.add(edge)

        if not edge_pairs:
            return 0.0

        differences = []

        for first, second in edge_pairs:

            differences.append(
                abs(
                    queues[first]
                    - queues[second]
                )
            )

        return float(
            np.mean(differences)
        )

    # =====================================================
    # BALANCE DECISION
    # =====================================================

    def _is_balanced(
        self,
        imbalance_index: float,
    ) -> bool:
        """
        Determine whether the queue distribution is within
        the configured imbalance threshold.

        This is an analytical threshold, not a traffic-control
        action.
        """

        return bool(
            imbalance_index
            <= self.balance_threshold
        )

    # =====================================================
    # MAIN ANALYSIS
    # =====================================================

    def analyse(
        self,
        queues: Mapping[str, float],
    ) -> QueueBalanceResult:
        """
        Analyse one network queue state.

        Parameters
        ----------
        queues:
            Dictionary containing queue lengths.

        Example
        -------
        {
            "A": 12,
            "B": 20,
            "C": 10,
            "D": 18,
        }
        """

        normalised = self._normalise_queues(
            queues
        )

        values = list(
            normalised.values()
        )

        mean_queue = self._mean(
            values
        )

        maximum = float(
            max(values)
            if values
            else 0.0
        )

        minimum = float(
            min(values)
            if values
            else 0.0
        )

        standard_deviation = self._std(
            values
        )

        queue_range = float(
            maximum - minimum
        )

        imbalance_index = (
            self._imbalance_index(
                values,
                mean_queue,
            )
        )

        max_min_ratio = (
            self._max_min_ratio(
                maximum,
                minimum,
            )
        )

        if normalised:

            most_congested = max(
                normalised,
                key=normalised.get,
            )

            least_congested = min(
                normalised,
                key=normalised.get,
            )

        else:

            most_congested = ""
            least_congested = ""

        neighbour_difference = (
            self._neighbour_queue_difference(
                normalised
            )
        )

        network_consistency = (
            self._network_neighbour_consistency(
                normalised
            )
        )

        balanced = self._is_balanced(
            imbalance_index
        )

        return QueueBalanceResult(

            intersection_queues=normalised,

            mean_queue=round(
                mean_queue,
                4,
            ),

            max_queue=round(
                maximum,
                4,
            ),

            min_queue=round(
                minimum,
                4,
            ),

            queue_std=round(
                standard_deviation,
                4,
            ),

            queue_range=round(
                queue_range,
                4,
            ),

            imbalance_index=round(
                imbalance_index,
                4,
            ),

            max_min_ratio=(
                round(
                    max_min_ratio,
                    4,
                )
                if math.isfinite(
                    max_min_ratio
                )
                else float("inf")
            ),

            most_congested_intersection=(
                most_congested
            ),

            least_congested_intersection=(
                least_congested
            ),

            balanced=balanced,

            balance_threshold=round(
                self.balance_threshold,
                4,
            ),

            neighbour_queue_difference={
                key: round(
                    value,
                    4,
                )
                for key, value
                in neighbour_difference.items()
            },

            network_neighbour_consistency=round(
                network_consistency,
                4,
            ),
        )

    # =====================================================
    # ANALYSE FROM AGENT STATISTICS
    # =====================================================

    def analyse_agent_statistics(
        self,
        agent_statistics: Mapping[str, Mapping],
        queue_key: str = "queue_length",
    ) -> QueueBalanceResult:
        """
        Analyse queues directly from agent statistics.

        Expected format:

        {
            "A": {"queue_length": 12},
            "B": {"queue_length": 20},
            "C": {"queue_length": 10},
            "D": {"queue_length": 18},
        }
        """

        queues = {}

        for intersection in self.intersections:

            state = agent_statistics.get(
                intersection,
                {},
            )

            if isinstance(state, Mapping):

                queues[intersection] = (
                    state.get(
                        queue_key,
                        0.0,
                    )
                )

            else:

                queues[intersection] = 0.0

        return self.analyse(
            queues
        )

    # =====================================================
    # ANALYSE FROM TRAFFIC AGENTS
    # =====================================================

    def analyse_controller(
        self,
        marl_controller,
    ) -> QueueBalanceResult:
        """
        Analyse queue balance directly from a
        MARLController instance.

        Uses each TrafficAgent's statistics.queue_length.

        This allows the module to work with the existing
        MARL architecture without modifying the controller.
        """

        queues = {}

        for intersection in self.intersections:

            try:

                agent = (
                    marl_controller.get_agent(
                        intersection
                    )
                )

                queue = (
                    agent.statistics.queue_length
                )

            except (
                KeyError,
                AttributeError,
            ):

                queue = 0.0

            queues[intersection] = (
                self._safe_queue(queue)
            )

        return self.analyse(
            queues
        )


# =========================================================
# CONVENIENCE FUNCTION
# =========================================================

def analyse_queue_balance(
    queues: Mapping[str, float],
    balance_threshold: float = 0.25,
) -> Dict:
    """
    Convenience function for one-shot analysis.
    """

    analyzer = QueueBalanceAnalyzer(
        balance_threshold=balance_threshold
    )

    result = analyzer.analyse(
        queues
    )

    return result.to_dict()


# =========================================================
# STANDALONE TESTS
# =========================================================

def _run_queue_balance_tests() -> None:
    """
    Standalone tests.

    These tests do not start SUMO and do not retrain PPO.
    """

    analyzer = QueueBalanceAnalyzer(
        balance_threshold=0.25
    )

    # =====================================================
    # TEST 1 — BALANCED NETWORK
    # =====================================================

    balanced_result = analyzer.analyse(
        {
            "A": 10,
            "B": 11,
            "C": 10,
            "D": 11,
        }
    )

    assert (
        balanced_result.mean_queue
        == 10.5
    )

    assert (
        balanced_result.max_queue
        == 11.0
    )

    assert (
        balanced_result.min_queue
        == 10.0
    )

    assert (
        balanced_result.most_congested_intersection
        in {"B", "D"}
    )

    assert (
        balanced_result.balanced
        is True
    )

    # =====================================================
    # TEST 2 — IMBALANCED NETWORK
    # =====================================================

    imbalanced_result = analyzer.analyse(
        {
            "A": 5,
            "B": 50,
            "C": 8,
            "D": 45,
        }
    )

    assert (
        imbalanced_result.max_queue
        == 50.0
    )

    assert (
        imbalanced_result.min_queue
        == 5.0
    )

    assert (
        imbalanced_result.queue_range
        == 45.0
    )

    assert (
        imbalanced_result.most_congested_intersection
        == "B"
    )

    assert (
        imbalanced_result.balanced
        is False
    )

    # =====================================================
    # TEST 3 — ZERO QUEUES
    # =====================================================

    empty_result = analyzer.analyse(
        {
            "A": 0,
            "B": 0,
            "C": 0,
            "D": 0,
        }
    )

    assert (
        empty_result.mean_queue
        == 0.0
    )

    assert (
        empty_result.imbalance_index
        == 0.0
    )

    assert (
        empty_result.balanced
        is True
    )

    # =====================================================
    # TEST 4 — MISSING INTERSECTION
    # =====================================================

    missing_result = analyzer.analyse(
        {
            "A": 10,
            "B": 20,
        }
    )

    assert (
        set(
            missing_result.intersection_queues.keys()
        )
        == {"A", "B", "C", "D"}
    )

    assert (
        missing_result.intersection_queues["C"]
        == 0.0
    )

    assert (
        missing_result.intersection_queues["D"]
        == 0.0
    )

    # =====================================================
    # TEST 5 — NEIGHBOUR ANALYSIS
    # =====================================================

    neighbour_result = analyzer.analyse(
        {
            "A": 10,
            "B": 20,
            "C": 10,
            "D": 20,
        }
    )

    assert (
        "A"
        in neighbour_result.neighbour_queue_difference
    )

    assert (
        "B"
        in neighbour_result.neighbour_queue_difference
    )

    assert (
        neighbour_result.network_neighbour_consistency
        >= 0.0
    )

    # =====================================================
    # TEST 6 — CONVENIENCE FUNCTION
    # =====================================================

    result = analyse_queue_balance(
        {
            "A": 10,
            "B": 10,
            "C": 10,
            "D": 10,
        }
    )

    assert isinstance(
        result,
        dict,
    )

    assert (
        result["mean_queue"]
        == 10.0
    )

    # =====================================================
    # FINAL RESULT
    # =====================================================

    print(
        "QueueBalanceAnalyzer tests: PASS"
    )

    print(
        f"Balanced network: "
        f"{balanced_result.to_dict()}"
    )

    print(
        f"Imbalanced network: "
        f"{imbalanced_result.to_dict()}"
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    _run_queue_balance_tests()