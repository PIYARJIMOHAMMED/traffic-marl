from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable


class AuditLogger:
    """
    Append-only decision audit logger.

    Each RL decision can be recorded with:

    - timestamp
    - intersection
    - proposed action
    - safety-approved action
    - final action
    - governance status
    - reason
    """

    def __init__(
        self,
        output_path: str | Path,
    ):

        self.output_path = Path(
            output_path
        )

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.fieldnames = [
            "timestamp",
            "intersection",
            "proposed_action",
            "safety_approved_action",
            "final_action",
            "human_approval_required",
            "human_approved",
            "status",
            "reason",
        ]

    # ========================================================
    # WRITE ONE DECISION
    # ========================================================

    def log(
        self,
        approval_decision,
    ) -> None:

        data = approval_decision.to_dict()

        row = {
            "timestamp": data["timestamp"],
            "intersection": data["intersection"],
            "proposed_action": data["proposed_action"],
            "safety_approved_action": data[
                "safety_approved_action"
            ],
            "final_action": data["final_action"],
            "human_approval_required": data[
                "human_approval_required"
            ],
            "human_approved": data[
                "human_approved"
            ],
            "status": data["status"],
            "reason": data["reason"],
        }

        file_exists = self.output_path.exists()

        with open(
            self.output_path,
            "a",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=self.fieldnames
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)

    # ========================================================
    # LOG NETWORK DECISIONS
    # ========================================================

    def log_network(
        self,
        approval_decisions: Iterable,
    ) -> None:

        for decision in approval_decisions:
            self.log(decision)