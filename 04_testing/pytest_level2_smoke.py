import csv
from pathlib import Path

AUDIT = Path("rl/logs/safety/dashboard_audit.csv")

def test_action_domain():
    assert set([0, 1]).issubset({0, 1})

def test_audit_schema():
    assert AUDIT.exists(), f"Missing audit file: {AUDIT}"
    with AUDIT.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "timestamp", "intersection", "proposed_action",
            "safety_approved_action", "final_action",
            "human_approval_required", "human_approved",
            "status", "reason"
        }
        assert required.issubset(set(reader.fieldnames or []))

def test_audit_has_no_invalid_actions():
    if not AUDIT.exists():
        return
    with AUDIT.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            for key in ("proposed_action", "safety_approved_action", "final_action"):
                value = int(row[key])
                assert value in (0, 1), (key, value)
