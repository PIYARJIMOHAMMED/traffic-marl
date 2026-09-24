# MAI-07 — User and Administrator Guide

## Start
Launch the Flask dashboard using the project's existing dashboard command.

## Select scenario
Choose a configured SUMO scenario before starting an episode.

## Governance
1. Prepare episode.
2. Obtain human approval.
3. Start simulation.
4. Observe PPO proposals and safety decisions.
5. In approval-required mode, review risk-triggered decisions.
6. Approve or reject according to the displayed safety-authorized state.

## Important
The human operator should never attempt to bypass the Safety Validator.

## Audit
Review:
`rl/logs/safety/dashboard_audit.csv`

Required fields:
timestamp, intersection, proposed_action, safety_approved_action, final_action, human_approval_required, human_approved, status, reason.
