# MAI-07 — Level 2 Engineering, Security and Governance Evidence Report

## Objective
Level 2 hardens the research prototype with explicit security controls, governance boundaries, testing procedures, reproducibility requirements, and evidence collection.

## Existing verified evidence
The Level-1 safety audit currently contains:
- 24,888 records
- 10,699 safety overrides
- 10,673 automatic approvals
- 2,085 emergency approvals
- 1,171 human approvals
- 260 human rejections
- 0 explicit violations
- 0 invalid actions

Safety overrides are interventions, not violations.

## Engineering controls
1. Safety Validator sits between PPO proposals and actuation.
2. Traffic Controller enforces minimum-green/yellow timing.
3. Human approval is required for governed episode startup.
4. Approval-required mode pauses risk-triggered decisions.
5. Emergency mode prioritizes safety-authorized action.
6. Audit records preserve decision traceability.
7. PPO is restricted to observation and action proposal responsibilities.

## Security testing
Use the threat model, security test plan and red-team cases in this package.

## Reproducibility
The saved 500K model is treated as an immutable evaluation artifact. Evaluation must not retrain or modify it.

## Limitations
- This is a SUMO research prototype and not a physical traffic-signal controller.
- Statistical uncertainty from the initial three-episode-per-scenario run was limited because traffic realizations were identical.
- Some high-demand scenarios produced SUMO emergency-braking warnings; these are simulation/environment observations and are not counted as PPO safety violations.

## Level-2 completion rule
Mark Level 2 complete only after the local commands have been executed and their outputs are archived with this report.
