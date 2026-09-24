# MAI-07 — Agent Tool Allow-List and Permission Model

## PPO-MARL — ALLOWED
- Read structured traffic observations.
- Read local intersection state.
- Read neighbour information.
- Propose signal action 0 or 1.
- Produce model inference output.

## PPO-MARL — NOT ALLOWED
- Execute arbitrary shell commands.
- Modify model weights.
- Modify safety rules.
- Disable the Safety Validator.
- Directly control SUMO outside the controller interface.
- Read secrets or credentials.
- Modify audit history.
- Change experiment scenario files during an active episode.
- Approve its own human-governance request.

## Safety Validator — ALLOWED
- Read proposed action.
- Read safety state.
- Check minimum green duration.
- Check yellow transition protection.
- Override unsafe proposals.
- Produce a safety-authorized action.

## Human operator — ALLOWED
- Approve a pending review.
- Reject a pending proposal.
- Pause/resume governed execution.
- Start an approved episode.

## Traffic Controller — ALLOWED
- Apply only the final safety/governance-authorized action.
- Read actual SUMO phase/state.
- Enforce signal transition timing.

## SUMO/TraCI — ALLOWED
- Simulation state queries.
- Vehicle telemetry queries.
- Traffic-light phase commands through the controller.

## Architectural rule
The policy is advisory. The Safety Validator and governance layer retain authority over actuation.
