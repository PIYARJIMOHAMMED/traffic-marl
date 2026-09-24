# MAI-07 — Human-Governed Agent Control Specification

## Control flow

Prepare Episode
→ Human Approval
→ SUMO starts
→ PPO-MARL proposes action
→ Safety Validator
→ Human Approval Gate when required
→ Traffic Controller
→ SUMO

## Governance modes

### Automatic
After episode authorization, safety-approved actions are applied automatically.

### Approval Required
Risk-triggered decisions pause for human review. The human can approve or reject. Rejection retains the safety-authorized action rather than granting arbitrary control.

### Emergency
The safety policy can immediately replace an unsafe proposed action. Human approval is still required for episode authorization.

## Audit requirements
Each governance event should record:
- timestamp
- intersection
- proposed action
- safety-approved action
- final action
- human approval required
- human approved
- status
- reason

## Safety priority
Human governance does not override the safety validator with an unsafe action. Safety authorization remains authoritative.
