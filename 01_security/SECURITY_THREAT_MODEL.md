# MAI-07 — Security Threat Model

## System
Multi-Agent Reinforcement Learning for Network-Wide Adaptive Traffic Control.

## Security objectives
1. Prevent unsafe signal actions.
2. Prevent bypass of the Safety Validator.
3. Prevent unauthorized control of SUMO/TraCI.
4. Preserve human approval gates.
5. Preserve auditability and traceability.
6. Prevent model/runtime tampering.
7. Prevent secret leakage.
8. Reject malformed or out-of-range actions.

## Trust boundaries
- Browser/dashboard → Flask backend
- Flask backend → governance layer
- PPO policy → Safety Validator
- Safety Validator → Human Approval Gate
- Human Approval Gate → Traffic Controller
- Traffic Controller → SUMO/TraCI
- Runtime → audit logs and result artifacts

## Threat register

| ID | Threat | Impact | Mitigation | Evidence |
|---|---|---|---|---|
| T01 | Invalid PPO action | Unsafe signal command | Validate action domain {0,1} | Unit/security tests |
| T02 | Safety bypass | Unsafe command reaches controller | Controller accepts safety-approved action path | Architecture + tests |
| T03 | Human approval bypass | Unauthorized intervention | Explicit approval/rejection state | Governance tests |
| T04 | Premature phase change | Traffic instability | Minimum-green and yellow protection | Safety audit |
| T05 | Emergency command abuse | Unsafe emergency behavior | Emergency path remains safety-authorized | Governance tests |
| T06 | Scenario tampering during live run | Invalid experiment | Scenario selected before episode | Operational procedure |
| T07 | Model artifact tampering | Untrusted policy | Hash/checksum and controlled artifact directory | Reproducibility checklist |
| T08 | Secret leakage | Credential compromise | Environment variables; no secrets in source | Secret scan |
| T09 | Arbitrary OS/tool execution | System compromise | Tool allow-list; no arbitrary model tool | Architecture |
| T10 | Audit log tampering | Loss of evidence | Append-oriented audit handling and backups | Audit procedure |
| T11 | Malformed API input | Runtime failure | Schema/type/range validation | API tests |
| T12 | Dependency vulnerability | Supply-chain risk | Dependency audit and pinned versions | Dependency report |

## Residual risks
- SUMO is a simulation environment, not a production traffic-signal controller.
- The dashboard is a research prototype and should not directly control physical infrastructure.
- Identical traffic realizations in earlier Level-1 runs limit statistical uncertainty evidence.
- Simulation emergency-braking warnings require continued monitoring.

## Security boundary
The PPO policy proposes actions; it does not have authority to directly execute arbitrary system commands. Safety and governance layers remain between policy output and SUMO actuation.
