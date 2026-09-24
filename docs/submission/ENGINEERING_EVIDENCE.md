\# Engineering Evidence



\## Validation



Final unified evaluation:



\- 150 expected episodes

\- 150 completed

\- 0 runtime errors

\- 0 missing metrics

\- 0 invalid episodes

\- 0 negative throughput records



Status: PASS



\## Testing



Safety module tests cover:



\- valid action

\- minimum-green constraint

\- yellow-transition protection

\- invalid action

\- network-level action validation



\## Failure Handling



The system validates actions before execution.



Invalid or unsafe actions are overridden by the safety layer.



\## Logging



Safety and governance decisions are recorded through the audit logging

component.



\## Containerization



The repository contains Docker configuration including:



\- backend/Dockerfile

\- docker-compose.yml



\## Observability



Evaluation outputs provide structured records for:



\- reward

\- waiting time

\- queue length

\- average speed

\- throughput

\- episode

\- policy

\- scenario



\## Dependency Management



Python dependencies are maintained through project requirements files.



\## Reproducibility



Experiments use fixed scenario definitions, explicit policies, episode counts,

simulation limits and saved model checkpoints.



\## Security Limitation



The current prototype should not be connected directly to real traffic

infrastructure. Authentication, authorization, secrets management and

operational controls require additional deployment hardening before field use.

