# MAI-07 — Level 2 Engineering Gate

## Current evidence

### Docker
Docker Desktop evidence shows the `traffic-marl` Compose project with:
- `traffic_api` running on 8000:8000
- `traffic_db` running on 5432:5432
- `traffic_mlflow` present
- local images for `traffic-marl-api`, `postgres:16`, and `python:3.10-slim`

This verifies that the existing Docker/Compose stack is operational in the user's environment.

### Automated tests
- `python -m compileall rl` — PASS
- `pytest -q` — 20 passed, 1 skipped, 1 warning

### Dependency scan
`pip-audit` executed and found 27 known vulnerabilities in 4 packages. This is a finding to remediate, not a PASS.

### Static security scan
Bandit:
- High: 0
- Medium: 5
- Low: 150

The five medium findings are XML parsing warnings for project SUMO XML files. They should be remediated with defusedxml or explicitly risk-accepted/documented.

### Safety audit
- 24,888 audit records
- 0 explicit violations
- 0 invalid actions

## CI/CD
The supplied GitHub Actions workflow runs:
1. compile
2. pytest
3. pip-audit
4. Bandit
5. Docker build
6. Docker Compose configuration validation

Security scans are report-only (`continue-on-error`) while known findings are being remediated. This is intentional: the pipeline should expose findings rather than hide them.

## Important scope statement
The Docker stack currently containerizes the API/database/MLflow services. SUMO and the live Flask dashboard are not automatically claimed as containerized by this file. Full-project containerization should only be claimed if the dashboard and SUMO runtime are also successfully run in the containerized environment.
