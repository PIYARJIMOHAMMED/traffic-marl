# MAI-07 — Level 2 Test Execution Procedure

Run from project root.

## 1. Syntax
`python -m compileall rl`

Expected: no compilation errors.

## 2. Unit/integration tests
`pytest -q`

Expected: all tests pass; an existing PytestReturnNotNoneWarning may remain non-blocking if no test fails.

## 3. Model/environment validation
`python rl\evaluation\level1_research_evaluation.py --episodes 1`

Use only if a smoke test is needed; do not repeat the expensive full Level-1 campaign unnecessarily.

## 4. Safety audit
`python rl\evaluation\level1_research_evaluation.py --audit-only`

Expected from the current audit:
- records: 24888
- safety overrides: 10699
- violations: 0
- invalid actions: 0

## 5. Security scans
Run the commands in `LEVEL2_COMMANDS.md`.

Archive terminal output into `06_evidence`.
