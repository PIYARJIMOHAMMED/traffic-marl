# MAI-07 — Level 2 Commands

## From project root

### Test suite
`pytest -q`

### Syntax
`python -m compileall rl`

### Safety audit
`python rl\evaluation\level1_research_evaluation.py --audit-only`

### Dependency inventory
`python -m pip freeze > rl\results\level2_pip_freeze.txt`

### Check installed package metadata
`python -m pip list`

### Optional pip vulnerability audit
`python -m pip install pip-audit`
`pip-audit`

### Optional Bandit static security scan
`python -m pip install bandit`
`bandit -r rl backend dashboard simulation -ll`

### Optional secret scan
`python -m pip install detect-secrets`
`detect-secrets scan > rl\results\level2_detect_secrets.json`

Only archive outputs after reviewing findings. Do not hide or delete findings.
