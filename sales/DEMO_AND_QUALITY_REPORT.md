# Zehnora technical quality and demo report

**Baseline commit:** `4913ae0`  
**Version:** 0.7.0  
**Date:** 2026-10-11

## Measured evidence

| Check | Result |
|---|---|
| Offline tests | 76 passed |
| Statement coverage | 82% package total, tests omitted |
| Python CI | 3.10, 3.11, 3.12 passed |
| Package CI | wheel/sdist build and CLI smoke passed |
| Ruff | passed |
| Bandit | passed |
| pip-audit | no known vulnerabilities |
| Clean wheel | installed; package/module version 0.7.0 |
| Web health | HTTP 200, version 0.7.0 |
| Buyer source bundle | deterministic test passed; manual exclusion inspection passed |

Coverage detail from the measured baseline includes 91% for `agent.py`, 95% for
`knowledge.py`, 87% for `research.py`, 90% for `server.py`, 86% for `tools.py`,
and 83% for `web.py`. `llm.py` measured 68%; CLI entry paths measured 0% and are
smoke-tested separately. Coverage measures executed statements, not product
correctness or model quality.

## Honest limitations

- No open-model weights are included or committed.
- No independent real-hardware benchmark validates the profile quality claims.
- The default offline brain is extractive/command-oriented, not a frontier LLM.
- DuckDuckGo live behavior was not tested from the restricted build network.
- No production users, uptime record, revenue, or customer acceptance exists.
- No independent penetration test or native desktop application exists.
- Co-occurrence graph edges are not proof of causality.

## 20-minute buyer demonstration

1. **Integrity (2 min):** show closing commit, ZIP checksum, internal manifest.
2. **Local operation (3 min):** start web/CLI with no API credential; show status.
3. **Arabic intent (3 min):** use incomplete/misspelled Arabic commands and show
   routing without claiming benchmark-level language quality.
4. **Grounded memory (4 min):** load a fixed licensed fixture, ask a connected
   question, inspect citations and concept links.
5. **Research (3 min):** if network is available, run a buyer-approved bounded
   topic; otherwise show deterministic fixture-based behavior.
6. **Security (3 min):** demonstrate rejected private URL, escaped file path,
   unauthenticated request, and disabled command execution.
7. **Rebuild (2 min):** run tests and package build or show immutable CI logs.

## Reproduction commands

```bash
python -m pytest -q zehnora/tests
python -m coverage run --source=zehnora -m pytest -q zehnora/tests
python -m coverage report --omit='zehnora/tests/*' --fail-under=80
ruff check .
bandit -q -r zehnora -x zehnora/tests
pip-audit
python -m build
python -m zehnora.scripts.build_asset_bundle --output sale_artifacts --version 0.7.0
```

The demo operator must distinguish measured facts from roadmap items and must not
claim legal clearance, perfect exclusivity, guaranteed security, or Claude-level
equivalence.
