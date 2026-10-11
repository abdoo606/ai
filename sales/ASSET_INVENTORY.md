# Zehnora asset inventory

## Included current-tree engineering assets

| Area | Location | Evidence |
|---|---|---|
| Core agent and tool loop | `zehnora/agent.py`, `zehnora/tools.py` | Unit tests |
| Local model orchestration | `zehnora/llm.py` | Intent/ensemble/deliberation tests |
| Knowledge and graph memory | `zehnora/knowledge.py` | Search, migration, graph tests |
| Autonomous research | `zehnora/research.py`, `zehnora/web.py` | Fixture and security tests |
| Local web product | `zehnora/server.py`, `zehnora/web/` | API/auth/rate-limit tests |
| CLI and packaging | `zehnora/cli.py`, `pyproject.toml` | Clean-wheel smoke test |
| Model profile installer | `zehnora/scripts/install_models.py` | Dry-run and pinned-manifest process |
| CI/security | `.github/workflows/zehnora-ci.yml` | Python 3.10–3.12 green checks |
| Product documentation | `zehnora/*.md`, root `README.md` | Review required |
| Transaction room | `sales/` | Counsel/buyer review required |

## Excluded generated or sensitive assets

- `.git/` history from the clean buyer bundle
- `zehnora/data/` runtime databases
- `zehnora/models/` downloaded weights
- Local workspaces, caches, wheels, build outputs, credentials, and signing keys

## Third-party boundary

See `THIRD_PARTY_NOTICES.md`. Dependency and model ownership does not transfer;
only whatever license rights are validly available may accompany the product.

## Version at transaction-room preparation

- Product version: 0.7.0
- Test baseline: 76 passing tests; 82% measured package statement coverage
- Supported CI interpreters: Python 3.10, 3.11, 3.12
- Package: `local-zehnora-assistant`
- Console commands: `zehnora-local`, `zehnora-local-web`
