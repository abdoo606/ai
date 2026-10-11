# README_AI: handoff for agents working on Zehnora

Read this first. It tells you what this project is, what state it is in, what the user wants, and where to go next. Keep it up to date when you change the project.

## 1. What this is

Zehnora (`ذهنورا`) is a private, API-free Arabic/English personal research assistant in the repo `abdoo606/ai` (branch `arena/a202685b-ai`). The working name replaced the earlier fictional-assistant brand to reduce obvious trademark risk; it still requires formal clearance before transfer. The current tree contains Zehnora and its transaction materials only. The unrelated Drosophila/connectome code, data, notebooks, environments, and outputs were removed at the owner's direction, although they remain visible in historical Git revisions.

## 2. The user's requirements (verbatim intent)

- The user writes Levantine Arabic. **Reply in Arabic, Levantine dialect**, matching their style. Code, comments and identifiers stay in English.
- Python. Data and packages only from GitHub and PyPI (the sandbox allows only github.com, codeload, api.github.com, registry.npmjs.org, pypi.org, files.pythonhosted.org).
- Zehnora must:
  1. Learn any domain the user asks for: search the web, read sources, understand them, archive them, and use them later. **Implemented as bounded deep-learning passes** (curriculum search, subtopic expansion, deduplication, local FTS memory, and source URLs).
  2. Support a **web interface and a desktop** interface. Web is done. Desktop is only a terminal (`python -m zehnora`). A real desktop GUI is **not built**.
  3. Support voice and command execution. Voice input and output are in the web page (Web Speech API, Chrome or Edge). Command execution exists but is **off by default** (`ZEHNORA_ALLOW_SHELL=1`).
  4. **Be standalone: no API key, no paid service.** The user said explicitly: "I don't want an API key; I want it standalone, like the brain project." Hosted model/API backends have now been removed completely. Only the built-in offline brain and an optional model running locally through Transformers are allowed. Never add or request API keys.
- The user does not mind reusing or adapting code from other projects.

## 3. Current state (as of this handoff)

The current transaction strategy is a **one-time assignment/sale of the entire Zehnora asset**, not subscriptions, seats, retail tiers, or individual licenses. The repository includes a buyer diligence room under `sales/`; version 0.7.0 uses proprietary current-revision metadata with an explicit warning that earlier public MIT grants and public Git history cannot be revoked. Counsel must review title, historical licensing, contributors, third-party/model terms, brand, tax, security, and transfer documents before closing.

The runtime default is standalone. The strongest optional mode is `ZEHNORA_BACKEND=powerful`: intent → senior reasoner/tool use → adversarial critic → Arabic/English revision. Logical roles share one strong physical model pipeline in memory. All load lazily and runtime downloads are disabled by default.

Working:
- `config.py`: local-only settings for intent, reasoner, critic, Arabic writer, and English writer paths plus 1–3 verification rounds. `powerful` activates deliberation; roles share `zehnora/models/reasoning` by default. There are no API credentials or hosted-model settings.
- `knowledge.py`: hybrid local reasoning memory: normalized FTS5/BM25 plus weighted concept/co-occurrence tables. New chunks update the graph. Queries use graph expansion and reciprocal-rank fusion; answers use maximal marginal relevance and citations. Indexed graph lookups, WAL, a 32 MiB cache, and expansion caching keep it fast. Old FTS schema migrates automatically; old content can be graph-indexed with `rebuild_connections`.
- `research.py`: `DeepResearcher` builds a broad bilingual curriculum, runs bounded multi-pass searches, reads and deduplicates pages, discovers subtopics, and archives them locally. `learn` / `اتعلم` routes here.
- `web.py`: stdlib only. `html_to_text` (drops script, style, nav, footer, header, form), `fetch_text` (http/https only, 2 MB cap), `parse_search_results` (DuckDuckGo HTML), `search`.
- `tools.py`: `ToolRegistry`. Tools: `get_time`, `web_search`, `fetch_url`, `learn_topic`, `learn_deep`, `recall`, `summarize_topic`, `connect_ideas`, `rebuild_connections`, `list_sources`, `list_topics`, `forget_topic`, `read_file`, `write_file`, `list_files`, and `run_command` (only with shell enabled). File tools are confined to the workspace (`~/zehnora_workspace`). Results are capped at 4000 characters. Errors come back as text, never as exceptions.
- `llm.py`: fuzzy commands and language detection remain. `DeliberativeLocalLLM` uses intent, senior reasoner, adversarial critic, and final language writer passes. `HuggingFaceTransformersLLM` caches pipelines by model path/device so logical roles share weights rather than duplicating a 14B/32B model. Hosted-model clients do not exist.
- `agent.py`: `Assistant.ask()`. Runs a bounded tool loop (`MAX_STEPS=6`), keeps the last 40 messages, and turns LLM errors into text.
- `server.py`: hardened FastAPI boundary with optional constant-time bearer/header authentication, failed-auth throttling, per-client expensive-route rate limiting, bounded Pydantic inputs, health endpoint, no-store/security headers, and routes: `GET /`, `GET /api/status`, `POST /api/chat`, `POST /api/reset`, `POST /api/learn`, `GET /api/recall?q=`. Relative URLs in the UI. Run with `uvicorn zehnora.server:get_app --factory --host 0.0.0.0 --port 8000`.
- `web/index.html`: chat, learn panel, recall panel, mic and speaker buttons.
- `cli.py` and `__main__.py`: terminal REPL. Commands: `exit`, `reset`.
- `tests/test_zehnora.py` plus `test_server.py`: 76 offline tests (including deterministic source-bundle coverage, chunking, FTS, parsing, file confinement, tool errors, intents, agent loop, and end-to-end learn then recall). The measured package statement coverage is 82%, with an 80% CI floor. Run with `coverage run --source=zehnora -m pytest -q zehnora/tests`. They use fixtures and scripted models, with no network.
- `scripts/build_asset_bundle.py`: builds a deterministic, source-only buyer archive and SHA-256 file. It excludes Git metadata/history, model weights, runtime databases, credentials, caches, and generated artifacts. This archive is a technical delivery aid, not evidence of title or legal clearance.
- `../sales/`: includes a plain-Arabic seller brief, MIT-impact memo, USD 2,500 fast-close recommendation, buyer teaser/FAQ, measured quality report, public-source prospect pipeline, outreach templates, counsel intake, and a 14-day sale runbook. Prospects are candidates only; none has expressed interest.

Known gaps and bugs:
- The default offline brain is not generative, but now answers direct questions extractively, summarizes learned topics, lists sources, and emits inline citations. The optional Hugging Face backend provides generative local reasoning.
- The DuckDuckGo search was **never tested live** (the sandbox has no route to it). Parsing is tested only on a fixture. Expect breakage if the HTML changes.
- `learn_deep` filters short pages and deduplicates URLs/exact content during each run, but does not yet read PDFs or detect semantic near-duplicates across separate runs.
- Retrieval is hybrid BM25 + concept graph, not neural embeddings. The graph records co-occurrence, so it must never be presented as proof of causation.
- Optional `ZEHNORA_ACCESS_TOKEN` authentication, failed-auth throttling, and request rate limits are implemented. Packaged web startup binds to loopback by default; any non-loopback commercial deployment still needs TLS and network controls.
- Commands require both enablement and an executable allowlist, use `shell=False`, a restricted PATH, workspace cwd, and a 60-second cap. Consumer releases still need OS sandboxing before enabling commands.
- The desktop GUI, device and smart-home control, and live voice in the terminal are not built.

## 4. Where we are heading (priority order)

1. **Continue improving autonomous deep learning and grounded offline answers.** Graph expansion, reciprocal-rank fusion, MMR sentence selection, summaries, source listing, citations, and Arabic normalization are implemented. Next add contradiction detection and learning-by-date queries.
2. **Complete whole-asset transaction readiness.** Packaging, multi-version CI, API security tests, SSRF controls, SECURITY.md, deterministic clean-source bundling, and a release checklist exist. Still required: seller/title and contributor evidence, historical-license counsel review, third-party/model license matrix, real-model/hardware benchmarks, an SBOM, OS sandboxing, independent penetration testing, signed delivery checksums, brand clearance, definitive assignment documents, tax advice, and buyer acceptance.
3. **Verify live web search and learning** when network access allows it (DuckDuckGo HTML, then a fallback engine if it fails). Add a retry and a rate limit.
4. **Desktop app**: a small native-feeling window (for example pywebview wrapping the existing FastAPI server, or a Tkinter shell), plus a global hotkey. Keep the same API. Do not duplicate logic.
5. **Live voice on desktop**: Python speech input and output with packages from PyPI only (for example `SpeechRecognition` with an offline engine, `pyttsx3`). Check that the required model files are not downloaded from hosts outside the allowlist.
6. **Local model ensemble (standalone)**: compact/balanced/powerful/max/frontier profiles and deliberate reason→critic→revision are implemented. Next benchmark real hardware for Arabic quality, malformed input, tool JSON reliability, latency, memory, and quantization; never commit weights.
7. Device and smart-home control only after explicit permission and a safety design (confirmation before actions).

## 5. Rules for agents working here

- Work only on branch `arena/a202685b-ai`. Push only to it. Never create or switch branches.
- Never commit or request secrets. The user sets environment variables themselves.
- Keep `zehnora/data/`, model weights, buyer-confidential records, credentials, generated sale bundles, and `~/zehnora_workspace` out of Git.
- Do not reintroduce unrelated connectome/fruit-fly materials into the current tree.
- Do not describe this as a subscription, seat license, retail tier, paid beta, or sale of perfect exclusivity. The intended transaction is one whole-asset assignment, subject to counsel-reviewed historical-license disclosures.
- Run `pytest -q zehnora/tests` before every commit. It must pass.
- After changing server code, restart the uvicorn process (it does not reload automatically).
- Keep every intelligence path local. Do not reintroduce hosted-model or paid API clients, even as opt-in options.
- Keep file tools inside the workspace. Keep shell off by default.
- When you add a tool, register it in `tools.build_registry`, give it a JSON schema, and add a test.
- Update this file and `zehnora/README.md` when you change the design.

## 6. Quick map

| File | Purpose |
|---|---|
| `config.py` | env settings, `llm_mode` |
| `knowledge.py` | SQLite FTS5 knowledge base |
| `research.py` | autonomous deep curriculum research and archiving |
| `web.py` | search, fetch, HTML to text |
| `tools.py` | tool registry and tools |
| `llm.py` | fuzzy router, shared local models, deliberate reasoning/critique/revision |
| `agent.py` | `Assistant` loop and history |
| `server.py` | authenticated/rate-limited FastAPI app, health/status routes |
| `cli.py`, `__main__.py` | terminal mode |
| `scripts/install_models.py` | one-time compact/balanced/powerful/max/frontier local model installer |
| `scripts/build_asset_bundle.py` | deterministic source-only buyer archive and checksums |
| `../sales/` | transaction perimeter, diligence, draft terms, and handover room |
| `../THIRD_PARTY_NOTICES.md` | dependency/model/content and historical-license diligence map |
| `web/index.html` | web UI |
| `tests/` | offline unit, security, API, migration, orchestration tests |
| `SECURITY.md`, `RELEASE_CHECKLIST.md` | security and commercial-release gates |
| `pyproject.toml` | wheel/sdist metadata and console commands |
| `README.md` | user-facing docs |
| `README_AI.md` | this file |

## 7. History of decisions

- The user first asked for a cinematic-style assistant with voice, command execution, and learning of any domain. Platforms: web and desktop.
- On 2026-10-11 the product was fully rebranded to the coined working name Zehnora (`ذهنورا`). Exact-name GitHub search returned no repository, but `sales/BRAND_CLEARANCE.md` explicitly blocks paid launch pending formal trademark/domain/common-law clearance.
- Brain choice: an LLM-style tool-using agent with a knowledge base, not the spiking model. Reason: the spiking model cannot serve as an assistant core.
- The user rejected all API-key/hosted-model options. They were removed completely. Runtime intelligence must remain on-device.
- On 2026-10-11 the user rejected retail/per-user licensing and requested one serious sale of the entire AI asset. Transaction-room documents and a clean-source delivery builder were added for that strategy.
- On 2026-10-11 the user directed removal of the unrelated fruit-fly/Drosophila project. It is absent from the current tree but remains in historical commits; that history and possible prior MIT grants must be disclosed rather than concealed.
