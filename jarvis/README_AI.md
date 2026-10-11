# README_AI: handoff for agents working on Jarvis

Read this first. It tells you what this project is, what state it is in, what the user wants, and where to go next. Keep it up to date when you change the project.

## 1. What this is

Jarvis is a personal assistant inspired by the JARVIS of Iron Man. It lives in `jarvis/` in the repo `abdoo606/ai` (branch `arena/a202685b-ai`). The repo is a fork of an original Drosophila whole-brain model. **Do not modify the Drosophila files at the repo root** (`model.py`, `utils.py`, `example.ipynb`, `figures.ipynb`, `results/`, `scripts/`, `Readme.md`, `environment*.yml`). All new work goes in its own directory.

Sibling project: `human_brain_agents/`, a spiking-network prototype of a human-like sensorimotor decision loop (auditory, visual, tactile stimuli to motor command, reaction times, noise robustness, reward learning). It is paused. Its code and tests are already committed (39 tests pass). Do not resume it unless the user asks.

## 2. The user's requirements (verbatim intent)

- The user writes Levantine Arabic. **Reply in Arabic, Levantine dialect**, matching their style. Code, comments and identifiers stay in English.
- Python. Data and packages only from GitHub and PyPI (the sandbox allows only github.com, codeload, api.github.com, registry.npmjs.org, pypi.org, files.pythonhosted.org).
- Jarvis must:
  1. Learn any domain the user asks for: search the web, read sources, understand them, archive them, and use them later. **Implemented as bounded deep-learning passes** (curriculum search, subtopic expansion, deduplication, local FTS memory, and source URLs).
  2. Support a **web interface and a desktop** interface. Web is done. Desktop is only a terminal (`python -m jarvis`). A real desktop GUI is **not built**.
  3. Support voice and command execution. Voice input and output are in the web page (Web Speech API, Chrome or Edge). Command execution exists but is **off by default** (`JARVIS_ALLOW_SHELL=1`).
  4. **Be standalone: no API key, no paid service.** The user said explicitly: "I don't want an API key; I want it standalone, like the brain project." Hosted model/API backends have now been removed completely. Only the built-in offline brain and an optional model running locally through Transformers are allowed. Never add or request API keys.
- The user does not mind reusing or adapting code from other projects.

## 3. Current state (as of this handoff)

Restored onto `arena/a202685b-ai` from the earlier Jarvis work and actively developed here. The default is standalone. The optional local ensemble has separate intent, Arabic, and English model folders; all load lazily and runtime downloads are disabled by default.

Working:
- `config.py`: local-only settings, including separate `JARVIS_HF_INTENT_MODEL`, `JARVIS_HF_ARABIC_MODEL`, and `JARVIS_HF_ENGLISH_MODEL` folders. `JARVIS_BACKEND=ensemble` activates all three lazily. There are no API credentials or hosted-model settings.
- `knowledge.py`: hybrid local reasoning memory: normalized FTS5/BM25 plus weighted concept/co-occurrence tables. New chunks update the graph. Queries use graph expansion and reciprocal-rank fusion; answers use maximal marginal relevance and citations. Indexed graph lookups, WAL, a 32 MiB cache, and expansion caching keep it fast. Old FTS schema migrates automatically; old content can be graph-indexed with `rebuild_connections`.
- `research.py`: `DeepResearcher` builds a broad bilingual curriculum, runs bounded multi-pass searches, reads and deduplicates pages, discovers subtopics, and archives them locally. `learn` / `اتعلم` routes here.
- `web.py`: stdlib only. `html_to_text` (drops script, style, nav, footer, header, form), `fetch_text` (http/https only, 2 MB cap), `parse_search_results` (DuckDuckGo HTML), `search`.
- `tools.py`: `ToolRegistry`. Tools: `get_time`, `web_search`, `fetch_url`, `learn_topic`, `learn_deep`, `recall`, `summarize_topic`, `connect_ideas`, `rebuild_connections`, `list_sources`, `list_topics`, `forget_topic`, `read_file`, `write_file`, `list_files`, and `run_command` (only with shell enabled). File tools are confined to the workspace (`~/jarvis_workspace`). Results are capped at 4000 characters. Errors come back as text, never as exceptions.
- `llm.py`: `RuleBasedLLM` includes filler removal and typo-tolerant fuzzy commands. `LocalModelEnsembleLLM` first uses deterministic routing, then a small intent model for ambiguous/incomplete text, and routes replies to an Arabic or English reasoning model by script ratio. All three wrap lazy `HuggingFaceTransformersLLM` instances. Hosted-model clients do not exist.
- `agent.py`: `Assistant.ask()`. Runs a bounded tool loop (`MAX_STEPS=6`), keeps the last 40 messages, and turns LLM errors into text.
- `server.py`: FastAPI. Routes: `GET /`, `GET /api/status`, `POST /api/chat`, `POST /api/reset`, `POST /api/learn`, `GET /api/recall?q=`. Relative URLs in the UI. Run with `uvicorn jarvis.server:get_app --factory --host 0.0.0.0 --port 8000`.
- `web/index.html`: chat, learn panel, recall panel, mic and speaker buttons.
- `cli.py` and `__main__.py`: terminal REPL. Commands: `exit`, `reset`.
- `tests/test_jarvis.py`: 62 offline tests (chunking, FTS, parsing, file confinement, tool errors, intents, agent loop, end-to-end learn then recall). Run with `pytest -q jarvis/tests`. They use fixtures and scripted models, with no network.

Known gaps and bugs:
- The default offline brain is not generative, but now answers direct questions extractively, summarizes learned topics, lists sources, and emits inline citations. The optional Hugging Face backend provides generative local reasoning.
- The DuckDuckGo search was **never tested live** (the sandbox has no route to it). Parsing is tested only on a fixture. Expect breakage if the HTML changes.
- `learn_deep` filters short pages and deduplicates URLs/exact content during each run, but does not yet read PDFs or detect semantic near-duplicates across separate runs.
- Retrieval is hybrid BM25 + concept graph, not neural embeddings. The graph records co-occurrence, so it must never be presented as proof of causation.
- No authentication on the web server. Bind to 0.0.0.0 only in trusted networks (the sandbox preview needs it).
- Shell: `run_command` runs with `shell=True` in the workspace, with a timeout but no sandbox. Only enable it deliberately.
- The desktop GUI, device and smart-home control, and live voice in the terminal are not built.

## 4. Where we are heading (priority order)

1. **Continue improving autonomous deep learning and grounded offline answers.** Graph expansion, reciprocal-rank fusion, MMR sentence selection, summaries, source listing, citations, and Arabic normalization are implemented. Next add contradiction detection and learning-by-date queries.
2. **Verify live web search and learning** when network access allows it (DuckDuckGo HTML, then a fallback engine if it fails). Add a retry and a rate limit.
3. **Desktop app**: a small native-feeling window (for example pywebview wrapping the existing FastAPI server, or a Tkinter shell), plus a global hotkey. Keep the same API. Do not duplicate logic.
4. **Live voice on desktop**: Python speech input and output with packages from PyPI only (for example `SpeechRecognition` with an offline engine, `pyttsx3`). Check that the required model files are not downloaded from hosts outside the allowlist.
5. **Local model ensemble (standalone)**: intent + Arabic + English brains and a one-time public model installer are implemented. Next benchmark real hardware for Arabic quality, malformed input, tool JSON reliability, latency, and memory; never commit weights.
6. Device and smart-home control only after explicit permission and a safety design (confirmation before actions).

## 5. Rules for agents working here

- Work only on branch `arena/a202685b-ai`. Push only to it. Never create or switch branches.
- Never commit or request secrets. The user sets environment variables themselves.
- Keep `jarvis/data/` and `~/jarvis_workspace` out of Git (`jarvis/data/` is in `.gitignore`).
- Run `pytest -q jarvis/tests` before every commit. It must pass.
- After changing server code, restart the uvicorn process (it does not reload automatically).
- Keep every intelligence path local. Do not reintroduce hosted-model or paid API clients, even as opt-in options.
- Keep file tools inside the workspace. Keep shell off by default.
- When you add a tool, register it in `tools.build_registry`, give it a JSON schema, and add a test.
- Update this file and `jarvis/README.md` when you change the design.

## 6. Quick map

| File | Purpose |
|---|---|
| `config.py` | env settings, `llm_mode` |
| `knowledge.py` | SQLite FTS5 knowledge base |
| `research.py` | autonomous deep curriculum research and archiving |
| `web.py` | search, fetch, HTML to text |
| `tools.py` | tool registry and tools |
| `llm.py` | fuzzy router, language detection, local 3-model ensemble, model factory |
| `agent.py` | `Assistant` loop and history |
| `server.py` | FastAPI app, `get_app()` factory |
| `cli.py`, `__main__.py` | terminal mode |
| `scripts/install_models.py` | one-time public English/Arabic/intent model installer |
| `web/index.html` | web UI |
| `tests/test_jarvis.py` | offline tests |
| `README.md` | user-facing docs |
| `README_AI.md` | this file |

## 7. History of decisions

- The user first asked for a JARVIS-like assistant, with voice, command execution, and learning of any domain. Platforms: web and desktop.
- Brain choice: an LLM-style tool-using agent with a knowledge base, not the spiking model. Reason: the spiking model cannot serve as an assistant core.
- The user rejected all API-key/hosted-model options. They were removed completely. Runtime intelligence must remain on-device.
- The brain project was paused by the user. Its emotion regions (amygdala, orbitofrontal, insula, anterior cingulate) were removed on purpose. Do not add emotion back.
