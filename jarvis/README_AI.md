# README_AI: handoff for agents working on Jarvis

Read this first. It tells you what this project is, what state it is in, what the user wants, and where to go next. Keep it up to date when you change the project.

## 1. What this is

Jarvis is a personal assistant inspired by the JARVIS of Iron Man. It lives in `jarvis/` in the repo `abdoo606/ai` (branch `arena/6058b526-ai`). The repo is a fork of an original Drosophila whole-brain model. **Do not modify the Drosophila files at the repo root** (`model.py`, `utils.py`, `example.ipynb`, `figures.ipynb`, `results/`, `scripts/`, `Readme.md`, `environment*.yml`). All new work goes in its own directory.

Sibling project: `human_brain_agents/`, a spiking-network prototype of a human-like sensorimotor decision loop (auditory, visual, tactile stimuli to motor command, reaction times, noise robustness, reward learning). It is paused. Its code and tests are already committed (39 tests pass). Do not resume it unless the user asks.

## 2. The user's requirements (verbatim intent)

- The user writes Levantine Arabic. **Reply in Arabic, Levantine dialect**, matching their style. Code, comments and identifiers stay in English.
- Python. Data and packages only from GitHub and PyPI (the sandbox allows only github.com, codeload, api.github.com, registry.npmjs.org, pypi.org, files.pythonhosted.org).
- Jarvis must:
  1. Learn any domain the user asks for: search the web, read sources, understand them, archive them, and use them later. **Done in basic form** (BM25 retrieval over stored chunks, with source URLs).
  2. Support a **web interface and a desktop** interface. Web is done. Desktop is only a terminal (`python -m jarvis`). A real desktop GUI is **not built**.
  3. Support voice and command execution. Voice input and output are in the web page (Web Speech API, Chrome or Edge). Command execution exists but is **off by default** (`JARVIS_ALLOW_SHELL=1`).
  4. **Be standalone: no API key, no paid service.** The user said explicitly: "I don't want an API key; I want it standalone, like the brain project." The default backend is `offline`. The OpenAI-compatible backend exists only as an opt-in (`JARVIS_BACKEND=openai`). Do not make it the default. Do not ask the user for keys in chat.
- The user does not mind reusing or adapting code from other projects.

## 3. Current state (as of this handoff)

Committed on `arena/6058b526-ai`: `1f94471` (Jarvis first version). The standalone change (default offline, optional backend) and this file are in the commit after it.

Working:
- `config.py`: settings from env vars (`JARVIS_BACKEND`, `JARVIS_LLM_API_KEY`, `JARVIS_LLM_BASE_URL`, `JARVIS_LLM_MODEL`, `JARVIS_HOME`, `JARVIS_WORKSPACE`, `JARVIS_ALLOW_SHELL`).
- `knowledge.py`: `KnowledgeBase` on SQLite with FTS5. Tables: `topics`, `sources`, virtual `chunks`. Chunks are about 800 characters, split on paragraphs. Search uses BM25 with `OR` over quoted tokens. `has_source` prevents duplicates. Sources are cited by URL.
- `web.py`: stdlib only. `html_to_text` (drops script, style, nav, footer, header, form), `fetch_text` (http/https only, 2 MB cap), `parse_search_results` (DuckDuckGo HTML), `search`.
- `tools.py`: `ToolRegistry`. Tools: `get_time`, `web_search`, `fetch_url`, `learn_topic`, `recall`, `list_topics`, `forget_topic`, `read_file`, `write_file`, `list_files`, and `run_command` (only with shell enabled). File tools are confined to the workspace (`~/jarvis_workspace`). Results are capped at 4000 characters. Errors come back as text, never as exceptions.
- `llm.py`: `RuleBasedLLM` (the offline brain). It maps English and Arabic commands to tools with regexes, then returns the tool result as text. `OpenAICompatibleLLM` (opt-in, tool calling over urllib). `make_llm(settings)` chooses between them.
- `agent.py`: `Assistant.ask()`. Runs a bounded tool loop (`MAX_STEPS=6`), keeps the last 40 messages, and turns LLM errors into text.
- `server.py`: FastAPI. Routes: `GET /`, `GET /api/status`, `POST /api/chat`, `POST /api/reset`, `POST /api/learn`, `GET /api/recall?q=`. Relative URLs in the UI. Run with `uvicorn jarvis.server:get_app --factory --host 0.0.0.0 --port 8000`.
- `web/index.html`: chat, learn panel, recall panel, mic and speaker buttons.
- `cli.py` and `__main__.py`: terminal REPL. Commands: `exit`, `reset`.
- `tests/test_jarvis.py`: 38 offline tests (chunking, FTS, parsing, file confinement, tool errors, intents, agent loop, end-to-end learn then recall). Run with `pytest -q jarvis/tests`. They use fixtures and scripted models, with no network.

Known gaps and bugs:
- The offline brain is only a command router. It cannot reason, summarise, or answer free questions. Recall returns raw chunks, not synthesised answers.
- The DuckDuckGo search was **never tested live** (the sandbox has no route to it). Parsing is tested only on a fixture. Expect breakage if the HTML changes.
- `learn_topic` stores the first N readable pages. It does not check quality, deduplicate near-identical content, or read PDFs.
- Chunks are ranked by BM25 only. There are no embeddings.
- No authentication on the web server. Bind to 0.0.0.0 only in trusted networks (the sandbox preview needs it).
- Shell: `run_command` runs with `shell=True` in the workspace, with a timeout but no sandbox. Only enable it deliberately.
- The desktop GUI, device and smart-home control, and live voice in the terminal are not built.

## 4. Where we are heading (priority order)

1. **Make the offline assistant genuinely useful without any model.** Add extractive answering: pick the best sentences from the top hits and present them with their URLs. Add more intents (summarise a topic, list sources for a topic, show what was learned on a date). Add better Arabic normalisation (alef and ya variants, tashkeel removal) in `_fts_query` and in the intent router.
2. **Verify live web search and learning** when network access allows it (DuckDuckGo HTML, then a fallback engine if it fails). Add a retry and a rate limit.
3. **Desktop app**: a small native-feeling window (for example pywebview wrapping the existing FastAPI server, or a Tkinter shell), plus a global hotkey. Keep the same API. Do not duplicate logic.
4. **Live voice on desktop**: Python speech input and output with packages from PyPI only (for example `SpeechRecognition` with an offline engine, `pyttsx3`). Check that the required model files are not downloaded from hosts outside the allowlist.
5. **Local model option (standalone)**: if a local model can be installed from PyPI with its weights bundled (not downloaded from Hugging Face, which is not on the allowlist), add it as a third backend. Decide with the user first, because it changes the install size.
6. Device and smart-home control only after explicit permission and a safety design (confirmation before actions).

## 5. Rules for agents working here

- Work only on branch `arena/6058b526-ai`. Push only to it. Never create or switch branches.
- Never commit or request secrets. The user sets environment variables themselves.
- Keep `jarvis/data/` and `~/jarvis_workspace` out of Git (`jarvis/data/` is in `.gitignore`).
- Run `pytest -q jarvis/tests` before every commit. It must pass.
- After changing server code, restart the uvicorn process (it does not reload automatically).
- Keep the default path standalone. Any feature that calls a paid API must be opt-in.
- Keep file tools inside the workspace. Keep shell off by default.
- When you add a tool, register it in `tools.build_registry`, give it a JSON schema, and add a test.
- Update this file and `jarvis/README.md` when you change the design.

## 6. Quick map

| File | Purpose |
|---|---|
| `config.py` | env settings, `llm_mode` |
| `knowledge.py` | SQLite FTS5 knowledge base |
| `web.py` | search, fetch, HTML to text |
| `tools.py` | tool registry and tools |
| `llm.py` | `RuleBasedLLM` (offline), `OpenAICompatibleLLM` (opt-in), `make_llm` |
| `agent.py` | `Assistant` loop and history |
| `server.py` | FastAPI app, `get_app()` factory |
| `cli.py`, `__main__.py` | terminal mode |
| `web/index.html` | web UI |
| `tests/test_jarvis.py` | offline tests |
| `README.md` | user-facing docs |
| `README_AI.md` | this file |

## 7. History of decisions

- The user first asked for a JARVIS-like assistant, with voice, command execution, and learning of any domain. Platforms: web and desktop.
- Brain choice: an LLM-style tool-using agent with a knowledge base, not the spiking model. Reason: the spiking model cannot serve as an assistant core.
- Later the user rejected the API key requirement and asked for a standalone version. Decision: offline by default, with the OpenAI backend as an opt-in only.
- The brain project was paused by the user. Its emotion regions (amygdala, orbitofrontal, insula, anterior cingulate) were removed on purpose. Do not add emotion back.
