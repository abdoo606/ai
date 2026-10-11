# Jarvis

A personal assistant with tools, memory and web learning. It runs in a browser (with voice input and spoken replies) or in a terminal.

It is standalone: the default needs no API key and no external model. Everything runs locally, except web search and page reads, which need internet access. For real local language-model reasoning, an optional **Hugging Face Transformers** backend runs open models on the user's own CPU/GPU, also without an API key.

## Features

- **Autonomous deep learning**: `learn <topic>` / `اتعلم <موضوع>` creates a broad curriculum (fundamentals, branches, history, methods, applications, advanced work, current research, and open questions), searches every part, reads and deduplicates sources, discovers more subtopics, then archives the material locally with its URLs. Each pass is bounded and can be repeated to refresh or extend knowledge.
- **Connected reasoning**: every learned chunk updates a local concept graph. Questions use graph-based query expansion plus reciprocal-rank fusion to connect evidence across sources, then maximal-marginal-relevance selects useful, non-repetitive sentences. Reported links are labelled as co-occurrence unless a source proves causation.
- **Fast recall**: SQLite FTS5/BM25, indexed graph edges, a 32 MiB page cache, WAL mode, and cached query expansions keep retrieval lightweight without an embedding server.
- **Grounded answers**: local extractive answering cites every source, compares connected evidence, and does not invent unsupported text.
- **Arabic retrieval**: Arabic diacritics, alef variants, and ya/alef-maqsura variants are normalized in the search index while original source text is preserved.
- **Local open model (optional)**: Hugging Face `transformers` adds conversational reasoning and tool use without sending prompts to a service.
- **Web search and reading**: DuckDuckGo search, and readable text from any http(s) page.
- **Files**: read, write and list files, **only inside the workspace** (`~/jarvis_workspace` by default). Paths that escape it are refused.
- **Commands**: shell commands are **off by default**. Enable them with `JARVIS_ALLOW_SHELL=1`.
- **Interfaces**: web page (`/`) with a chat, a learn panel, and microphone and speaker buttons (Chrome or Edge); terminal mode (`python -m jarvis`).

## Setup

```bash
pip install -r jarvis/requirements.txt
```

Python 3.10 or newer. SQLite with FTS5 is included in standard Python builds.

### Optional: local Hugging Face brain

```bash
pip install -r jarvis/requirements-huggingface.txt

# One-time model download (no API key). This can be several GB.
JARVIS_BACKEND=huggingface JARVIS_HF_ALLOW_DOWNLOAD=1 python -m jarvis "Hello"

# Later runs can be forced to use only the cached/local model.
JARVIS_BACKEND=huggingface python -m jarvis
```

For a fully controlled installation, download a compatible instruct model into
`jarvis/models/<name>` and set `JARVIS_HF_MODEL` to that folder. Jarvis loads
Transformers lazily and never downloads weights unless
`JARVIS_HF_ALLOW_DOWNLOAD=1` is explicitly set.

### Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `JARVIS_BACKEND` | `offline` | `offline` uses the built-in brain; `huggingface` runs an open model locally. There is no hosted/API backend. |
| `JARVIS_HF_MODEL` | `jarvis/models/brain` | Local model folder. A Hub model ID is accepted only when the one-time download flag is enabled. |
| `JARVIS_HF_ALLOW_DOWNLOAD` | `0` | Model downloads are blocked by default. Set `1` only for the initial model download. |
| `JARVIS_HF_DEVICE` | `auto` | `auto`, `cpu`, `mps`, or a CUDA device index such as `0`. |
| `JARVIS_HF_MAX_NEW_TOKENS` | `512` | Maximum generated tokens per local-model step. |
| `JARVIS_HOME` | `jarvis/data` | Where the knowledge base is stored. Git-ignored. |
| `JARVIS_WORKSPACE` | `~/jarvis_workspace` | The only folder the file tools can touch. |
| `JARVIS_ALLOW_SHELL` | `0` | Set to `1` to let the assistant run shell commands in the workspace. |

## Run

Web (listens on all interfaces, port 8000):

```bash
uvicorn jarvis.server:get_app --factory --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000`.

Terminal:

```bash
python -m jarvis                  # interactive
python -m jarvis "learn qubits"   # deep-learning pass
python -m jarvis "اربط الطاقة والكتلة"  # inspect evidence-backed links
```

Databases created before the concept-graph upgrade can be indexed once from
chat with `rebuild connections` through the tool interface, or by calling the
`rebuild_connections` tool. New sources are indexed automatically.

## Local web endpoints

| Method | Path | Body / query | Returns |
|---|---|---|---|
| GET | `/api/status` | | LLM mode, topics, shell flag |
| POST | `/api/chat` | `{"message": "..."}` | `{"reply": "..."}` |
| POST | `/api/reset` | | clears the conversation |
| POST | `/api/learn` | `{"topic": "...", "max_sources": 24, "depth": 2}` | deep-research result |
| GET | `/api/recall` | `?q=...` | `{"result": "..."}` |

## Design

- `knowledge.py`: SQLite FTS5/BM25 plus an indexed weighted concept graph. Connected search uses graph expansion and reciprocal-rank fusion; answer selection uses maximal marginal relevance and source citations. Chunks are about 800 characters.
- `research.py`: autonomous curriculum generation, bounded multi-pass research, URL/content deduplication, subtopic discovery, and local archiving.
- `web.py`: standard-library HTTP, HTML-to-text, and DuckDuckGo result parsing.
- `tools.py`: tools with JSON schemas. Output is capped at 4000 characters. Errors are returned as text, not raised.
- `llm.py`: the zero-dependency offline router and an optional lazy-loaded local Hugging Face Transformers brain with JSON tool calling. No hosted-model client exists.
- `agent.py`: the assistant loop. It calls tools for up to 6 steps, then answers, and keeps the last 40 messages of history.
- `server.py`: FastAPI. The UI uses relative URLs, so it works behind a preview proxy.

## Tests

```bash
pytest -q jarvis/tests
```

The tests use fixtures for web pages and scripted models. They need no network and no API key.

## Security notes

- Files: confined to the workspace (checked after path resolution, so `..` and absolute paths are refused).
- Shell: disabled unless `JARVIS_ALLOW_SHELL=1`. Only enable it on a machine you trust, because the model can run any command that the account can.
- Web fetch: only `http` and `https`. Pages are capped at 2 MB.
- No model-service credentials are accepted or stored. The knowledge base, local model weights, and workspace are git-ignored.

## Limits

- The default offline mode can route commands and answer from stored sources extractively, but it is not a generative conversational model. Use the local Hugging Face backend for that.
- Deep learning needs public websites while it is researching; answering from already archived memory works offline. Search providers and source sites can change or rate-limit.
- No finite system can literally download “everything in the world.” Jarvis uses explicit source/depth budgets, reports what it covered, and can run more passes.
- Learned text is stored as-is. Check important answers against the cited sources.
- Not yet built: a desktop GUI, device and smart-home control, and live voice in the terminal.

See `README_AI.md` for the developer handoff (state, decisions, next steps).
