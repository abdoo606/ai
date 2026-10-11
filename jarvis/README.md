# Jarvis

A personal assistant with tools, memory and web learning. It runs in a browser (with voice input and spoken replies) or in a terminal.

It is standalone: the default needs no API key and no external model. Everything runs locally, except web search and page reads, which need internet access. For real local language-model reasoning, an optional **Hugging Face Transformers** backend runs open models on the user's own CPU/GPU, also without an API key.

## Features

- **Learn any topic**: `learn <topic>` searches the web, reads the top pages, and stores them in a local SQLite database with full-text (BM25) indexing. Each chunk keeps its source URL.
- **Recall**: local extractive answering selects the most relevant sentences and cites every source; it does not invent unsupported text.
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
| `JARVIS_BACKEND` | `offline` | `offline` needs nothing; `huggingface` runs a local open model; `openai` is optional. |
| `JARVIS_HF_MODEL` | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | Local folder or Hugging Face model ID used by the local backend. |
| `JARVIS_HF_ALLOW_DOWNLOAD` | `0` | Model downloads are blocked by default. Set `1` only for the initial model download. |
| `JARVIS_HF_DEVICE` | `auto` | `auto`, `cpu`, `mps`, or a CUDA device index such as `0`. |
| `JARVIS_HF_MAX_NEW_TOKENS` | `512` | Maximum generated tokens per local-model step. |
| `JARVIS_LLM_API_KEY` | (empty) | Used only with `JARVIS_BACKEND=openai`. Set it in your environment, never in the code. |
| `JARVIS_LLM_BASE_URL` | `https://api.openai.com/v1` | Any OpenAI-compatible endpoint (OpenRouter, a local server, ...). |
| `JARVIS_LLM_MODEL` | `gpt-4o-mini` | Model name. |
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
python -m jarvis "learn qubits"   # one question
```

## API

| Method | Path | Body / query | Returns |
|---|---|---|---|
| GET | `/api/status` | | LLM mode, topics, shell flag |
| POST | `/api/chat` | `{"message": "..."}` | `{"reply": "..."}` |
| POST | `/api/reset` | | clears the conversation |
| POST | `/api/learn` | `{"topic": "...", "sources": 3}` | `{"result": "..."}` |
| GET | `/api/recall` | `?q=...` | `{"result": "..."}` |

## Design

- `knowledge.py`: SQLite + FTS5 store (`topics`, `sources`, `chunks`), normalized multilingual indexing, BM25 retrieval, and citation-preserving extractive answers. Chunks are about 800 characters.
- `web.py`: standard-library HTTP, HTML-to-text, and DuckDuckGo result parsing.
- `tools.py`: tools with JSON schemas. Output is capped at 4000 characters. Errors are returned as text, not raised.
- `llm.py`: the zero-dependency offline router, an optional lazy-loaded Hugging Face Transformers backend with JSON tool calling, and an optional OpenAI-compatible client.
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
- Secrets: read only from the environment. The knowledge base and workspace are git-ignored.

## Limits

- The default offline mode can route commands and answer from stored sources extractively, but it is not a generative conversational model. Use the local Hugging Face backend for that.
- The web search depends on DuckDuckGo's HTML endpoint. It can change or rate-limit.
- Learned text is stored as-is. Check important answers against the cited sources.
- Not yet built: a desktop GUI, device and smart-home control, and live voice in the terminal.

See `README_AI.md` for the developer handoff (state, decisions, next steps).
