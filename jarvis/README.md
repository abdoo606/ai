# Jarvis

A personal assistant with tools, memory and web learning. It runs in a browser (with voice input and spoken replies) or in a terminal.

It is standalone: the default needs no API key and no external model. Everything runs locally, except web search and page reads, which need internet access. For real local language-model reasoning, an optional **Hugging Face Transformers** backend runs open models on the user's own CPU/GPU, also without an API key.

## Features

- **Autonomous deep learning**: `learn <topic>` / `اتعلم <موضوع>` creates a broad curriculum (fundamentals, branches, history, methods, applications, advanced work, current research, and open questions), searches every part, reads and deduplicates sources, discovers more subtopics, then archives the material locally with its URLs. Each pass is bounded and can be repeated to refresh or extend knowledge.
- **Connected reasoning**: every learned chunk updates a local concept graph. Questions use graph-based query expansion plus reciprocal-rank fusion to connect evidence across sources, then maximal-marginal-relevance selects useful, non-repetitive sentences. Reported links are labelled as co-occurrence unless a source proves causation.
- **Fast recall**: SQLite FTS5/BM25, indexed graph edges, a 32 MiB page cache, WAL mode, and cached query expansions keep retrieval lightweight without an embedding server.
- **Grounded answers**: local extractive answering cites every source, compares connected evidence, and does not invent unsupported text.
- **Arabic retrieval**: Arabic diacritics, alef variants, and ya/alef-maqsura variants are normalized in the search index while original source text is preserved.
- **Powerful local deliberation (optional)**: Jarvis runs an intent pass, senior reasoning pass, adversarial verification pass, and final Arabic/English revision. The reasoner, critic, and language writers are separate logical brains that share one strong physical model in memory.
- **Hardware profiles**: `compact` (Qwen3-4B), `balanced` (8B), `powerful` (14B), `max` (32B), and server-class `frontier` (Qwen3-235B-A22B). Larger profiles improve capability but need dramatically more RAM/GPU memory.
- **Robust intent**: a smaller multilingual model plus a fast fuzzy fallback recovers meaning from incomplete, dialectal, abbreviated, or misspelled Arabic/English commands.
- **Web search and reading**: DuckDuckGo search, and readable text from any http(s) page.
- **Files**: read, write and list files, **only inside the workspace** (`~/jarvis_workspace` by default). Paths that escape it are refused.
- **Commands**: disabled by default. Enabling requires both `JARVIS_ALLOW_SHELL=1` and an executable allowlist; commands run with `shell=False` inside the workspace.
- **Interfaces**: web page (`/`) with a chat, a learn panel, and microphone and speaker buttons (Chrome or Edge); terminal mode (`python -m jarvis`).

## Setup

```bash
# Source checkout
pip install -r jarvis/requirements.txt

# Or install the packaged application and console commands
pip install .
jarvis-local "time"
jarvis-local-web                 # binds safely to 127.0.0.1 by default
```

Python 3.10 or newer. SQLite with FTS5 is included in standard Python builds.

### Optional: powerful local reasoning

```bash
pip install -r jarvis/requirements-huggingface.txt

# Preview model sizes/paths without downloading.
python jarvis/scripts/install_models.py --profile powerful --dry-run

# One-time public download; no account or API key. The default powerful profile
# uses Qwen3-14B for shared reasoning/writing and Qwen3-1.7B for intent.
python jarvis/scripts/install_models.py --profile powerful

# Later runs are fully local and use a deliberate reason → critique → revise loop.
JARVIS_BACKEND=powerful python -m jarvis
```

Profiles: `compact` uses Qwen3-4B, `balanced` uses Qwen3-8B, `powerful` uses
Qwen3-14B, `max` uses Qwen3-32B, and `frontier` uses Qwen3-235B-A22B for
multi-GPU/server hardware. The strong model is loaded once and shared
by senior reasoner, critic, Arabic writer, and English writer role prompts. This
avoids four copies in memory. The installer resolves and records immutable model
commit revisions for reproducible/supply-chain-aware releases.
`JARVIS_REASONING_ROUNDS=1..3` trades latency for
additional critique/revision passes. The single-model and simple ensemble modes
remain available as `huggingface` and `ensemble`.

### Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `JARVIS_BACKEND` | `offline` | `powerful` enables deliberate local reasoning; `ensemble`, `huggingface`, and `offline` are lighter modes. No hosted/API backend. |
| `JARVIS_HF_REASONING_MODEL` | `jarvis/models/reasoning` | Senior reasoning model folder. |
| `JARVIS_HF_CRITIC_MODEL` | `jarvis/models/reasoning` | Verifier model folder; shared by default to avoid duplicate weights. |
| `JARVIS_HF_ENGLISH_MODEL` | `jarvis/models/reasoning` | English writer model folder. |
| `JARVIS_HF_ARABIC_MODEL` | `jarvis/models/reasoning` | Arabic writer model folder. |
| `JARVIS_HF_INTENT_MODEL` | `jarvis/models/intent` | Fast multilingual intent model folder. |
| `JARVIS_HF_MODEL` | `jarvis/models/brain` | Folder for legacy single-model mode. |
| `JARVIS_HF_ALLOW_DOWNLOAD` | `0` | Runtime downloads are blocked by default; use the explicit installer. |
| `JARVIS_HF_REVISION` | empty | Immutable commit SHA required if runtime download is deliberately enabled. |
| `JARVIS_HF_DEVICE` | `auto` | `auto`, `cpu`, `mps`, or a CUDA device index such as `0`. |
| `JARVIS_HF_MAX_NEW_TOKENS` | `1024` | Maximum tokens from reasoning/writer passes. |
| `JARVIS_HF_INTENT_MAX_NEW_TOKENS` | `256` | Maximum tokens from the intent pass. |
| `JARVIS_REASONING_ROUNDS` | `1` | Critique/revision rounds, clamped to 1–3. |
| `JARVIS_HOME` | `jarvis/data` | Where the knowledge base is stored. Git-ignored. |
| `JARVIS_WORKSPACE` | `~/jarvis_workspace` | The only folder the file tools can touch. |
| `JARVIS_ACCESS_TOKEN` | empty | Optional local web access token; required outside loopback deployments. |
| `JARVIS_RATE_LIMIT_PER_MINUTE` | `60` | Per-client limit for chat/research requests (1–600). |
| `JARVIS_ALLOW_PRIVATE_WEB` | `0` | Permit fetching local/private network URLs. Keep off for normal use. |
| `JARVIS_ALLOW_SHELL` | `0` | First switch required for command execution. |
| `JARVIS_ALLOWED_COMMANDS` | empty | Comma-separated executable allowlist, e.g. `python,git`; empty keeps commands unavailable. |

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
- `llm.py`: fuzzy routing, language detection, shared model loading, and a deliberate local pipeline: intent → senior reasoner/tool use → adversarial critic → Arabic/English final revision. No hosted-model client exists.
- `scripts/install_models.py`: explicit one-time installer with compact/balanced/powerful/max profiles and a local manifest.
- `agent.py`: the assistant loop. It calls tools for up to 6 steps, then answers, and keeps the last 40 messages of history.
- `server.py`: FastAPI. The UI uses relative URLs, so it works behind a preview proxy.

## Tests

```bash
pip install -r jarvis/requirements-dev.txt
pytest -q
python -m build
```

The suite covers memory/search/graph migrations, deep research, Arabic
normalization, malformed intents, deliberative orchestration, workspace and
command boundaries, SSRF protection, authentication, rate limits, validation,
API behavior, and packaging without network or model weights. GitHub Actions
runs it on Python 3.10–3.12.

## Security notes

- The packaged server binds to loopback. Set `JARVIS_ACCESS_TOKEN` plus TLS/reverse-proxy network controls before exposing it elsewhere.
- Files are confined to the workspace after path resolution.
- Commands require two explicit switches, an executable allowlist, run with `shell=False`, have a 60-second maximum, and use a restricted `PATH`.
- Web fetching blocks URL credentials, private/loopback/link-local/reserved targets, unsafe redirects, unsupported content types, and pages over 2 MB. Private fetches require explicit opt-in.
- API bodies and query lengths are bounded; expensive endpoints are rate-limited; security/no-store headers are emitted.
- No model-service credentials or telemetry exist. Memory, model weights, and workspaces are git-ignored.

See [SECURITY.md](SECURITY.md) for reporting/deployment guidance and
[RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) before commercial distribution.

## Limits

- A local 14B/32B model is not guaranteed to equal Claude's frontier-scale training, context, or reliability. The deliberate multi-pass pipeline closes part of the quality gap, but hardware and model size remain real limits.
- The `powerful` profile needs roughly 36 GB system RAM with unquantized Transformers weights; `max` needs roughly 80 GB. `frontier` is server-class and can require roughly 500 GB unquantized memory. Use `balanced`/`compact` on smaller machines or add a quantized model manually.
- The default offline mode can route commands and answer from stored sources extractively, but it is not a generative conversational model. Use `JARVIS_BACKEND=powerful` for the strongest local mode.
- Deep learning needs public websites while it is researching; answering from already archived memory works offline. Search providers and source sites can change or rate-limit.
- No finite system can literally download “everything in the world.” Jarvis uses explicit source/depth budgets, reports what it covered, and can run more passes.
- Learned text is stored as-is. Check important answers against the cited sources.
- Not yet built: a desktop GUI, device and smart-home control, and live voice in the terminal.

See `README_AI.md` for the developer handoff (state, decisions, next steps).
