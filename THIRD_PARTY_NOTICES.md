# Third-party notices and diligence map

This file is an engineering inventory, not a legal opinion. The buyer and seller
must verify the exact version, source, license text, notices, commercial-use
terms, export restrictions, and model acceptable-use terms included in the
final transfer build.

## Runtime Python dependencies

| Component | Expected license family | Role |
|---|---|---|
| FastAPI | MIT | Local web application |
| Uvicorn | BSD-3-Clause | Local ASGI server |
| Pydantic | MIT | Input validation |
| Starlette | BSD-3-Clause | Web framework dependency |

## Optional local-model dependencies

| Component | Expected license family | Role |
|---|---|---|
| Transformers | Apache-2.0 | Local model loading/inference |
| huggingface-hub | Apache-2.0 | One-time pinned model acquisition |
| safetensors | Apache-2.0 | Model weight format |
| PyTorch | BSD-style | Tensor runtime |

## Default model profiles

The installer currently identifies Qwen3 repositories for intent and reasoning
profiles. The installer resolves immutable revisions and writes them to
`zehnora/models/manifest.json`. Model weights are not part of this repository
or source-asset bundle. Before a transaction or bundled distribution:

1. Generate the final manifest with exact repository IDs and commit revisions.
2. Retrieve and archive each model card and license for those exact revisions.
3. Confirm commercial use, redistribution, trademark, attribution, and use-case
   restrictions with counsel.
4. State clearly whether weights are transferred, separately downloaded by the
   buyer, or excluded from the transaction.

## Web-derived runtime data

Zehnora can fetch and archive public pages at a user's direction. Copyright and
database rights in fetched content remain with their owners. Runtime knowledge
databases and fetched content are excluded from the default asset bundle unless
a definitive agreement explicitly identifies lawful transferable data.

## Removed unrelated project

The current tree no longer contains the Drosophila connectome source, notebooks,
data, outputs, environments, or downloader. Historical Git commits still exist
and must be reviewed or excluded from the buyer delivery package.
