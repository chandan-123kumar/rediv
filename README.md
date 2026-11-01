# Local AI Support Assistant

Small, grounded support Q&A pipeline using the sample local knowledge base.

## Set up with uv

```bash
uv sync
```

This creates the project virtual environment in `.venv`. The default environment includes FastAPI, Hypercorn, and the OpenAI client; BM25 mode still needs no API key. Install the optional vector database dependency for hybrid mode:

```bash
uv sync --extra hybrid
```

## Run

```bash
uv run python main.py
uv run python validate.py
```

The default path uses BM25 and needs no API key. To enable hybrid BM25 + OpenAI embeddings/Qdrant retrieval, copy `.env.example` to `.env`, set `OPENAI_API_KEY`, and sync with `--extra hybrid`. The `.env` file is ignored by Git. `requirements.txt` remains as a pip compatibility option; `pyproject.toml` is the primary dependency configuration.

## Local API

Start the FastAPI application on localhost:

```bash
uv run hypercorn application.api:app --bind 127.0.0.1:8000
```

Send `POST /answer` with `{"question":"Are webhooks delivered exactly once?"}`. The API and batch CLI use the same application service.

Evaluate the running API against paraphrases and unsupported questions:

```bash
uv run python evaluate_api.py
```

This writes `artifacts/api_eval_summary.json`.
If the environment does not allow a local listening socket, run the same FastAPI routes in-process:

```bash
RETRIEVAL_FALLBACK_ON_ERROR=true uv run python evaluate_api.py --in-process
```

## Artifacts

The pipeline generates retrieval records, top-three retrieval results, structured answers, an evaluation summary, and a JSONL run log under `artifacts/`.

## Assumptions and tradeoffs

- OpenAI is used only to embed the question and knowledge chunks; answer wording is deterministic and grounded in retrieved text.
- Qdrant runs locally in embedded mode under `artifacts/qdrant/`; the index is reused when the knowledge base and embedding model are unchanged.
- With `OPENAI_API_KEY`, BM25 and Qdrant candidates are fused with RRF. Without it, retrieval uses deterministic BM25 only.
- Article JSON remains the source of truth. Local vectors and index files are generated data.
- Unsupported questions receive a safe escalation message instead of a guessed answer.
