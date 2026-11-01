# Support Assistant Specification

## Purpose

Provide a local support assistant that answers only from `kb/articles.json`, cites supporting article IDs, and refuses unsupported questions.

## Requirements

- Load and chunk the knowledge base in code; preserve article ID, title, and product metadata.
- Retrieve up to three relevant chunks for each question.
- With `OPENAI_API_KEY`, use OpenAI embeddings with local Qdrant and BM25 together; merge candidate rankings with Reciprocal Rank Fusion (RRF).
- Without a key, use BM25 only. The default path must work without network access or credentials.
- Generate concise, structured answers from retrieved context only. Return `insufficient_context: true` for unsupported questions; do not invent policies or details.
- Cite only supporting article IDs and remove duplicate citations.
- Provide a FastAPI `POST /answer` endpoint and a batch pipeline.
- Keep `main.py` as a thin CLI entry point; application and retrieval logic live in separate modules.
- Generate `index_preview.json`, `retrieval_results.json`, `answers.json`, `eval_summary.json`, and `run_log.jsonl` under `artifacts/`.
- Use `uv` for the project environment and dependency lock.

## API contract

Request:

```json
{"question":"Are webhooks delivered exactly once?"}
```

Response:

```json
{
  "question_id":"generated-id",
  "answer":"Webhooks are delivered at least once, so duplicates are possible.",
  "citations":["ART-302"],
  "insufficient_context":false
}
```

## Evaluation and acceptance

- Supported paraphrases for password reset, 2FA recovery, API limits, and webhook delivery cite the expected articles.
- Unsupported CEO-contact and refund-policy questions are refused with no citations.
- `python evaluate_api.py` checks a running API; `python evaluate_api.py --in-process` exercises the FastAPI app without a listening socket.
- `uv run python main.py` regenerates batch artifacts; `uv run python validate.py` validates them.
- `uv sync` creates `.venv`; `uv sync --extra hybrid` installs Qdrant for hybrid retrieval.
