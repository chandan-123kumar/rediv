"""Shared application service for batch and HTTP requests."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from application.artifacts import write_json, write_jsonl
from application.answering import answer_question
from application.evaluation import evaluate_answers
from config import ROOT, load_dotenv
from retrieval.chunking import chunk_articles, load_articles
from retrieval.factory import build_retriever


class SupportService:
    def __init__(self, root: Path = ROOT):
        self.root = root
        self.articles = load_articles(root / "kb" / "articles.json")
        self.chunks = chunk_articles(self.articles)
        self.by_chunk = {chunk["chunk_id"]: chunk for chunk in self.chunks}
        self.retriever, self.retrieval_mode = build_retriever(self.chunks, root)

    def answer(self, question: str, question_id: str | None = None) -> dict[str, Any]:
        hits = self.retriever.search(question, limit=3)
        result = answer_question(question, hits, self.by_chunk)
        return {"question_id": question_id or str(uuid.uuid4()), **result, "retrieved_chunks": hits}

    def close(self) -> None:
        close = getattr(self.retriever, "close", None)
        if close:
            close()


def create_service(root: Path = ROOT) -> SupportService:
    load_dotenv(root / ".env")
    return SupportService(root)


def run_batch(root: Path = ROOT) -> None:
    service = create_service(root)
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    questions = json.loads((root / "questions.json").read_text(encoding="utf-8"))
    write_json(artifacts / "index_preview.json", service.chunks)

    retrieval_results, answers, logs = [], [], []
    for question in questions:
        result = service.answer(question["question"], question["id"])
        hits = result.pop("retrieved_chunks")
        retrieval_results.append({"question_id": question["id"], "question": question["question"],
                                  "retrieved_chunks": hits})
        answers.append(result)
        logs.append({"question_id": question["id"], "retrieval_chunk_ids": [hit["chunk_id"] for hit in hits],
                     "retrieval_mode": service.retrieval_mode, "used_fallback": service.retrieval_mode == "bm25",
                     "timestamp": datetime.now(timezone.utc).isoformat()})

    write_json(artifacts / "retrieval_results.json", retrieval_results)
    write_json(artifacts / "answers.json", answers)
    write_jsonl(artifacts / "run_log.jsonl", logs)
    write_json(artifacts / "eval_summary.json", evaluate_answers(answers))
    print(f"Generated artifacts for {len(questions)} questions using {service.retrieval_mode} retrieval.")
    service.close()
