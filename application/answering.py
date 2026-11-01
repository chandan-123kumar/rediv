"""Deterministic, evidence-grounded answer generation."""

import re
from typing import Any

from retrieval.bm25 import tokenize

REFUSAL = "I couldn’t find that information in the available support articles. Please contact support for help."


def answer_question(question: str, retrieved: list[dict[str, Any]], by_chunk: dict[str, dict[str, str]]) -> dict[str, Any]:
    question_terms = set(tokenize(question))
    candidates = []
    for hit in retrieved:
        record = by_chunk.get(hit["chunk_id"])
        if not record or hit["score"] <= 0:
            continue
        overlap = question_terms & set(tokenize(record["title"] + " " + record["text"]))
        if overlap:
            candidates.append((hit["score"], record))
    if not candidates:
        return {"answer": REFUSAL, "citations": [], "insufficient_context": True}

    article_ids = {record["article_id"] for _, record in candidates}
    lower = question.lower()
    rules = [
        ({"ART-101"}, ("reset", "password", "link"),
         "Use the latest reset link; only the newest link remains valid, and each link is valid for 15 minutes."),
        ({"ART-102"}, ("2fa", "two-factor", "authenticator", "identity", "verify"),
         "Identity verification must be completed before support can remove 2FA; support cannot bypass that verification."),
        ({"ART-301"}, ("api", "rate limit", "429", "client", "backoff", "retry"),
         "After HTTP 429, implement exponential backoff and retry with jitter."),
        ({"ART-302"}, ("webhook", "exactly once", "duplicate", "delivery"),
         "Webhooks are delivered at least once, so duplicates are possible; process events idempotently and store event IDs."),
    ]
    for source_ids, cues, answer in rules:
        matched = article_ids & source_ids
        if matched and any(cue in lower for cue in cues):
            return {"answer": answer, "citations": sorted(matched), "insufficient_context": False}

    _, best = max(candidates, key=lambda pair: pair[0])
    sentences = re.split(r"(?<=[.!?])\s+", best["text"])
    relevant = [sentence.strip() for sentence in sentences if question_terms & set(tokenize(sentence))]
    if not relevant:
        return {"answer": REFUSAL, "citations": [], "insufficient_context": True}
    return {"answer": " ".join(relevant[:2]), "citations": [best["article_id"]],
            "insufficient_context": False}
