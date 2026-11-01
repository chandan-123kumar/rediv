"""Dependency-free BM25 retrieval."""

import math
import re
from collections import Counter
from typing import Any

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "can", "do", "does", "for", "from",
    "how", "i", "if", "in", "is", "it", "me", "my", "of", "on", "or", "should", "the",
    "their", "them", "there", "this", "to", "use", "what", "when", "which", "with", "you", "s",
    "email", "emails", "day", "days", "hour", "hours", "business", "time", "usually", "within",
}
PLURAL_MAP = {"webhooks": "webhook", "withdrawals": "withdrawal", "clients": "client"}


def tokenize(text: str) -> list[str]:
    return [PLURAL_MAP.get(token, token) for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS]


class BM25Retriever:
    def __init__(self, records: list[dict[str, str]], k1: float = 1.5, b: float = 0.75):
        self.records = records
        self.documents = [tokenize(f"{r['title']} {r['product']} {r['text']}") for r in records]
        self.k1, self.b = k1, b
        self.avgdl = sum(map(len, self.documents)) / max(len(self.documents), 1)
        self.df = Counter(term for doc in self.documents for term in set(doc))

    def search(self, question: str, limit: int = 3) -> list[dict[str, Any]]:
        query = tokenize(question)
        total = len(self.documents)
        ranked = []
        for record, doc in zip(self.records, self.documents):
            counts = Counter(doc)
            score = 0.0
            for term in query:
                if term not in counts:
                    continue
                idf = math.log(1 + (total - self.df[term] + 0.5) / (self.df[term] + 0.5))
                tf = counts[term]
                denom = tf + self.k1 * (1 - self.b + self.b * len(doc) / max(self.avgdl, 1))
                score += idf * tf * (self.k1 + 1) / denom
            ranked.append((record, score))
        ranked.sort(key=lambda pair: (-pair[1], pair[0]["chunk_id"]))
        return [{"chunk_id": r["chunk_id"], "article_id": r["article_id"], "score": round(score, 6)}
                for r, score in ranked[:limit]]
