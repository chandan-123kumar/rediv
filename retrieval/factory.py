"""Construct the configured retriever once per application process."""

import os
from pathlib import Path

from retrieval.bm25 import BM25Retriever
from retrieval.hybrid import HybridRetriever


def build_retriever(records: list[dict[str, str]], root: Path):
    if os.getenv("OPENAI_API_KEY", "").strip():
        try:
            return HybridRetriever(records, root), "hybrid_qdrant_bm25"
        except Exception:
            if os.getenv("RETRIEVAL_FALLBACK_ON_ERROR", "false").lower() != "true":
                raise
            return BM25Retriever(records), "bm25"
    return BM25Retriever(records), "bm25"
