"""Hybrid BM25 and local Qdrant retrieval with Reciprocal Rank Fusion."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from retrieval.bm25 import BM25Retriever


class HybridRetriever:
    def __init__(self, records: list[dict[str, str]], root: Path, candidate_limit: int = 10, rrf_k: int = 60):
        from openai import OpenAI
        from qdrant_client import QdrantClient, models

        self.records = records
        self.bm25 = BM25Retriever(records)
        self.candidate_limit = candidate_limit
        self.rrf_k = rrf_k
        self.model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.collection = "support_kb"
        path = Path(os.getenv("QDRANT_PATH", "artifacts/qdrant"))
        self.db_path = path if path.is_absolute() else root / path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.qdrant = QdrantClient(path=str(self.db_path))
        self.models = models
        try:
            self._ensure_index()
        except Exception:
            self.qdrant.close()
            raise

    @property
    def manifest_path(self) -> Path:
        return self.db_path / "support_kb_manifest.json"

    def _fingerprint(self) -> str:
        serialized = json.dumps(self.records, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(f"{self.model}\nchunk-v1\n{serialized}".encode("utf-8")).hexdigest()

    def _ensure_index(self) -> None:
        fingerprint = self._fingerprint()
        if self.qdrant.collection_exists(self.collection) and self.manifest_path.exists():
            try:
                manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                if manifest.get("fingerprint") == fingerprint:
                    return
            except (OSError, json.JSONDecodeError):
                pass

        if self.qdrant.collection_exists(self.collection):
            self.qdrant.delete_collection(self.collection)
        texts = [f"{r['title']} ({r['product']}). {r['text']}" for r in self.records]
        vectors = [item.embedding for item in self.openai.embeddings.create(model=self.model, input=texts).data]
        self.qdrant.create_collection(
            collection_name=self.collection,
            vectors_config=self.models.VectorParams(size=len(vectors[0]), distance=self.models.Distance.COSINE),
        )
        self.qdrant.upsert(collection_name=self.collection, points=[
            self.models.PointStruct(id=i, vector=vector, payload=dict(record))
            for i, (record, vector) in enumerate(zip(self.records, vectors))
        ])
        self.manifest_path.write_text(
            json.dumps({"fingerprint": fingerprint, "embedding_model": self.model}, indent=2) + "\n",
            encoding="utf-8",
        )

    def search(self, question: str, limit: int = 3) -> list[dict[str, Any]]:
        pool = max(limit, self.candidate_limit)
        lexical = self.bm25.search(question, pool)
        vector = self.openai.embeddings.create(model=self.model, input=[question]).data[0].embedding
        semantic = self.qdrant.query_points(collection_name=self.collection, query=vector, limit=pool).points
        dense = [{"chunk_id": point.payload["chunk_id"], "article_id": point.payload["article_id"]}
                 for point in semantic]

        fused: dict[str, dict[str, Any]] = {}
        for rank, hit in enumerate(lexical, start=1):
            if hit["score"] <= 0:
                continue
            item = fused.setdefault(hit["chunk_id"], {"chunk_id": hit["chunk_id"],
                "article_id": hit["article_id"], "score": 0.0, "bm25_rank": rank})
            item["score"] += 1 / (self.rrf_k + rank)
        for rank, hit in enumerate(dense, start=1):
            item = fused.setdefault(hit["chunk_id"], {"chunk_id": hit["chunk_id"],
                "article_id": hit["article_id"], "score": 0.0, "bm25_rank": float("inf")})
            item["score"] += 1 / (self.rrf_k + rank)

        ranked = sorted(fused.values(), key=lambda item: (-item["score"], item["bm25_rank"], item["chunk_id"]))
        return [{"chunk_id": item["chunk_id"], "article_id": item["article_id"],
                 "score": round(item["score"], 8)} for item in ranked[:limit]]

    def close(self) -> None:
        self.qdrant.close()
