"""Knowledge-base loading and deterministic chunk creation."""

import json
from pathlib import Path


def load_articles(path: Path) -> list[dict[str, str]]:
    articles = json.loads(path.read_text(encoding="utf-8"))
    required = {"id", "title", "product", "content"}
    if not isinstance(articles, list):
        raise ValueError("Knowledge base must be a JSON array")
    seen = set()
    for article in articles:
        if not isinstance(article, dict) or not required.issubset(article):
            raise ValueError("Each knowledge-base article needs id, title, product, and content")
        if article["id"] in seen:
            raise ValueError(f"Duplicate article ID: {article['id']}")
        seen.add(article["id"])
    return articles


def chunk_articles(articles: list[dict[str, str]], max_words: int = 90) -> list[dict[str, str]]:
    records = []
    for article in articles:
        words = article["content"].split()
        pieces = [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)] or [""]
        for index, text in enumerate(pieces, start=1):
            records.append({"chunk_id": f"{article['id']}-C{index}", "article_id": article["id"],
                            "title": article["title"], "product": article["product"], "text": text})
    return records
