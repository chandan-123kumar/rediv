"""Validate required inputs and generated pipeline artifacts."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED_INPUTS = [ROOT / "kb/articles.json", ROOT / "questions.json"]
REQUIRED_ARTIFACTS = [ROOT / f"artifacts/{name}" for name in
                      ("index_preview.json", "retrieval_results.json", "answers.json", "eval_summary.json")]


def main() -> None:
    for path in REQUIRED_INPUTS + REQUIRED_ARTIFACTS:
        if not path.is_file():
            raise SystemExit(f"Missing required file: {path.relative_to(ROOT)}")
    articles = json.loads((ROOT / "kb/articles.json").read_text(encoding="utf-8"))
    article_ids = {article["id"] for article in articles}
    answers = json.loads((ROOT / "artifacts/answers.json").read_text(encoding="utf-8"))
    for answer in answers:
        required = {"question_id", "answer", "citations", "insufficient_context"}
        if not required.issubset(answer):
            raise SystemExit(f"Answer missing fields: {answer}")
        if not isinstance(answer["answer"], str) or not isinstance(answer["citations"], list):
            raise SystemExit(f"Answer has invalid field types: {answer}")
        if len(answer["citations"]) != len(set(answer["citations"])):
            raise SystemExit(f"Duplicate citations for {answer['question_id']}")
        if not set(answer["citations"]).issubset(article_ids):
            raise SystemExit(f"Unknown citation for {answer['question_id']}")
    q5 = next((answer for answer in answers if answer["question_id"] == "Q5"), None)
    if q5 is None or q5["insufficient_context"] is not True:
        raise SystemExit("Q5 must have insufficient_context=true")
    for name in ("index_preview.json", "retrieval_results.json", "eval_summary.json"):
        json.loads((ROOT / "artifacts" / name).read_text(encoding="utf-8"))
    print("Validation passed.")


if __name__ == "__main__":
    main()
