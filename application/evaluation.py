"""Sample-question evaluation summary."""

from typing import Any


def evaluate_answers(answers: list[dict[str, Any]]) -> dict[str, Any]:
    expected = {"Q1": "ART-101", "Q2": "ART-102", "Q3": "ART-301", "Q4": "ART-302"}
    by_id = {answer["question_id"]: answer for answer in answers}
    checks = [{"question_id": question_id,
               "passed": source in by_id.get(question_id, {}).get("citations", [])
                         and not by_id.get(question_id, {}).get("insufficient_context", True),
               "expected_article_id": source} for question_id, source in expected.items()]
    unsupported = by_id.get("Q5", {})
    checks.append({"question_id": "Q5", "passed": unsupported.get("insufficient_context") is True,
                   "expected_insufficient_context": True})
    return {"passed": all(check["passed"] for check in checks), "checks": checks}
