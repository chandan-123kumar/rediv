"""Run the offline evaluation cases against a running local FastAPI server."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent


async def _asgi_post(app, question: str) -> tuple[int, dict]:
    payload = json.dumps({"question": question}).encode("utf-8")
    incoming = False
    sent = []

    async def receive():
        nonlocal incoming
        if not incoming:
            incoming = True
            return {"type": "http.request", "body": payload, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1", "method": "POST", "scheme": "http", "path": "/answer",
        "raw_path": b"/answer", "query_string": b"", "root_path": "",
        "headers": [(b"content-type", b"application/json"),
                    (b"content-length", str(len(payload)).encode("ascii"))],
        "client": ("testclient", 50000), "server": ("testserver", 80), "state": {},
    }
    await app(scope, receive, send)
    start = next(message for message in sent if message["type"] == "http.response.start")
    body = b"".join(message.get("body", b"") for message in sent if message["type"] == "http.response.body")
    return start["status"], json.loads(body.decode("utf-8"))


async def _evaluate_asgi(cases: list[dict]) -> list[dict]:
    from application.api import app

    results = []
    async with app.router.lifespan_context(app):
        for case in cases:
            status, body = await _asgi_post(app, case["question"])
            results.append((status, body))
    return results


def _check_case(case: dict, status: int, body: dict) -> dict:
    errors = []
    if status != 200:
        errors.append(f"unexpected HTTP status {status}")
    for field in ("question_id", "answer", "citations", "insufficient_context"):
        if field not in body:
            errors.append(f"missing response field: {field}")
    if body.get("insufficient_context") is not case["expected_insufficient_context"]:
        errors.append("insufficient_context did not match expectation")
    expected_article = case["expected_article_id"]
    if expected_article and expected_article not in body.get("citations", []):
        errors.append(f"expected citation {expected_article} was absent")
    if not expected_article and body.get("citations"):
        errors.append("unsupported question returned citations")
    answer = str(body.get("answer", "")).lower()
    for phrase in case["expected_answer_contains"]:
        if phrase.lower() not in answer:
            errors.append(f"answer did not contain {phrase!r}")
    return {"case_id": case["case_id"], "passed": not errors, "errors": errors,
            "question_id": body.get("question_id"), "citations": body.get("citations", []),
            "insufficient_context": body.get("insufficient_context")}


def evaluate(base_url: str, in_process: bool = False) -> dict:
    cases = json.loads((ROOT / "eval_cases.json").read_text(encoding="utf-8"))
    results = []
    if in_process:
        responses = asyncio.run(_evaluate_asgi(cases))
        results = [_check_case(case, status, body) for case, (status, body) in zip(cases, responses)]
    else:
      for case in cases:
        try:
            request = Request(
                f"{base_url.rstrip('/')}/answer",
                data=json.dumps({"question": case["question"]}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=30) as response:
                body = json.loads(response.read().decode("utf-8"))
                status = response.status
            results.append(_check_case(case, status, body))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            results.append({"case_id": case["case_id"], "passed": False, "errors": [str(exc)]})

    summary = {"passed": all(result["passed"] for result in results),
               "passed_count": sum(result["passed"] for result in results),
               "case_count": len(results), "results": results}
    output = ROOT / "artifacts" / "api_eval_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Local FastAPI base URL")
    parser.add_argument("--in-process", action="store_true", help="Call FastAPI via ASGI without opening a socket")
    args = parser.parse_args()
    summary = evaluate(args.base_url, in_process=args.in_process)
    print(f"API evaluation: {summary['passed_count']}/{summary['case_count']} cases passed.")
    if not summary["passed"]:
        print(json.dumps(summary["results"], indent=2, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
