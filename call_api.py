"""Send a sample question to the local FastAPI endpoint and print its response."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sample_question() -> str:
    questions = json.loads((ROOT / "questions.json").read_text(encoding="utf-8"))
    if not questions:
        raise ValueError("questions.json has no sample questions")
    return questions[0]["question"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default=None, help="Question to send (defaults to the first sample question)")
    parser.add_argument("--url", default="http://127.0.0.1:8000/answer", help="FastAPI answer endpoint")
    args = parser.parse_args()
    question = args.question or sample_question()

    command = [
        "curl", "--silent", "--show-error", "--fail-with-body",
        "--request", "POST", args.url,
        "--header", "Content-Type: application/json",
        "--data-binary", json.dumps({"question": question}),
    ]
    try:
        response = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
        payload = json.loads(response.stdout)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    except subprocess.CalledProcessError as exc:
        print(exc.stdout, end="", file=sys.stderr)
        print(exc.stderr, end="", file=sys.stderr)
        sys.exit(1)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"Could not call {args.url}: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
