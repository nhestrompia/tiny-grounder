"""A fake OpenAI-compatible teacher endpoint for dry runs of the GPU scripts (no model, no GPU).

    uv run python -m tools.fake_teacher --port 8000

Answers /v1/models and /v1/chat/completions. Replies follow the teacher's output format, with
templated instructions derived from the element lines of the prompt, plus a `usage` block.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

_LINE_RE = re.compile(r'^\[(\d+)\] role=(\S+) text=("(?:[^"\\]|\\.)*") context=("(?:[^"\\]|\\.)*") .* same_text_elsewhere=(\d+)$')
_ADJECTIVES = ["quick", "calm", "bright", "plain", "bold", "tiny", "warm", "cool", "sharp", "soft", "dry", "wet", "odd"]
_NOUNS = ["route", "gate", "lever", "panel", "switch", "portal", "dial", "key", "latch", "door", "tool", "handle", "knob"]


def _tag(*parts: str) -> str:
    h = int(hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest(), 16)
    return f"{_ADJECTIVES[h % 13]} {_NOUNS[(h // 13) % 13]}"


def reply_items(user_prompt: str) -> dict[str, Any]:
    """The `{"items": [...]}` a teacher would return for the element lines of `user_prompt`."""
    items = []
    for line in user_prompt.splitlines():
        match = _LINE_RE.match(line)
        if not match:
            continue
        index, role, text, context, shared = int(match[1]), match[2], json.loads(match[3]), json.loads(match[4]), int(match[5])
        tag = _tag(text, context, str(index))
        items.append({
            "i": index,
            "plain": f"click {text}" if text else f"click the {role}",
            "paraphrase": f"use the {tag} control",
            "context": f"click {text} for {context.split(' · ')[0]}" if shared and text and context else None,
            "goal": f"finish the {tag} errand",
        })
    return {"items": items}


def completion(request: dict[str, Any]) -> dict[str, Any]:
    prompt = request["messages"][-1]["content"]
    content = json.dumps(reply_items(prompt))
    usage = {"prompt_tokens": sum(len(m["content"]) for m in request["messages"]) // 4, "completion_tokens": len(content) // 4}
    return {"choices": [{"message": {"role": "assistant", "content": content}}], "usage": usage}


class Handler(BaseHTTPRequestHandler):
    def _send(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/v1/models":
            self._send({"data": [{"id": "teacher"}]})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self._send({"error": "not found"}, 404)
            return
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self._send(completion(request))

    def log_message(self, format: str, *args: Any) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
