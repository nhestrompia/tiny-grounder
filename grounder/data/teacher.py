"""LLM teacher: pages -> instruction examples through any OpenAI-compatible chat endpoint.

    uv run python -m grounder.data.teacher --pages data/pages/train --out data/teacher_train.jsonl \
        --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct --api-key-env VLLM_API_KEY

Sampled elements go to the model in batches (`BATCH_SIZE` per call). Each reply is cached on disk
under a hash of the request, so a rerun costs nothing and an interrupted run resumes where it stopped.
Replies are validated by `grounder.data.filters`; rejection counts are printed.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import httpx

from grounder.data.filters import filter_all, format_rejections, label_counts
from grounder.data.pages import iter_pages, write_examples
from grounder.data.template_teacher import SAMPLE_SIZE, sample_elements
from grounder.records import Element, Example, Page, equiv_ids

logger = logging.getLogger("grounder.teacher")

BATCH_SIZE = 10
N_NEIGHBOURS = 3
RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
SCHEMA_REJECTED_STATUSES = frozenset({400, 422})
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$")

SYSTEM_PROMPT = """\
You write training data for a model that finds the right UI element on a web page. For each numbered element, \
write what a user would say so that this element is the one to act on.

Return one JSON object: {"items": [{"i": <element number>, "plain": ..., "paraphrase": ..., "context": ..., "goal": ...}]}
- plain: a short natural command, 2-10 words. It may reuse the element's words.
- paraphrase: the same intent in different words. Do NOT use any word of the element's text (ignore "click", "button", "the").
- context: null unless the element's text is shared with other elements ("same_text_elsewhere" > 0). Then a command that \
names the element's text and singles it out by what its context says, never by position or number.
- goal: what the user wants to achieve, such that this element is the next step. Do not name the element.
Never mention element numbers, ids, coordinates, or words like "first", "second" or "third". Write every field for every element.

Example input:
Page: "Shoply - Home" (shoply.example)
[0] role=button text="Add to cart" context="Blue cotton shirt · $24" near=["Blue cotton shirt", "$24", "Wishlist"] same_text_elsewhere=2
[1] role=link text="Track order" context="" near=["Help", "Returns", "Sign in"] same_text_elsewhere=0
Example output:
{"items": [
{"i": 0, "plain": "click Add to cart for the blue cotton shirt", "paraphrase": "put the blue cotton shirt in my basket", \
"context": "add to cart for the blue cotton shirt", "goal": "buy the blue cotton shirt"},
{"i": 1, "plain": "open Track order", "paraphrase": "see where my parcel is", "context": null, \
"goal": "find out when my package will arrive"}]}"""

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "i": {"type": "integer"},
                    "plain": {"type": "string"},
                    "paraphrase": {"type": "string"},
                    "context": {"type": ["string", "null"]},
                    "goal": {"type": "string"},
                },
                "required": ["i", "plain", "paraphrase", "context", "goal"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}
RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {"name": "instructions", "strict": True, "schema": RESPONSE_SCHEMA}}

KIND_FIELDS = ("plain", "paraphrase", "context")


class BadResponse(Exception):
    """The model's reply is not the expected JSON."""


class TeacherError(Exception):
    """A request failed for good (non-retryable status, or retries exhausted)."""


# --- prompt -----------------------------------------------------------------------------------


def neighbour_texts(el: Element, elements: Sequence[Element], k: int = N_NEIGHBOURS) -> list[str]:
    """Texts of the `k` labelled elements whose box centres are nearest to `el`."""

    def centre(e: Element) -> tuple[float, float]:
        return e.box[0] + e.box[2] / 2, e.box[1] + e.box[3] / 2

    cx, cy = centre(el)
    others = [e for e in elements if e.id != el.id and e.text]
    others.sort(key=lambda e: (math.dist((cx, cy), centre(e)), e.id))
    return [e.text for e in others[:k]]


def element_line(index: int, el: Element, page: Page, counts: dict[str, int]) -> str:
    shared = counts[el.text.casefold()] - 1 if el.text else 0
    return (f"[{index}] role={el.role} text={json.dumps(el.text, ensure_ascii=False)} "
            f"context={json.dumps(el.context, ensure_ascii=False)} "
            f"near={json.dumps(neighbour_texts(el, page.elements), ensure_ascii=False)} same_text_elsewhere={shared}")


def build_messages(page: Page, batch: Sequence[Element]) -> list[dict[str, str]]:
    counts = label_counts(page.elements)
    host = re.sub(r"^https?://", "", page.url).split("/")[0]
    lines = [f'Page: "{page.title}" ({host})'] + [element_line(i, el, page, counts) for i, el in enumerate(batch)]
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "\n".join(lines)}]


# --- reply parsing ----------------------------------------------------------------------------


def parse_items(content: str) -> list[dict[str, Any]]:
    """The `items` of a reply; tolerates a markdown fence and a bare list."""
    try:
        data = json.loads(_FENCE_RE.sub("", content.strip()))
    except json.JSONDecodeError as e:
        raise BadResponse(f"not JSON: {e}") from e
    items = data.get("items") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise BadResponse("no items list")
    return [i for i in items if isinstance(i, dict)]


def items_to_examples(page: Page, batch: Sequence[Element], items: Sequence[dict[str, Any]]) -> list[Example]:
    """plain / paraphrase / context become mode "do" examples; goal becomes a mode "goal" example."""
    out: list[Example] = []
    for item in items:
        index = item.get("i")
        if not isinstance(index, int) or not 0 <= index < len(batch):
            continue
        el = batch[index]
        equiv = equiv_ids(page.elements, el.id)
        for kind in KIND_FIELDS:
            query = item.get(kind)
            if isinstance(query, str) and query.strip():
                out.append(Example(page.page_id, kind, "do", query.strip(), [], el.id, equiv))
        goal = item.get("goal")
        if isinstance(goal, str) and goal.strip():
            out.append(Example(page.page_id, "goal", "goal", goal.strip(), [], el.id, equiv))
    return out


# --- endpoint client --------------------------------------------------------------------------


@dataclass
class TeacherConfig:
    model: str
    cache_dir: Path
    seed: int = 0
    temperature: float = 0.7
    max_tokens: int = 2048
    retries: int = 4
    backoff_s: float = 1.0
    concurrency: int = 16
    per_page: int = SAMPLE_SIZE


class ChatClient:
    """Chat-completions client with a disk cache, retries and a json_schema -> plain JSON fallback."""

    def __init__(self, http: httpx.AsyncClient, cfg: TeacherConfig) -> None:
        self.http = http
        self.cfg = cfg
        self.use_schema = True
        cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    def _body(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.cfg.model, "messages": messages, "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens, "seed": self.cfg.seed,
        }
        if self.use_schema:
            body["response_format"] = RESPONSE_FORMAT
        return body

    def _cache_path(self, body: dict[str, Any]) -> Path:
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()
        return self.cfg.cache_dir / f"{key}.json"

    async def items(self, messages: list[dict[str, str]]) -> list[dict[str, Any]]:
        """Parsed `items` for a prompt. Only replies that parse are cached."""
        for attempt in range(self.cfg.retries + 1):
            body = self._body(messages)
            cache = self._cache_path(body)
            if cache.exists():
                return parse_items(json.loads(cache.read_text(encoding="utf-8"))["content"])
            try:
                response = await self.http.post("/chat/completions", json=body)
                if response.status_code in SCHEMA_REJECTED_STATUSES and self.use_schema:
                    logger.info("endpoint rejected response_format (HTTP %d); using plain JSON prompt", response.status_code)
                    self.use_schema = False
                    continue
                if response.status_code in RETRY_STATUSES:
                    raise httpx.HTTPStatusError(f"HTTP {response.status_code}", request=response.request, response=response)
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                items = parse_items(content)
            except (httpx.TransportError, BadResponse, KeyError, IndexError, json.JSONDecodeError) as e:
                logger.warning("attempt %d failed: %s: %s", attempt + 1, type(e).__name__, e)
            except httpx.HTTPStatusError as e:
                if e.response.status_code not in RETRY_STATUSES:
                    raise TeacherError(f"HTTP {e.response.status_code}: {e.response.text[:200]}") from e
                logger.warning("attempt %d failed: %s", attempt + 1, e)
            else:
                tmp = cache.with_suffix(".tmp")
                tmp.write_text(json.dumps({"content": content}), encoding="utf-8")
                tmp.replace(cache)
                return items
            await asyncio.sleep(self.cfg.backoff_s * 2**attempt)
        raise TeacherError(f"gave up after {self.cfg.retries + 1} attempts")


async def generate(pages: Sequence[Page], http: httpx.AsyncClient, cfg: TeacherConfig) -> tuple[list[Example], int]:
    """Raw (unfiltered) examples for `pages` and the number of batches that failed for good."""
    client = ChatClient(http, cfg)
    gate = asyncio.Semaphore(cfg.concurrency)
    failed = 0

    async def run_batch(page: Page, batch: list[Element]) -> list[Example]:
        nonlocal failed
        async with gate:
            try:
                items = await client.items(build_messages(page, batch))
            except TeacherError as e:
                failed += 1
                logger.error("page %s: batch failed: %s", page.page_id, e)
                return []
        return items_to_examples(page, batch, items)

    tasks = []
    for page in pages:
        sampled = sample_elements(page, random.Random(f"{cfg.seed}:{page.page_id}"), cfg.per_page)
        tasks += [run_batch(page, sampled[i:i + BATCH_SIZE]) for i in range(0, len(sampled), BATCH_SIZE)]
    return [ex for group in await asyncio.gather(*tasks) for ex in group], failed


# --- CLI --------------------------------------------------------------------------------------


async def _run(args: argparse.Namespace) -> int:
    pages = list(iter_pages(args.pages))[: args.limit_pages]
    key = os.environ.get(args.api_key_env, "") if args.api_key_env else ""
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    cfg = TeacherConfig(model=args.model, cache_dir=Path(args.cache_dir), seed=args.seed,
                        concurrency=args.concurrency, per_page=args.per_page)
    async with httpx.AsyncClient(base_url=args.base_url.rstrip("/"), headers=headers, timeout=args.timeout) as http:
        raw, failed = await generate(pages, http, cfg)
    result = filter_all({p.page_id: p for p in pages}, raw)
    write_examples(result.kept, args.out)
    print(f"pages={len(pages)} raw={len(raw)} kept={len(result.kept)} failed_batches={failed}")
    print(f"rejected: {format_rejections(result.rejected)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate instruction examples with an LLM.")
    parser.add_argument("--pages", required=True, help="pages directory (e.g. data/pages/train)")
    parser.add_argument("--out", required=True, help="examples JSONL to write")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible endpoint, e.g. http://localhost:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="env var holding the API key (optional)")
    parser.add_argument("--cache-dir", default="data/teacher_cache")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--per-page", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--limit-pages", type=int)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
