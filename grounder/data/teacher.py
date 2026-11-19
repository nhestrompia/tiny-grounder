"""LLM teacher: pages -> instruction examples through any OpenAI-compatible chat endpoint.

    uv run python -m grounder.data.teacher --pages data/pages --split train --out data/teacher_train.jsonl \
        --base-url http://localhost:8000/v1 --model teacher --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'

Sampled elements go to the model in batches (`BATCH_SIZE` per call). Each reply is cached on disk
under a hash of the request, so a rerun costs nothing and an interrupted run resumes where it stopped.
Only English pages are used (`--min-english`, see grounder.data.language). Replies are validated by `grounder.data.filters`; rejection counts are printed. Progress and throughput
(requests, prompt / completion tokens per second, cache hits, ETA) are logged every `--log-every` calls.
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
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import httpx

from grounder.data.crawl import SPLITS
from grounder.data.filters import filter_all, format_rejections, label_counts
from grounder.data.language import add_min_english_argument, english_pages
from grounder.data.pages import load_page, page_paths, write_examples
from grounder.data.template_teacher import SAMPLE_SIZE, sample_elements
from grounder.records import Element, Example, Page, equiv_ids

logger = logging.getLogger("grounder.teacher")

BATCH_SIZE = 10
N_NEIGHBOURS = 3
MAX_TOKENS = 1536  # 10 items of five short fields are ~900 tokens; headroom, not a budget
RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
SCHEMA_REJECTED_STATUSES = frozenset({400, 422})
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$")

SYSTEM_PROMPT = """\
You write training data for a model that finds the right UI element on a web page or in a desktop app window. For each numbered element, \
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


def page_source(url: str) -> str:
    """The web host, or "macOS app <name>" for accessibility dumps (ax://<app or bundle id>; Screen2AX has no name)."""
    if url.startswith("ax://"):
        name = url.removeprefix("ax://")
        return "macOS app" if name.startswith("s2ax-") else f"macOS app {name}"
    return re.sub(r"^https?://", "", url).split("/")[0]


def build_messages(page: Page, batch: Sequence[Element]) -> list[dict[str, str]]:
    counts = label_counts(page.elements)
    lines = [f'Page: "{page.title}" ({page_source(page.url)})'] + [element_line(i, el, page, counts) for i, el in enumerate(batch)]
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
    max_tokens: int = MAX_TOKENS
    retries: int = 4
    backoff_s: float = 1.0
    concurrency: int = 16
    per_page: int = SAMPLE_SIZE
    extra_body: dict[str, Any] = field(default_factory=dict)
    log_every: int = 100


@dataclass
class Stats:
    """Running totals of a generation run; `line` is the progress message."""

    total: int
    calls: int = 0
    cache_hits: int = 0
    failed: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    started: float = field(default_factory=time.monotonic)

    def line(self, now: float | None = None) -> str:
        elapsed = max((time.monotonic() if now is None else now) - self.started, 1e-9)
        live = self.calls - self.cache_hits
        eta = f"{(self.total - self.calls) * elapsed / live / 60:.1f} min" if live else "n/a"
        return (f"{self.calls}/{self.total} calls, {self.prompt_tokens / elapsed:.0f} prompt tok/s, "
                f"{self.completion_tokens / elapsed:.0f} completion tok/s, cache hits {self.cache_hits}, "
                f"failed {self.failed}, ETA {eta} (at the live-call rate)")


class ChatClient:
    """Chat-completions client with a disk cache, retries and a json_schema -> plain JSON fallback."""

    def __init__(self, http: httpx.AsyncClient, cfg: TeacherConfig, stats: Stats) -> None:
        self.http = http
        self.cfg = cfg
        self.stats = stats
        self.use_schema = True
        cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    def _body(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.cfg.model, "messages": messages, "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens, "seed": self.cfg.seed,
        }
        if self.use_schema:
            body["response_format"] = RESPONSE_FORMAT
        return body | self.cfg.extra_body

    def _cache_path(self, body: dict[str, Any]) -> Path:
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()
        return self.cfg.cache_dir / f"{key}.json"

    async def items(self, messages: list[dict[str, str]]) -> list[dict[str, Any]]:
        """Parsed `items` for a prompt. Only replies that parse are cached."""
        for attempt in range(self.cfg.retries + 1):
            body = self._body(messages)
            cache = self._cache_path(body)
            if cache.exists():
                self.stats.cache_hits += 1
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
                payload = response.json()
                content = payload["choices"][0]["message"]["content"]
                items = parse_items(content)
            except (httpx.TransportError, BadResponse, KeyError, IndexError, json.JSONDecodeError) as e:
                logger.warning("attempt %d failed: %s: %s", attempt + 1, type(e).__name__, e)
            except httpx.HTTPStatusError as e:
                if e.response.status_code not in RETRY_STATUSES:
                    raise TeacherError(f"HTTP {e.response.status_code}: {e.response.text[:200]}") from e
                logger.warning("attempt %d failed: %s", attempt + 1, e)
            else:
                usage = payload.get("usage") or {}
                self.stats.prompt_tokens += usage.get("prompt_tokens", 0)
                self.stats.completion_tokens += usage.get("completion_tokens", 0)
                tmp = cache.with_suffix(".tmp")
                tmp.write_text(json.dumps({"content": content}), encoding="utf-8")
                tmp.replace(cache)
                return items
            await asyncio.sleep(self.cfg.backoff_s * 2**attempt)
        raise TeacherError(f"gave up after {self.cfg.retries + 1} attempts")


async def generate(pages: Sequence[Page], http: httpx.AsyncClient, cfg: TeacherConfig) -> tuple[list[Example], int]:
    """Raw (unfiltered) examples for `pages` and the number of batches that failed for good."""
    batches = []
    for page in pages:
        sampled = sample_elements(page, random.Random(f"{cfg.seed}:{page.page_id}"), cfg.per_page)
        batches += [(page, sampled[i:i + BATCH_SIZE]) for i in range(0, len(sampled), BATCH_SIZE)]
    stats = Stats(total=len(batches))
    client = ChatClient(http, cfg, stats)
    gate = asyncio.Semaphore(cfg.concurrency)

    async def run_batch(page: Page, batch: list[Element]) -> list[Example]:
        out: list[Example] = []
        async with gate:
            try:
                out = items_to_examples(page, batch, await client.items(build_messages(page, batch)))
            except TeacherError as e:
                stats.failed += 1
                logger.error("page %s: batch failed: %s", page.page_id, e)
        stats.calls += 1
        if stats.calls % cfg.log_every == 0 or stats.calls == stats.total:
            logger.info(stats.line())
        return out

    groups = await asyncio.gather(*(run_batch(page, batch) for page, batch in batches))
    return [ex for group in groups for ex in group], stats.failed


# --- CLI --------------------------------------------------------------------------------------


def pick_pages(pages_dir: str | Path, limit: int | None, seed: int, min_english: float) -> list[Page]:
    """The English pages under `pages_dir`, or a seeded random `limit` of them (spreads a pilot over sites).

    Non-English pages are never offered to the teacher, and the limit counts English pages only.
    """
    everything = [load_page(p) for p in page_paths(pages_dir)]
    pages = list(english_pages(everything, min_english))
    logger.info("%d English pages of %d (%d skipped)", len(pages), len(everything), len(everything) - len(pages))
    if limit is not None and limit < len(pages):
        pages = sorted(random.Random(seed).sample(pages, limit), key=lambda p: p.page_id)
    return pages


async def _run(args: argparse.Namespace) -> int:
    pages = pick_pages(Path(args.pages) / args.split if args.split else args.pages, args.limit_pages, args.seed, args.min_english)
    key = os.environ.get(args.api_key_env, "") if args.api_key_env else ""
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    cfg = TeacherConfig(model=args.model, cache_dir=Path(args.cache_dir), seed=args.seed,
                        concurrency=args.concurrency, per_page=args.per_page, max_tokens=args.max_tokens,
                        extra_body=args.extra_body, log_every=args.log_every)
    async with httpx.AsyncClient(base_url=args.base_url.rstrip("/"), headers=headers, timeout=args.timeout) as http:
        raw, failed = await generate(pages, http, cfg)
    result = filter_all({p.page_id: p for p in pages}, raw)
    write_examples(result.kept, args.out)
    kinds = Counter(ex.kind for ex in raw)
    print(f"pages={len(pages)} raw={len(raw)} kept={len(result.kept)} ({len(result.kept) / max(len(raw), 1):.1%} pass) "
          f"failed_batches={failed}")
    print(f"raw per kind: {format_rejections(kinds)}")
    print(f"rejected: {format_rejections(result.rejected)}")
    if kinds["paraphrase"]:
        print(f"paraphrases sharing no word with the element: {1 - result.rejected['paraphrase_overlap'] / kinds['paraphrase']:.1%}")
    return 0


def json_object(text: str) -> dict[str, Any]:
    value = json.loads(text)
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate instruction examples with an LLM.")
    parser.add_argument("--pages", required=True, help="pages directory (e.g. data/pages, or data/pages/train)")
    parser.add_argument("--split", choices=SPLITS, help="read only {pages}/{split}")
    parser.add_argument("--out", required=True, help="examples JSONL to write")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible endpoint, e.g. http://localhost:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="env var holding the API key (optional)")
    parser.add_argument("--cache-dir", default="data/teacher_cache")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--per-page", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--limit-pages", type=int, help="seeded random sample of this many pages (a pilot)")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--extra-body", type=json_object, default={}, help="JSON merged into every request body, e.g. chat_template_kwargs")
    parser.add_argument("--log-every", type=int, default=100, help="log progress every N calls")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--seed", type=int, default=0)
    add_min_english_argument(parser)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
