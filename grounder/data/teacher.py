"""LLM teacher: pages -> instruction examples through any OpenAI-compatible chat endpoint.

    uv run python -m grounder.data.teacher --pages data/pages --split train --out data/teacher_train.jsonl \
        --base-url http://localhost:8000/v1 --model teacher --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'

For OpenRouter: `--base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY` and
`--extra-body '{"provider": {"require_parameters": true}, "usage": {"include": true}}'`; see scripts/teacher_api.sh.

Sampled elements go to the model in batches (`BATCH_SIZE` per call). Each reply is cached on disk
under a hash of the request, so a rerun costs nothing and an interrupted run resumes where it stopped.
Only English pages are used (`--min-english`, see grounder.data.language).

A reply must cover every element of its batch exactly once with every field filled (`validate_items`); an incomplete
reply is requested once more, then the valid items are kept and the missing ones counted. A bad key or an empty
account (HTTP 401 / 402) stops the run; `--max-cost-usd` stops it when the reported cost passes a budget. Items
that survive go through `grounder.data.filters`; rejection counts are printed. Progress, coverage, throughput and
cost are logged every `--log-every` calls and summarised at the end (`--summary-out` writes them as JSON).
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
from grounder.data.filters import filter_all, format_rejections, is_duplicated, label_counts
from grounder.data.language import add_min_english_argument, english_pages
from grounder.data.pages import load_pages_from, write_examples
from grounder.data.template_teacher import SAMPLE_SIZE, sample_elements
from grounder.records import Element, Example, Page, equiv_ids

logger = logging.getLogger("grounder.teacher")

BATCH_SIZE = 10
N_NEIGHBOURS = 3
MAX_TOKENS = 1536  # 10 items of five short fields are ~900 tokens; headroom, not a budget
RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
SCHEMA_REJECTED_STATUSES = frozenset({400, 422})
FATAL_STATUSES = frozenset({401, 402})  # bad key / out of credits: no later call can succeed, so the whole run stops
INCOMPLETE_RETRIES = 1  # a reply that skips elements or fields is asked for once more
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
REQUIRED_FIELDS = ("plain", "paraphrase", "goal")  # `context` is required too, but only for elements with a shared text


class BadResponse(Exception):
    """The model's reply is not the expected JSON."""


class TeacherError(Exception):
    """A request failed for good (non-retryable status, or retries exhausted)."""


class Transient(Exception):
    """A failure worth retrying: a retryable HTTP status, or an `error` object in a 200 reply."""


class SchemaRejected(Exception):
    """The endpoint refused the `response_format` json_schema."""


class RunHalted(Exception):
    """Every later call would fail too (bad key, out of credits): stop the run and say why."""


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


def context_required(page: Page, batch: Sequence[Element]) -> list[bool]:
    """Per batch element: its text is shared on the page, so the reply must give a `context` query."""
    counts = label_counts(page.elements)
    return [is_duplicated(el, counts) for el in batch]


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


@dataclass
class Coverage:
    """How well one reply covers its batch. `received` counts valid items; `fields_empty` counts blank required fields."""

    expected: int
    received: int
    duplicate: int = 0
    fields_empty: int = 0

    @property
    def missing(self) -> int:
        return self.expected - self.received


def _filled(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_items(items: Sequence[dict[str, Any]], needs_context: Sequence[bool]) -> tuple[list[dict[str, Any]], Coverage]:
    """The valid items of a reply (every index 0..n-1 once, every required field a non-empty string) and its coverage.

    The first valid item of an index wins; an item with a blank required field is dropped, so its index is missing.
    """
    accepted: dict[int, dict[str, Any]] = {}
    duplicate = fields_empty = 0
    for item in items:
        index = item.get("i")
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(needs_context):
            continue
        if index in accepted:
            duplicate += 1
            continue
        required = REQUIRED_FIELDS + (("context",) if needs_context[index] else ())
        blank = sum(not _filled(item.get(name)) for name in required)
        if blank:
            fields_empty += blank
        else:
            accepted[index] = item
    valid = [accepted[i] for i in sorted(accepted)]
    return valid, Coverage(len(needs_context), len(valid), duplicate, fields_empty)


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
    max_cost_usd: float | None = None


@dataclass
class Stats:
    """Running totals of a generation run; `line` is the progress message, `summary` the final report.

    Item counts cover every batch that got a reply: `items_expected` elements asked for, `items_received` valid
    items returned, plus duplicate items and blank required fields seen along the way.
    """

    total: int
    calls: int = 0
    cache_hits: int = 0
    failed: int = 0
    skipped: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    items_expected: int = 0
    items_received: int = 0
    items_duplicate: int = 0
    fields_empty: int = 0
    halt: str | None = None
    started: float = field(default_factory=time.monotonic)

    @property
    def items_missing(self) -> int:
        return self.items_expected - self.items_received

    def add_usage(self, usage: dict[str, Any] | None) -> None:
        usage = usage or {}
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        cost = usage.get("cost")
        self.cost_usd += cost if isinstance(cost, (int, float)) else 0.0

    def add_coverage(self, cov: Coverage) -> None:
        self.items_expected += cov.expected
        self.items_received += cov.received
        self.items_duplicate += cov.duplicate
        self.fields_empty += cov.fields_empty

    def items_text(self) -> str:
        return (f"items {self.items_received}/{self.items_expected} (missing {self.items_missing}, "
                f"duplicate {self.items_duplicate}, empty fields {self.fields_empty})")

    def line(self, now: float | None = None) -> str:
        elapsed = max((time.monotonic() if now is None else now) - self.started, 1e-9)
        live = self.calls - self.cache_hits
        eta = f"{(self.total - self.calls) * elapsed / live / 60:.1f} min" if live else "n/a"
        return (f"{self.calls}/{self.total} calls, {self.prompt_tokens / elapsed:.0f} prompt tok/s, "
                f"{self.completion_tokens / elapsed:.0f} completion tok/s, cache hits {self.cache_hits}, "
                f"failed {self.failed}, {self.items_text()}, cost ${self.cost_usd:.4f}, "
                f"ETA {eta} (at the live-call rate)")

    def summary(self, now: float | None = None) -> dict[str, Any]:
        """Everything the pilot needs to extrapolate a full run, as a JSON-serialisable dict."""
        elapsed = (time.monotonic() if now is None else now) - self.started
        return {
            "calls": self.calls, "live_calls": self.calls - self.cache_hits, "cache_hits": self.cache_hits,
            "failed": self.failed, "skipped": self.skipped, "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens, "cost_usd": self.cost_usd, "seconds": elapsed,
            "items_expected": self.items_expected, "items_received": self.items_received,
            "items_missing": self.items_missing, "items_duplicate": self.items_duplicate,
            "fields_empty": self.fields_empty, "halt": self.halt,
        }

    def usage_text(self) -> str:
        return (f"usage (live calls only): {self.prompt_tokens} prompt tokens, {self.completion_tokens} completion tokens, "
                f"cost ${self.cost_usd:.4f}")


def fatal_message(status: int, detail: str) -> str:
    if status == 402:
        return f"out of credits (HTTP 402): top up the account and rerun; finished calls are cached. {detail}"
    return f"API key rejected (HTTP {status}): {detail}"


def check_payload_error(payload: Any) -> None:
    """Raise for an `error` object in a reply; OpenRouter sends one with HTTP 200 for upstream failures."""
    if not isinstance(payload, dict):
        raise BadResponse("reply is not a JSON object")
    error = payload.get("error")
    if not error:
        return
    detail = error if isinstance(error, str) else str(error.get("message", error))
    code = error.get("code") if isinstance(error, dict) else None
    status = int(code) if str(code).isdigit() else None
    if status in FATAL_STATUSES:
        raise RunHalted(fatal_message(status, detail[:200]))
    if status in RETRY_STATUSES:
        raise Transient(f"error {status} in reply: {detail[:200]}")
    raise TeacherError(f"error {code} in reply: {detail[:200]}")


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
        """Keyed on the whole request body, so the model id (and every other parameter) is part of the key."""
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()
        return self.cfg.cache_dir / f"{key}.json"

    @staticmethod
    def _store(cache: Path, content: str) -> None:
        tmp = cache.with_suffix(".tmp")
        tmp.write_text(json.dumps({"content": content}), encoding="utf-8")
        tmp.replace(cache)

    async def _post(self, body: dict[str, Any]) -> str:
        """The reply text of one request, counting its usage. Raises Transient, SchemaRejected, RunHalted or TeacherError."""
        response = await self.http.post("/chat/completions", json=body)
        status = response.status_code
        if status in FATAL_STATUSES:
            raise RunHalted(fatal_message(status, response.text[:200]))
        if status in SCHEMA_REJECTED_STATUSES and self.use_schema:
            raise SchemaRejected(f"HTTP {status}")
        if status in RETRY_STATUSES:
            raise Transient(f"HTTP {status}")
        if response.is_error:
            raise TeacherError(f"HTTP {status}: {response.text[:200]}")
        payload = response.json()
        check_payload_error(payload)
        content = payload["choices"][0]["message"]["content"]
        self.stats.add_usage(payload.get("usage"))
        return content

    async def items(self, messages: list[dict[str, str]], needs_context: Sequence[bool]) -> tuple[list[dict[str, Any]], Coverage]:
        """The valid `items` for a prompt and how completely they cover it. Only replies that parse are cached.

        A reply that skips elements or fields is requested once more; if it is still incomplete the better of
        the two is kept (a cache hit is final and never retried).
        """
        best: tuple[str, Path, list[dict[str, Any]], Coverage] | None = None
        incomplete_retries = INCOMPLETE_RETRIES
        for attempt in range(self.cfg.retries + 1):
            body = self._body(messages)
            cache = self._cache_path(body)  # keyed on the attempt-0 body, so a rerun finds whatever attempt was kept
            if cache.exists():
                self.stats.cache_hits += 1
                return validate_items(parse_items(json.loads(cache.read_text(encoding="utf-8"))["content"]), needs_context)
            try:
                # A retry with the same seed would often return the same (incomplete) reply.
                content = await self._post(body | {"seed": body["seed"] + attempt} if attempt else body)
                valid, cov = validate_items(parse_items(content), needs_context)
            except SchemaRejected as e:
                logger.info("endpoint rejected response_format (%s); using plain JSON prompt", e)
                self.use_schema = False
                continue
            except (httpx.TransportError, Transient, BadResponse, KeyError, IndexError, json.JSONDecodeError) as e:
                logger.warning("attempt %d failed: %s: %s", attempt + 1, type(e).__name__, e)
            else:
                if cov.missing == 0 or incomplete_retries == 0:
                    if best is not None and best[3].received > cov.received:
                        content, cache, valid, cov = best
                    self._store(cache, content)
                    return valid, cov
                logger.warning("attempt %d incomplete: %d of %d items valid; retrying", attempt + 1, cov.received, cov.expected)
                incomplete_retries -= 1
                best = (content, cache, valid, cov)
                continue
            await asyncio.sleep(self.cfg.backoff_s * 2**attempt)
        if best is not None:
            self._store(best[1], best[0])
            return best[2], best[3]
        raise TeacherError(f"gave up after {self.cfg.retries + 1} attempts")


def over_budget(stats: Stats, cfg: TeacherConfig) -> bool:
    return cfg.max_cost_usd is not None and stats.cost_usd > cfg.max_cost_usd


async def generate(pages: Sequence[Page], http: httpx.AsyncClient, cfg: TeacherConfig) -> tuple[list[Example], Stats]:
    """Raw (unfiltered) examples for `pages` and the run's stats.

    The run stops launching calls (`stats.halt` says why) when the endpoint reports a bad key or no credits left,
    or the accumulated cost exceeds `cfg.max_cost_usd`; calls already in flight finish.
    """
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
            if stats.halt is None and over_budget(stats, cfg):
                stats.halt = f"cost limit ${cfg.max_cost_usd} exceeded (spent ${stats.cost_usd:.4f})"
            if stats.halt:
                stats.skipped += 1
                return out
            try:
                items, cov = await client.items(build_messages(page, batch), context_required(page, batch))
            except TeacherError as e:
                stats.failed += 1
                logger.error("page %s: batch failed: %s", page.page_id, e)
            except RunHalted as e:
                stats.halt = stats.halt or str(e)
                logger.error("stopping: %s", e)
                return out
            else:
                stats.add_coverage(cov)
                if items:
                    out = items_to_examples(page, batch, items)
                else:
                    stats.failed += 1
                    logger.error("page %s: batch failed: no valid items in the reply", page.page_id)
        stats.calls += 1
        if stats.calls % cfg.log_every == 0 or stats.calls == stats.total:
            logger.info(stats.line())
        return out

    groups = await asyncio.gather(*(run_batch(page, batch) for page, batch in batches))
    return [ex for group in groups for ex in group], stats


# --- CLI --------------------------------------------------------------------------------------


def pick_pages(pages_dirs: Sequence[str | Path], limit: int | None, seed: int, min_english: float) -> list[Page]:
    """The English pages under `pages_dirs`, or a seeded random `limit` of them (spreads a pilot over sites and sources).

    Non-English pages are never offered to the teacher, and the limit counts English pages only.
    """
    everything = list(load_pages_from(pages_dirs).values())
    pages = list(english_pages(everything, min_english))
    logger.info("%d English pages of %d (%d skipped)", len(pages), len(everything), len(everything) - len(pages))
    if limit is not None and limit < len(pages):
        pages = sorted(random.Random(seed).sample(pages, limit), key=lambda p: p.page_id)
    return pages


def report(stats: Stats, summary_out: str | None) -> None:
    print(stats.items_text())
    print(stats.usage_text())
    if summary_out:
        Path(summary_out).write_text(json.dumps(stats.summary(), indent=2), encoding="utf-8")


def auth_headers(api_key_env: str | None) -> dict[str, str]:
    """`Authorization: Bearer <key>` from the environment variable `api_key_env`, or no header when it is unset."""
    key = os.environ.get(api_key_env, "") if api_key_env else ""
    return {"Authorization": f"Bearer {key}"} if key else {}


async def _run(args: argparse.Namespace) -> int:
    dirs = [Path(d) / args.split if args.split else Path(d) for d in args.pages]
    pages = pick_pages(dirs, args.limit_pages, args.seed, args.min_english)
    cfg = TeacherConfig(model=args.model, cache_dir=Path(args.cache_dir), seed=args.seed,
                        concurrency=args.concurrency, per_page=args.per_page, max_tokens=args.max_tokens,
                        extra_body=args.extra_body, log_every=args.log_every, retries=args.retries,
                        max_cost_usd=args.max_cost_usd)
    async with httpx.AsyncClient(base_url=args.base_url.rstrip("/"), headers=auth_headers(args.api_key_env), timeout=args.timeout) as http:
        raw, stats = await generate(pages, http, cfg)
    report(stats, args.summary_out)
    if stats.halt:
        print(f"HALTED: {stats.halt}\n{stats.skipped} batches not run; nothing written to {args.out}. "
              "Rerun the same command to resume: finished calls are cached.", file=sys.stderr)
        return 1
    result = filter_all({p.page_id: p for p in pages}, raw)
    write_examples(result.kept, args.out)
    kinds = Counter(ex.kind for ex in raw)
    print(f"pages={len(pages)} raw={len(raw)} kept={len(result.kept)} ({len(result.kept) / max(len(raw), 1):.1%} pass) "
          f"failed_batches={stats.failed}")
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
    parser.add_argument("--pages", nargs="+", required=True, help="pages directories (e.g. data/pages, or data/pages/train)")
    parser.add_argument("--split", choices=SPLITS, help="read only {pages}/{split} of each directory")
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
    parser.add_argument("--retries", type=int, default=4, help="retries per call after a transient failure")
    parser.add_argument("--max-cost-usd", type=float, help="stop launching calls once the reported cost exceeds this")
    parser.add_argument("--summary-out", help="write the run's usage and coverage totals as JSON")
    parser.add_argument("--seed", type=int, default=0)
    add_min_english_argument(parser)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
