"""Reverse task synthesis for collected macOS walks using the local Gemma 4 26B-A4B llama.cpp server.

    uv run --group local python tools/mac_goals.py --walk-root data/mac_walks --split train --out data/nextstep --name mac_train_v2
    uv run --group local python tools/mac_goals.py --walk-root data/mac_walks --split test --out data/nextstep --name mac_test_v2

The GUI collection must be finished first. This phase starts the already-installed llama.cpp server with the
already-downloaded GGUF (no model download), labels walks in parallel, writes goal-mode next-step examples in the
custom format read by tools/next_step_recall.py, writes a build report, stops the server it started, and freezes a
SHA-256 manifest. It imports and runs no ranker.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import hashlib
import json
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import httpx

from grounder.data.pages import load_pages_from, write_examples, write_page
from grounder.features import content_tokens
from grounder.item_tags import TargetLabel, item_tags
from grounder.records import Example, Page, action_repr
from tools.mac_policy import load_policy

ROOT = Path(__file__).resolve().parents[1]
SERVER_BIN = Path.home() / ".unsloth/llama.cpp/llama-server"
MODEL_GLOB = str(Path.home() / ".cache/huggingface/hub/models--google--gemma-4-26B-A4B-it-qat-q4_0-gguf/snapshots/*/gemma-4-26B_q4_0-it.gguf")
MODEL_NAME = "google/gemma-4-26B-A4B-it-qat-q4_0-gguf"
MODEL_LICENSE = "Apache-2.0"
MODEL_CARD = "https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf"
DEFAULT_ENDPOINT = "http://127.0.0.1:8793"
LOCAL_PART_SIZE = 14.4 * 1024**3


@dataclass(frozen=True)
class GoalLabel:
    coherent: bool
    goal_literal: str
    goal_vague: str
    stop_step: int


@dataclass
class LabelResult:
    walk: dict[str, Any]
    label: GoalLabel | None
    error: str = ""
    offending_tokens: list[str] | None = None
    attempts: int = 0
    elapsed: float = 0.0


def find_model(pattern: str = MODEL_GLOB) -> Path:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Gemma GGUF not found at {pattern}; this pipeline must use the local copy")
    path = Path(matches[-1])
    if not path.is_file():
        raise FileNotFoundError(f"Gemma GGUF is not a file: {path}")
    return path


def load_walks(root: Path, split: str, min_steps: int = 2) -> tuple[list[dict[str, Any]], Counter[str]]:
    walks: list[dict[str, Any]] = []
    drops: Counter[str] = Counter()
    for path in sorted((root / split / "walks").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            walk = json.loads(line)
            if len(walk.get("steps") or []) >= min_steps:
                walks.append(walk)
            else:
                drops["too_few_steps"] += 1
    return walks, drops


def compact_page_summary(page: Page, limit: int = 650) -> str:
    """A short, deterministic state description for the reverse-synthesis prompt."""
    bits = []
    if page.title:
        bits.append(f"title {page.title!r}")
    seen = set()
    for element in page.elements:
        if not element.text.strip():
            continue
        item = f"{element.role}:{element.text}"
        if item in seen:
            continue
        seen.add(item)
        bits.append(item)
        if len("; ".join(bits)) > limit:
            break
    return "; ".join(bits)[:limit]


def walk_prompt(walk: dict[str, Any], pages: dict[str, Page], extra_instruction: str = "") -> str:
    lines = [
        "You are labelling a macOS GUI trajectory by reverse task synthesis.",
        "Decide whether the actions form one coherent user task. If they do, write the user's goal.",
        "Return ONLY one JSON object, no markdown, with exactly these keys:",
        '{"coherent":true,"goal_literal":"...","goal_vague":"...","stop_step":0}',
        "Rules:",
        "- coherent is false when the clicks are unrelated, accidental, or too little happens for a real task.",
        "- goal_literal is a natural user goal and may name controls or visible labels.",
        "- goal_vague is the same goal phrased without reusing any content word from any target label.",
        "- stop_step is the zero-based step index where the goal is achieved; it must be within the walk.",
        f"- there are {len(walk['steps'])} steps, valid stop_step values are 0 through {len(walk['steps']) - 1}.",
    ]
    if extra_instruction:
        lines.append(f"- {extra_instruction}")
    lines.append(f"App: {walk['app']}")
    for step in walk["steps"]:
        action = step["action"]
        pre = pages[step["pre_page_id"]]
        post = pages[step["post_page_id"]]
        typed = f" typed {action.get('typed_text')!r}" if action.get("type") == "type" else ""
        lines.extend([
            f"Step {step['index']}: {action.get('type')}{typed} target role={action.get('role')!r} label={action.get('label')!r}",
            f"  before: {compact_page_summary(pre)}",
            f"  after:  {compact_page_summary(post)}",
        ])
    return "\n".join(lines)


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse the first balanced-looking JSON object in a chat completion; empty or non-JSON is a failure."""
    text = text.strip()
    if not text:
        raise ValueError("empty message content")
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in message content")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("JSON content is not an object")
    return value


def validate_label(raw: dict[str, Any], n_steps: int) -> GoalLabel:
    try:
        label = GoalLabel(
            coherent=bool(raw["coherent"]),
            goal_literal=str(raw["goal_literal"]).strip(),
            goal_vague=str(raw["goal_vague"]).strip(),
            stop_step=int(raw["stop_step"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"missing or invalid goal JSON field: {exc}") from exc
    if label.coherent and (not label.goal_literal or not label.goal_vague):
        raise ValueError("coherent labels need both goal strings")
    if not 0 <= label.stop_step < n_steps:
        raise ValueError(f"stop_step outside 0..{n_steps - 1}")
    return label


def vague_offending_tokens(goal_vague: str, walk: dict[str, Any], stop_step: int) -> list[str]:
    """Content-token overlap with every target label in the walk, as mandated (not only the truncated prefix)."""
    del stop_step  # the mandate is walk-wide; keep it in the signature because callers naturally have it
    target_tokens: set[str] = set()
    for step in walk["steps"]:
        target_tokens.update(content_tokens(str(step["action"].get("label") or "")))
    return sorted(content_tokens(goal_vague) & target_tokens)


class LlamaServer:
    """Manage the prebuilt local llama.cpp server; only a server started here is stopped here."""

    def __init__(self, endpoint: str, model: Path, server_bin: Path, log_path: Path, start: bool = True) -> None:
        self.endpoint, self.model, self.server_bin, self.log_path = endpoint, model, server_bin, log_path
        self.start_requested = start
        self.process: subprocess.Popen[Any] | None = None
        self.started = False
        self._log = None

    def _up(self) -> bool:
        try:
            return httpx.get(f"{self.endpoint}/health", timeout=2).status_code < 500
        except httpx.HTTPError:
            return False

    def __enter__(self) -> "LlamaServer":
        if self._up():
            if self.start_requested:
                raise RuntimeError(f"a server is already answering at {self.endpoint}; stop it or pass --no-start-server")
            return self
        if not self.start_requested:
            raise RuntimeError(f"no llama.cpp server is answering at {self.endpoint}")
        if not self.server_bin.is_file():
            raise FileNotFoundError(f"llama-server not found at {self.server_bin}")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("ab")
        parsed = urllib.parse.urlparse(self.endpoint)
        port = str(parsed.port or 8793)
        host = parsed.hostname or "127.0.0.1"
        command = [str(self.server_bin), "-m", str(self.model), "--host", host, "--port", port,
                   "-c", "16384", "-np", "4", "-ngl", "99", "--no-webui"]
        self.process = subprocess.Popen(command, stdout=self._log, stderr=subprocess.STDOUT)
        self.started = True
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"llama-server exited with {self.process.returncode}; see {self.log_path}")
            if self._up():
                return self
            time.sleep(0.5)
        raise RuntimeError(f"llama-server did not become ready; see {self.log_path}")

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._log is not None:
            self._log.close()
        if not self.started or self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)


class GoalClient:
    """One OpenAI-compatible client per worker thread."""

    def __init__(self, endpoint: str, timeout: float = 180.0, max_tokens: int = 400) -> None:
        self.endpoint, self.timeout, self.max_tokens = endpoint, timeout, max_tokens
        self.local = threading.local()

    def _client(self) -> httpx.Client:
        client = getattr(self.local, "client", None)
        if client is None:
            client = self.local.client = httpx.Client(base_url=self.endpoint, timeout=self.timeout)
        return client

    def label(self, walk: dict[str, Any], pages: dict[str, Page], extra_instruction: str = "",
              max_tokens: int | None = None) -> GoalLabel:
        body = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": walk_prompt(walk, pages, extra_instruction)}],
            "temperature": 0.3,
            "max_tokens": max_tokens or self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        response = self._client().post("/v1/chat/completions", json=body)
        response.raise_for_status()
        data = response.json()
        content = str(((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
        if not content.strip() and (max_tokens or self.max_tokens) < 800:
            body["max_tokens"] = 800
            response = self._client().post("/v1/chat/completions", json=body)
            response.raise_for_status()
            data = response.json()
            content = str(((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
        return validate_label(parse_json_object(content), len(walk["steps"]))


def label_walk(client: GoalClient, walk: dict[str, Any], pages: dict[str, Page]) -> LabelResult:
    started = time.monotonic()
    result = LabelResult(walk, None)
    try:
        label = client.label(walk, pages)
        result.attempts += 1
        if label.coherent:
            offending = vague_offending_tokens(label.goal_vague, walk, label.stop_step)
            if offending:
                # One mandated retry, explicitly naming the mechanical violations. A second violation drops the walk.
                retry_instruction = ("The previous goal_vague illegally reused these target-label content tokens: "
                                     + ", ".join(offending) + ". Rewrite it without any of them.")
                retry = client.label(walk, pages, retry_instruction)
                result.attempts += 1
                retry_offending = vague_offending_tokens(retry.goal_vague, walk, retry.stop_step) if retry.coherent else []
                if retry.coherent and not retry_offending:
                    label = retry
                else:
                    result.offending_tokens = sorted(set(offending + retry_offending))
                    result.error = "goal_vague reused target-label tokens after one retry"
                    result.elapsed = time.monotonic() - started
                    return result
        result.label = label
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    result.elapsed = time.monotonic() - started
    return result


def examples_for_label(walk: dict[str, Any], label: GoalLabel, pages: dict[str, Page]) -> list[Example]:
    examples: list[Example] = []
    history: list[str] = []
    query = label.goal_vague
    for step in walk["steps"][:label.stop_step + 1]:
        page = pages[step["pre_page_id"]]
        target_id = int(step["target"]["id"])
        target = next((e for e in page.elements if e.id == target_id), None)
        if target is None:
            break
        tags = item_tags(query, history, TargetLabel(
            text=str(step["action"].get("label") or target.text),
            visible_text=str(step["target"].get("visible_text") or ""),
            nonvisual_text=str(step["target"].get("nonvisual_text") or ""),
        ))
        examples.append(Example(
            page_id=page.page_id, kind="goal", mode="goal", query=query, history=list(history),
            target_id=target_id, equiv_ids=[int(v) for v in step["target"].get("equiv_ids") or [target_id]],
            source=str(walk["walk_id"]), tags=tags,
        ))
        action = step["action"]
        typed = str(action.get("typed_text") or "") if action.get("type") == "type" else None
        history.append(action_repr(target, typed) if typed else action_repr(target))
    return examples


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_entry(path: Path, root: Path) -> dict[str, Any]:
    try:
        display = str(path.relative_to(root))
    except ValueError:
        display = str(path)
    return {"path": display, "sha256": sha256_file(path), "bytes": path.stat().st_size}


def write_manifest(out: Path, name: str, output_files: Iterable[Path], counts: dict[str, int]) -> Path:
    manifest = {
        "set": name,
        "status": "frozen",
        "counts": counts,
        "outputs": [manifest_entry(path, ROOT) for path in sorted(output_files)],
    }
    path = out / f"{name}.manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def prepare_output_pages_dir(path: Path) -> None:
    """Replace only a directory previously marked as generated by this script; never delete unknown data."""
    marker = path / ".mac_goals_generated"
    if path.exists() and any(path.iterdir()):
        if not marker.exists():
            raise RuntimeError(f"{path} exists and was not generated by tools/mac_goals.py; refusing to replace it")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    marker.write_text("generated by tools/mac_goals.py\n", encoding="utf-8")


def synthesize_and_write(args: argparse.Namespace) -> dict[str, Any]:
    walk_root = Path(args.walk_root)
    out = Path(args.out)
    walks, load_drops = load_walks(walk_root, args.split, args.min_steps)
    policy = load_policy(args.policy)
    before = len(walks)
    walks = [walk for walk in walks if not policy.refusal(str(walk.get("app") or ""), args.split)]
    load_drops["policy_refusals"] += before - len(walks)
    if args.max_walks:
        walks = walks[:args.max_walks]
    pages = load_pages_from([walk_root / args.split / "pages"])
    needed = {step[key] for walk in walks for step in walk["steps"] for key in ("pre_page_id", "post_page_id")}
    missing = sorted(needed - set(pages))
    if missing:
        raise RuntimeError(f"missing {len(missing)} walk pages, first: {missing[:3]}")
    model = find_model(args.model_glob)
    started = time.monotonic()
    results: list[LabelResult] = []
    log_path = out / f"{args.name}-llama-server.log"
    with LlamaServer(args.endpoint, model, Path(args.server_bin), log_path, start=not args.no_start_server):
        client = GoalClient(args.endpoint, timeout=args.request_timeout, max_tokens=args.max_tokens)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(label_walk, client, walk, pages) for walk in walks]
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                result = future.result()
                results.append(result)
                state = "coherent" if result.label and result.label.coherent else "drop"
                print(f"[{index}/{len(walks)}] {result.walk['walk_id']}: {state} {result.error}", flush=True)
    elapsed = time.monotonic() - started

    counts: Counter[str] = Counter(load_drops)
    per_app: dict[str, Counter[str]] = {}
    examples: list[Example] = []
    used_pages: dict[str, Page] = {}
    samples: list[dict[str, Any]] = []
    drops: list[dict[str, Any]] = []
    for result in sorted(results, key=lambda r: r.walk["walk_id"]):
        app = result.walk["app"]
        per_app.setdefault(app, Counter())
        per_app[app]["walks"] += 1
        if result.error:
            counts["label_errors"] += 1
            per_app[app]["label_errors"] += 1
            drops.append({"walk_id": result.walk["walk_id"], "app": app, "reason": result.error,
                          "offending_tokens": result.offending_tokens or []})
            continue
        label = result.label
        if label is None or not label.coherent:
            counts["incoherent"] += 1
            per_app[app]["incoherent"] += 1
            drops.append({"walk_id": result.walk["walk_id"], "app": app, "reason": "incoherent"})
            continue
        made = examples_for_label(result.walk, label, pages)
        if not made:
            counts["missing_target_or_page"] += 1
            per_app[app]["missing_target_or_page"] += 1
            continue
        counts["coherent_walks"] += 1
        counts["examples"] += len(made)
        per_app[app]["coherent_walks"] += 1
        per_app[app]["examples"] += len(made)
        examples.extend(made)
        for example in made:
            used_pages[example.page_id] = pages[example.page_id]
        if len(samples) < 5:
            samples.append({"walk_id": result.walk["walk_id"], "app": app,
                            "goal_literal": label.goal_literal, "goal_vague": label.goal_vague,
                            "stop_step": label.stop_step, "steps": len(result.walk["steps"])})

    out.mkdir(parents=True, exist_ok=True)
    pages_dir = out / f"pages_{args.name}"
    prepare_output_pages_dir(pages_dir)
    written_pages = [write_page(page, pages_dir) for page in sorted(used_pages.values(), key=lambda p: p.page_id)]
    examples_path = out / f"{args.name}.jsonl"
    write_examples(examples, examples_path)

    tag_counts: Counter[str] = Counter()
    for example in examples:
        tag_counts.update(example.tags)
    hours = elapsed / 3600 if elapsed else 0
    throughput = len(results) / hours if hours else 0.0
    report = {
        "status": "built",
        "name": args.name,
        "split": args.split,
        "model": {"name": MODEL_NAME, "license": MODEL_LICENSE, "card": MODEL_CARD,
                  "gguf": str(model), "gguf_bytes": model.stat().st_size, "gguf_sha256": sha256_file(model),
                  "approx_reported_size_gb": round(LOCAL_PART_SIZE / 1024**3, 1),
                  "architecture": "Gemma 4 26B-A4B instruction-tuned sparse mixture-of-experts, official Google QAT q4_0 GGUF",
                  "server": str(Path(args.server_bin))},
        "generation": {"endpoint": args.endpoint, "concurrency": args.concurrency, "max_tokens": args.max_tokens,
                       "elapsed_seconds": round(elapsed, 2),
                       "walks_per_hour": round(throughput, 1)},
        "counts": dict(counts) | {"walks_input": len(walks), "pages": len(written_pages)},
        "per_app": {app: dict(counter) for app, counter in sorted(per_app.items())},
        "tag_counts": dict(tag_counts),
        "samples": samples,
        "drops": drops,
    }
    report["output_bytes"] = examples_path.stat().st_size + sum(p.stat().st_size for p in written_pages)
    report_path = out / f"{args.name}-build.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    manifest = write_manifest(out, args.name, [examples_path, report_path, *written_pages], report["counts"])
    report["manifest"] = str(manifest)
    return report


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--walk-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=ROOT / "configs/apps_policy.yaml")
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "data/nextstep")
    parser.add_argument("--name", required=True)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--server-bin", default=str(SERVER_BIN))
    parser.add_argument("--model-glob", default=MODEL_GLOB)
    parser.add_argument("--no-start-server", action="store_true", help="use an already-running server; do not stop it")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--min-steps", type=int, default=2)
    parser.add_argument("--max-walks", type=int, default=0, help="smoke-test only the first N walks")
    args = parser.parse_args(argv)
    try:
        report = synthesize_and_write(args)
    except Exception as exc:
        print(f"goal synthesis failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {report['counts'].get('examples', 0)} examples, {report['counts'].get('pages', 0)} pages; "
          f"manifest {report['manifest']}; {report['generation']['walks_per_hour']} walks/hour")
    print(f"tags: {report['tag_counts']}; counts: {report['counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
