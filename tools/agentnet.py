"""Convert AgentNet (xlangai/AgentNet, MIT, OpenCUA) macOS trajectories into pages and goal-mode next-step examples.

AgentNet records a screenshot and a pyautogui action per step, but no accessibility tree. Candidates are the text
lines macOS Vision recognises in the screenshot (`ocrlines`); a click becomes an example only when it lands inside a
recognised line. Clicks on bare icons are counted and dropped.

The screenshots are one 124 GB split zip. Nothing close to that is downloaded: the zip's central directory (39 MB) is
read once, then each needed screenshot is fetched with two HTTP range requests and saved as a JPEG.

Tasks that use one of the held-out test apps are dropped (VLC appears in 66 AgentNet tasks).

    uv run python tools/agentnet.py --tasks 300 --out data/agentnet          # pilot
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
import random
import re
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grounder.data.pages import page_id_for, slugify, write_page  # noqa: E402
from grounder.records import Element, Example, Page, action_repr  # noqa: E402

BASE = "https://huggingface.co/datasets/xlangai/AgentNet/resolve/main"
META_URL = f"{BASE}/meta_data_merged.jsonl"
TRAJ_URL = f"{BASE}/agentnet_win_mac_18k.jsonl"
ZIP_DIR = f"{BASE}/win_mac_images"
LAST_DISK = 23
PART_SIZE = 5368709120
OCRLINES = Path(__file__).resolve().parents[1] / "grounder/extract/macos/.build/release/ocrlines"
TEST_APP_TERMS = ("logic pro", "vlc", "keka", "chess", "podcast", "audio midi")
CLICK_RE = re.compile(r"pyautogui\.(click|doubleClick|rightClick)\(\s*x\s*=\s*([0-9.]+)\s*,\s*y\s*=\s*([0-9.]+)")
DEV_FRACTION = 0.1
HIT_PAD = 4.0
GOOD_QUALITY = ("ok", "good", "excellent")


RETRIES = 8


def hf_headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    """HF_TOKEN, when set, raises Hugging Face's rate limits for the many range requests."""
    token = os.environ.get("HF_TOKEN")
    return {**({"Authorization": f"Bearer {token}"} if token else {}), **(extra or {})}


def http_range(url: str, start: int, end: int) -> bytes:
    """Bytes start..end of url; rate limits (429) and server/network errors are retried with backoff."""
    req = urllib.request.Request(url, headers=hf_headers({"Range": f"bytes={start}-{end}"}))
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code != 429 and exc.code < 500 or attempt == RETRIES - 1:
                raise
            wait = float(exc.headers.get("Retry-After") or 0) or min(300.0, 5.0 * 2 ** attempt)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == RETRIES - 1:
                raise
            wait = min(300.0, 5.0 * 2 ** attempt)
        time.sleep(wait + random.uniform(0, 2))
    raise AssertionError("unreachable")


def part_url(disk: int) -> str:
    return f"{ZIP_DIR}/images.zip" if disk == LAST_DISK else f"{ZIP_DIR}/images.z{disk + 1:02d}"


def read_span(disk: int, offset: int, length: int) -> bytes:
    """`length` bytes from (disk, offset), continuing into the next part when the span crosses a part boundary."""
    out = b""
    while length > 0:
        take = min(length, PART_SIZE - offset) if disk < LAST_DISK else length
        out += http_range(part_url(disk), offset, offset + take - 1)
        length -= take
        disk, offset = disk + 1, 0
    return out


def parse_central_directory(blob: bytes) -> dict[str, tuple[int, int, int, int]]:
    """{file name: (disk, local header offset, compressed size, method)}, zip64 fields included."""
    entries: dict[str, tuple[int, int, int, int]] = {}
    p = 0
    while blob[p:p + 4] == b"PK\x01\x02":
        (_, _, _, meth, _, _, _, csz, usz, nl, el, cl, disk, _, _, off) = struct.unpack("<HHHHHHIIIHHHHHII", blob[p + 4:p + 46])
        name = blob[p + 46:p + 46 + nl].decode("utf-8")
        extra = blob[p + 46 + nl:p + 46 + nl + el]
        q = 0
        while q + 4 <= len(extra):
            hid, size = struct.unpack("<HH", extra[q:q + 4])
            data, r = extra[q + 4:q + 4 + size], 0
            if hid == 1:
                if usz == 0xFFFFFFFF:
                    usz = struct.unpack("<Q", data[r:r + 8])[0]; r += 8
                if csz == 0xFFFFFFFF:
                    csz = struct.unpack("<Q", data[r:r + 8])[0]; r += 8
                if off == 0xFFFFFFFF:
                    off = struct.unpack("<Q", data[r:r + 8])[0]; r += 8
                if disk == 0xFFFF:
                    disk = struct.unpack("<I", data[r:r + 4])[0]
            q += 4 + size
        entries[name.rsplit("/", 1)[-1]] = (disk, off, csz, meth)
        p += 46 + nl + el + cl
    return entries


def load_index(cache: Path) -> dict[str, tuple[int, int, int, int]]:
    path = cache / "zip-index.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    size =int(urllib.request.urlopen(urllib.request.Request(part_url(LAST_DISK), method="HEAD", headers=hf_headers()), timeout=60)
               .headers["Content-Length"])
    end = http_range(part_url(LAST_DISK), size - 70000, size - 1)
    k = end.rfind(b"PK\x06\x06")
    if k < 0:
        raise RuntimeError("no zip64 end record in the last part")
    fields = struct.unpack("<QHHIIQQQQ", end[k + 4:k + 56])
    cd_disk, cd_size, cd_off = fields[4], fields[7], fields[8]
    entries = parse_central_directory(read_span(cd_disk, cd_off, cd_size))
    cache.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(entries))
    return entries


def fetch_image(entries: dict[str, tuple[int, int, int, int]], name: str) -> bytes:
    disk, off, csz, meth = entries[name]
    header = read_span(disk, off, 30)
    if header[:4] != b"PK\x03\x04":
        raise RuntimeError(f"{name}: no local header at disk {disk} offset {off}")
    nl, el = struct.unpack("<HH", header[26:30])
    start, d = off + 30 + nl + el, disk
    if d < LAST_DISK and start >= PART_SIZE:
        d, start = d + 1, start - PART_SIZE
    data = read_span(d, start, csz)
    return zlib.decompress(data, -15) if meth == 8 else data


def download(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return path


def uses_test_app(meta: dict[str, Any]) -> bool:
    text = " ".join([*(meta.get("applications") or []), meta.get("instruction") or ""]).casefold()
    return any(term in text for term in TEST_APP_TERMS)


def select_tasks(meta_path: Path, limit: int | None) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    """macOS tasks with good annotation quality, not touching a held-out test app; first `limit` by task id hash."""
    counts: Counter[str] = Counter()
    chosen: dict[str, dict[str, Any]] = {}
    for line in meta_path.open(encoding="utf-8"):
        meta = json.loads(line)
        if meta.get("system") != "Darwin":
            continue
        counts["mac_tasks"] += 1
        if (meta.get("verify_feedback") or {}).get("quality") not in GOOD_QUALITY:
            counts["drop_quality"] += 1
            continue
        if uses_test_app(meta):
            counts["drop_test_app"] += 1
            continue
        chosen[meta["task_id"]] = meta
    ordered = sorted(chosen, key=lambda t: hashlib.sha1(t.encode()).hexdigest())
    keep = ordered[:limit] if limit else ordered
    return {t: chosen[t] for t in keep}, counts


def stream_trajectories(cache: Path, wanted: set[str]) -> Iterator[dict[str, Any]]:
    """Trajectories of `wanted` tasks; the 1.4 GB file is streamed once and the matching lines cached."""
    cached = cache / "trajectories-mac.jsonl"
    have: dict[str, dict[str, Any]] = {}
    if cached.exists():
        for line in cached.open(encoding="utf-8"):
            row = json.loads(line)
            have[row["task_id"]] = row
    missing = wanted - set(have)
    if missing:
        with urllib.request.urlopen(urllib.request.Request(TRAJ_URL, headers=hf_headers()), timeout=600) as resp, cached.open("a", encoding="utf-8") as out:
            for raw in io.TextIOWrapper(resp, encoding="utf-8"):
                tid = raw[raw.find('"task_id"'):][:80]
                if not any(t in tid for t in missing):
                    continue
                row = json.loads(raw)
                if row["task_id"] in missing:
                    out.write(raw if raw.endswith("\n") else raw + "\n")
                    have[row["task_id"]] = row
                    missing.discard(row["task_id"])
                    if not missing:
                        break
    for tid in sorted(wanted):
        if tid in have:
            yield have[tid]


@dataclass
class Step:
    image: str
    kind: str | None
    x: float
    y: float
    action: str
    usable: bool


def parse_steps(traj: list[dict[str, Any]], completed: bool) -> list[Step]:
    """Steps in order; a click is usable when the next step judges it correct and not redundant (the last step: when
    the task completed)."""
    steps = []
    for i, step in enumerate(traj):
        value = step.get("value") or {}
        m = CLICK_RE.search(value.get("code") or "")
        nxt = (traj[i + 1].get("value") or {}) if i + 1 < len(traj) else None
        ok = (nxt.get("last_step_correct") is True and nxt.get("last_step_redundant") is not True) if nxt else completed
        steps.append(Step(image=step.get("image") or "", kind=m.group(1) if m else None,
                          x=float(m.group(2)) if m else 0.0, y=float(m.group(3)) if m else 0.0,
                          action=(value.get("action") or "").strip(), usable=bool(m) and ok))
    return steps


def ocr(png: bytes) -> tuple[int, int, list[dict[str, Any]]]:
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        f.write(png)
        f.flush()
        proc = subprocess.run([str(OCRLINES), f.name], capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"ocrlines exited with {proc.returncode}")
    out = json.loads(proc.stdout)
    return out["width"], out["height"], out["lines"]


def build_page(site: str, title: str, width: int, height: int, lines: list[dict[str, Any]]) -> Page:
    elements = []
    for i, line in enumerate(lines):
        x, y, w, h = (float(v) for v in line["box"])
        text = (line.get("text") or "").strip()
        if not text or w <= 0 or h <= 0:
            continue
        elements.append(Element(id=len(elements), role="text", text=text, tag="text",
                                box=[x / width, y / height, w / width, h / height], abs_box=[x, y, w, h],
                                source="ocr"))
    page = Page(page_id="", url="", site=site, title=title, viewport=[width, height], doc_size=[width, height],
                elements=elements)
    page.page_id = page_id_for(site, page)
    return page


def hit_ids(page: Page, px: float, py: float) -> list[int]:
    """Elements whose box (padded) contains the click, smallest first."""
    hits = []
    for el in page.elements:
        x, y, w, h = el.abs_box
        if x - HIT_PAD <= px <= x + w + HIT_PAD and y - HIT_PAD <= py <= y + h + HIT_PAD:
            hits.append((w * h, el.id))
    return [i for _, i in sorted(hits)]


def task_apps(meta: dict[str, Any]) -> list[str]:
    """The applications a task names, else its websites (browser tasks often name only the site)."""
    return list(meta.get("applications") or meta.get("websites") or ["unknown"])


def is_dev(app: str) -> bool:
    return int(hashlib.sha1(app.casefold().encode()).hexdigest()[:8], 16) % 1000 < DEV_FRACTION * 1000


def convert_task(meta: dict[str, Any], row: dict[str, Any], entries: dict[str, Any], out: Path,
                 counts: Counter[str]) -> list[Example]:
    apps = task_apps(meta)
    site = f"agentnet-{slugify(apps[0])}"
    goal = (row.get("natural_language_task") or row.get("instruction") or "").strip()
    steps = parse_steps(row.get("traj") or [], bool(row.get("task_completed")))
    history: list[str] = []
    examples = []
    for step in steps:
        if not step.kind:
            if step.action:
                history.append(step.action.split(". ")[0][:100])
            continue
        counts["clicks"] += 1
        if not step.usable or step.image not in entries:
            counts["drop_unusable_click" if not step.usable else "drop_missing_image"] += 1
            history.append(step.action.split(". ")[0][:100])
            continue
        png = fetch_image(entries, step.image)
        width, height, lines = ocr(png)
        page = build_page(site, apps[0], width, height, lines)
        ids = hit_ids(page, step.x * width, step.y * height)
        if not ids:
            counts["drop_click_not_on_text"] += 1
            history.append(step.action.split(". ")[0][:100])
            continue
        target = ids[0]
        write_page(page, out / "pages")
        shot = out / "screens" / page.site / f"{page.page_id}.jpg"
        shot.parent.mkdir(parents=True, exist_ok=True)
        Image.open(io.BytesIO(png)).convert("RGB").save(shot, "JPEG", quality=85)
        examples.append(Example(page_id=page.page_id, kind="goal", mode="goal", query=goal, history=list(history),
                                target_id=target, equiv_ids=ids, source=f"agentnet-{meta['task_id']}",
                                tags=["agentnet", step.kind]))
        counts["examples"] += 1
        history.append(action_repr(page.elements[target]))
    return examples


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/agentnet"))
    ap.add_argument("--tasks", type=int, default=300, help="number of macOS tasks (0 = all)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)
    if not OCRLINES.exists():
        raise SystemExit(f"build ocrlines first: {OCRLINES}")
    cache = args.out / "cache"
    meta_path = download(META_URL, cache / "meta_data_merged.jsonl")
    tasks, counts = select_tasks(meta_path, args.tasks or None)
    print(f"[agentnet] {len(tasks)} tasks selected; {dict(counts)}", flush=True)
    entries = load_index(cache)
    print(f"[agentnet] zip index: {len(entries)} images", flush=True)
    rows = list(stream_trajectories(cache, set(tasks)))
    print(f"[agentnet] {len(rows)} trajectories", flush=True)

    done_path = args.out / "done.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if done_path.exists():
        for line in done_path.open(encoding="utf-8"):
            rec = json.loads(line)
            done[rec["task_id"]] = rec
    todo = [r for r in rows if r["task_id"] not in done]
    print(f"[agentnet] {len(done)} tasks already done, {len(todo)} to convert", flush=True)

    def work(row: dict[str, Any]) -> tuple[dict[str, Any], list[Example], Counter[str], bool]:
        c: Counter[str] = Counter()
        try:
            return row, convert_task(tasks[row["task_id"]], row, entries, args.out, c), c, True
        except Exception as exc:  # one bad task must not end a multi-hour run; it is retried on the next run
            return row, [], Counter({f"task_error:{type(exc).__name__}": 1}), False

    errors: Counter[str] = Counter()
    with ThreadPoolExecutor(args.workers) as pool, done_path.open("a", encoding="utf-8") as log:
        for n, (row, exs, c, ok) in enumerate(pool.map(work, todo), 1):
            if ok:
                rec = {"task_id": row["task_id"], "counts": dict(c), "examples": [e.to_dict() for e in exs]}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                done[row["task_id"]] = rec
            else:
                errors.update(c)
            if n % 25 == 0:
                print(f"[agentnet] {n}/{len(todo)} tasks; errors {dict(errors)}", flush=True)

    per_split: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    apps: dict[str, Counter[str]] = {"train": Counter(), "dev": Counter()}
    for tid, rec in done.items():
        if tid not in tasks:
            continue
        counts.update(rec["counts"])
        app = task_apps(tasks[tid])[0]
        split = "dev" if is_dev(app) else "train"
        per_split[split].extend(rec["examples"])
        apps[split][app] += len(rec["examples"])
    counts.update(errors)
    for split, exs in per_split.items():
        with (args.out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for ex in exs:
                f.write(json.dumps(ex) + "\n")
    report = {"counts": dict(counts), "apps": {s: dict(a.most_common()) for s, a in apps.items()},
              "examples": {s: len(e) for s, e in per_split.items()}, "test_app_terms": TEST_APP_TERMS}
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"[agentnet] done: {report['examples']}; {dict(counts)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
