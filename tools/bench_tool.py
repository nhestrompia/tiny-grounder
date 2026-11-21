"""Latency and size of the grounding tool: p50/p95 of `find` INCLUDING extraction, and package size.

    uv run python tools/bench_tool.py                       # fixtures + the live URLs below
    uv run python tools/bench_tool.py --no-live --reps 5

Gates (docs/improvement-plan.md step 5): p95 find <= 150 ms on 500 elements including extraction; package <= 3 MB.
Rows: each ranker x threads, over all pages ("all"), over pages with >= 300 elements ("big"), and on the largest
page's elements tiled/truncated to exactly 500 (rank-only, plus that page's extraction time added to each sample).
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

from grounder.extract.web import extract_page_sync
from grounder.records import Element
from grounder.playwright_helper import extract_landmarks
from grounder_tool import UIGrounder, make_ranker

ROOT = Path(__file__).resolve().parent.parent
LIVE = [
    "https://en.wikipedia.org/wiki/Alan_Turing", "https://github.com/python/cpython", "https://news.ycombinator.com/",
    "https://docs.python.org/3/library/json.html", "https://developer.mozilla.org/en-US/docs/Web/JavaScript",
    "https://www.python.org/", "https://stackoverflow.com/questions", "https://www.bbc.com/news",
    "https://www.nytimes.com/", "https://www.reddit.com/", "https://www.apple.com/", "https://www.mozilla.org/",
]
QUERIES = ["search box", "sign in", "pricing link", "download button", "contact us", "open the settings"]
RANKERS = [("bm25", "bm25", None), ("xs-s0 int8", f"model:{ROOT}/dist/xs-s0", True),
           ("xs-s0 fp32", f"model:{ROOT}/dist/xs-s0", False), ("m-s0 int8", f"model:{ROOT}/dist/m-s0", True)]


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(p / 100 * (len(xs) - 1)))]


def tile(els: list[Element], n: int) -> list[Element]:
    out = [Element(**{**e.to_dict(), "id": i}) for i, e in enumerate((els * (n // max(len(els), 1) + 1))[:n])]
    return out


def size_mb(paths: list[Path]) -> float:
    return sum(p.stat().st_size for p in paths if p.exists()) / 1e6


def package_sizes() -> list[str]:
    rows = []
    for label, d, int8 in [("xs-s0 int8", "xs-s0", True), ("xs-s0 fp32", "xs-s0", False), ("m-s0 int8", "m-s0", True)]:
        files = [ROOT / "dist" / d / n for n in (("model.int8.onnx" if int8 else "model.onnx"), "tokenizer.json", "config.json")]
        rows.append(f"| {label} | {size_mb(files):.2f} MB | {'ok' if size_mb(files) <= 3 else 'OVER 3 MB'} |")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["uv", "build", "--wheel", "-o", tmp, str(ROOT)], capture_output=True, check=False)
        wheels = list(Path(tmp).glob("*.whl"))
        if wheels:
            rows.append(f"| code wheel (grounder + grounder_tool, no models) | {wheels[0].stat().st_size / 1e6:.2f} MB | |")
    rows.append(f"| bm25 ranker (own code: rankers.py + baselines/bm25.py) | "
                f"{size_mb([ROOT / 'grounder_tool/rankers.py', ROOT / 'grounder/baselines/bm25.py']):.3f} MB | ok (+ rank-bm25 dep) |")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--no-live", action="store_true")
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    args = ap.parse_args()

    urls = [p.as_uri() for p in sorted((ROOT / "tests/fixtures").glob("*.html"))]
    if not args.no_live:
        urls += LIVE
    pages: list[tuple[str, object, list[float]]] = []  # (url, page, extraction ms samples)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        tab = browser.new_page(viewport={"width": 1280, "height": 800})
        for url in urls:
            try:
                tab.goto(url, wait_until="load", timeout=20_000)
                tab.wait_for_timeout(300)
                times, page = [], None
                for _ in range(args.reps):
                    t = time.perf_counter()
                    page = extract_page_sync(tab)
                    times.append((time.perf_counter() - t) * 1e3)
                pages.append((url, page, times))
            except Exception as exc:  # noqa: BLE001 - a dead live URL must not stop the bench
                print(f"skip {url}: {str(exc).splitlines()[0]}", file=sys.stderr)
        browser.close()
    if not pages:
        raise SystemExit("no pages extracted")
    print(f"pages: {len(pages)}; elements min/median/max = "
          f"{min(len(p.elements) for _, p, _ in pages)}/{int(statistics.median(len(p.elements) for _, p, _ in pages))}/"
          f"{max(len(p.elements) for _, p, _ in pages)}\n")
    print("| page | elements | extraction p50 ms |\n|---|---|---|")
    for url, page, ext in sorted(pages, key=lambda x: len(x[1].elements)):
        print(f"| {url.rsplit('/', 1)[-1] or url} | {len(page.elements)} | {statistics.median(ext):.1f} |")
    print()
    big_src = max(pages, key=lambda x: len(x[1].elements))
    big500 = tile(big_src[1].elements, 500)
    rows = [(len(p.elements), statistics.median(t), t) for _, p, t in pages if len(p.elements) >= 100]
    if not rows:
        rows = [(len(p.elements), statistics.median(t), t) for _, p, t in pages if p.elements]
    per_el = statistics.median(m / n for n, m, _ in rows)
    est500 = per_el * 500
    print(f"extraction ~ {per_el:.3f} ms/element (median over {len(rows)} real pages with >= 100 elements) -> "
          f"~{per_el * 500:.0f} ms at 500 elements (ESTIMATE used by the 500 row; no real 500-element page was available)\n")

    print("| ranker | threads | set | n | p50 ms | p95 ms | gate p95<=150 |\n|---|---|---|---|---|---|---|")
    for label, spec, int8 in RANKERS:
        for threads in args.threads if spec != "bm25" else [1]:
            g = UIGrounder(make_ranker(spec, int8, threads))
            g.find(big500, "warm up")
            sets: dict[str, list[float]] = {"all": [], "big (>=300 els)": [], "500 tiled + est. extraction": []}
            for _, page, ext in pages:
                for r in range(args.reps):
                    q = QUERIES[r % len(QUERIES)]
                    t = time.perf_counter()
                    g.find(page, q)
                    total = ext[r] + (time.perf_counter() - t) * 1e3
                    sets["all"].append(total)
                    if len(page.elements) >= 300:
                        sets["big (>=300 els)"].append(total)
            for r in range(args.reps * 4):
                t = time.perf_counter()
                g.find(big500, QUERIES[r % len(QUERIES)])
                sets["500 tiled + est. extraction"].append(est500 + (time.perf_counter() - t) * 1e3)
            for name, xs in sets.items():
                if xs:
                    p95 = pct(xs, 95)
                    print(f"| {label} | {threads} | {name} | {len(xs)} | {pct(xs, 50):.1f} | {p95:.1f} | "
                          f"{'ok' if p95 <= 150 else 'FAIL'} |")
    print("\n| observe ranker | elements | p50 ms incl extraction | p95 ms | gate <=200 |")
    print("|---|---|---|---|---|")
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        tab = browser.new_page(viewport={"width": 1280, "height": 800})
        tab.set_content("<title>Install documentation</title><h1>Docs</h1>" + "".join(
            f'<a href="/docs/{i}">Install package {i}</a><br>' for i in range(500)))
        for spec in ("exact", f"blend:{ROOT}/dist/xs-s0:8"):
            grounder = UIGrounder(spec)
            samples = []
            for _ in range(args.reps + 1):
                start = time.perf_counter()
                page = extract_page_sync(tab)
                landmarks, _ = extract_landmarks(tab)
                grounder.observe(page, "Install a package", landmarks=landmarks)
                samples.append((time.perf_counter() - start) * 1000)
            samples = samples[1:]
            print(f"| {spec} | {len(page.elements)} | {pct(samples, 50):.1f} | "
                  f"{pct(samples, 95):.1f} | {'ok' if pct(samples, 95) <= 200 else 'FAIL'} |")
        browser.close()
    print("\n| package | size | gate <=3 MB |\n|---|---|---|")
    print("\n".join(package_sizes()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
