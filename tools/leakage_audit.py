"""Train/eval leakage audit (improvement plan v2.1, step 1).

Reports how much each evaluation source overlaps TRAINING data, by registrable domain, normalised exact URL,
macOS/desktop app name and near-duplicate page content (element-text sets, Jaccard >= 0.9, found with MinHash
LSH and verified exactly). Writes docs/leakage.md, data/leakage/heldout_domains.txt (every registrable domain in
any training page) and data/leakage/train_apps.txt.

    uv run python tools/leakage_audit.py
"""

from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator
from urllib.parse import parse_qsl, urlencode, urlsplit

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grounder.data.crawl import registrable_domain  # noqa: E402

JACCARD = 0.9
MIN_TEXTS = 5  # pages with fewer distinct element texts are too generic for content matching
NUM_PERM = 64
BAND_ROWS = 4  # 16 bands x 4 rows: P(candidate | J=0.9) ~ 1, false candidates are removed by the exact check

# (name, directories). Training also uses data/pages/dev and data/deskforge/pages/dev.
TRAIN_SOURCES: dict[str, list[str]] = {
    "web": ["data/pages/train", "data/pages/dev"],
    "ax": ["data/pages_ax/train"],
    "s2ax": ["data/pages_s2ax/train"],
    "deskforge": ["data/deskforge/pages/train", "data/deskforge/pages/dev"],
    "mind2web": ["data/mind2web/pages/train"],
}
EVAL_SOURCES: dict[str, list[str]] = {
    "web_test": ["data/pages/test"],
    "deskforge_test_app": ["data/deskforge/pages/test_app"],
    "mind2web_test_task": ["data/mind2web/pages/test_task"],
    "mind2web_test_website": ["data/mind2web/pages/test_website"],
    "mind2web_test_domain": ["data/mind2web/pages/test_domain"],
    "guirilla_test": ["data/guirilla/pages/test"],
    "realq": ["data/realq/pages"],
}
_URL_LIKE = re.compile(r"^url:\s*(\S+)", re.M)
_APP_PREFIXES = ("org.gnome.", "gnome-", "gnome.", "com.apple.", "org.")


# ---- keys ---------------------------------------------------------------------------------------------------


def normalize_url(url: str) -> str:
    """Scheme, `www.`, fragment, trailing slash and query order removed; lower-cased host and path."""
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower()
    host = host[4:] if host.startswith("www.") else host
    path = parts.path.rstrip("/") or ""
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return f"{host}{path}" + (f"?{query}" if query else "")


def _alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.casefold())


def app_forms(name: str) -> set[str]:
    """Normalised spellings of an app name that count as the same app: the whole name, the last dotted segment
    (a bundle id such as `com.apple.Notes` -> `notes`) and the name without a `gnome-` style prefix. Names
    shorter than 4 characters are dropped (too generic to match on)."""
    raw = name.strip().casefold()
    forms = {_alnum(raw), _alnum(raw.rsplit(".", 1)[-1])}
    for prefix in _APP_PREFIXES:
        if raw.startswith(prefix):
            forms.add(_alnum(raw[len(prefix):]))
            forms.add(_alnum(raw[len(prefix):].rsplit(".", 1)[-1]))
    return {f for f in forms if len(f) >= 4}


def site_domain_key(site: str) -> str:
    """Mind2Web site names are `kohls` or `new.mta.info`: a dotted name is reduced to its registrable domain."""
    return registrable_domain(site) if "." in site else site.casefold()


@dataclass
class PageInfo:
    source: str
    page_id: str
    url: str
    site: str
    domain: str = ""  # registrable domain (or Mind2Web site key); "" for app pages
    sld: str = ""  # second-level label of `domain`, to match Mind2Web site names against web domains
    app: str = ""  # raw app name ("" for web pages)
    norm_url: str = ""
    texts: frozenset[str] = field(default_factory=frozenset)


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().casefold()


def page_info(source: str, d: dict) -> PageInfo:
    """Build the audit record of one raw page dict (`Page.to_dict()` layout)."""
    url, site = d.get("url", ""), d.get("site", "")
    scheme = urlsplit(url).scheme
    info = PageInfo(source=source, page_id=d.get("page_id", ""), url=url, site=site)
    if scheme in ("http", "https"):
        info.domain = registrable_domain(url)
        info.norm_url = normalize_url(url)
    elif scheme == "mind2web":
        info.domain = site_domain_key(site)
    elif scheme == "ax":
        rest = url[len("ax://"):]
        if rest.startswith("deskforge/"):
            info.app = rest[len("deskforge/"):]
        elif source == "s2ax":
            info.app = ""  # anonymised: the site is a hash, the app name is not recorded
        else:
            info.app = rest
    info.sld = info.domain.split(".")[0] if info.domain else ""
    texts = {norm_text(e.get("text", "")) for e in d.get("elements", [])}
    info.texts = frozenset(t for t in texts if len(t) >= 2)
    return info


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---- training index -----------------------------------------------------------------------------------------


def _h64(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8", "ignore"), digest_size=8).digest(), "little")


class MinHasher:
    def __init__(self, num_perm: int = NUM_PERM, seed: int = 7) -> None:
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, 2**63, size=num_perm, dtype=np.uint64) | np.uint64(1)
        self.b = rng.integers(0, 2**63, size=num_perm, dtype=np.uint64)
        self._cache: dict[str, int] = {}

    def signature(self, texts: Iterable[str]) -> np.ndarray:
        hashes = np.array([self._cache.setdefault(t, _h64(t)) for t in texts], dtype=np.uint64)
        with np.errstate(over="ignore"):
            return (hashes[:, None] * self.a[None, :] + self.b[None, :]).min(axis=0)


class TrainIndex:
    """Everything a training set contains that an eval page can collide with."""

    def __init__(self) -> None:
        self.domains: dict[str, set[str]] = defaultdict(set)  # domain -> training sources
        self.slds: dict[str, set[str]] = defaultdict(set)
        self.urls: dict[str, set[str]] = defaultdict(set)
        self.apps: dict[str, set[str]] = defaultdict(set)  # app form -> raw app names
        self.sets: list[PageInfo] = []
        self._buckets: dict[tuple[int, bytes], list[int]] = defaultdict(list)
        self._exact: dict[frozenset[str], int] = {}
        self._hasher = MinHasher()

    def add(self, info: PageInfo) -> None:
        if info.domain:
            self.domains[info.domain].add(info.source)
            self.slds[info.sld].add(info.source)
        if info.norm_url:
            self.urls[info.norm_url].add(info.source)
        if info.app:
            for form in app_forms(info.app):
                self.apps[form].add(info.app)
        if len(info.texts) >= MIN_TEXTS:
            idx = len(self.sets)
            self.sets.append(info)
            self._exact.setdefault(info.texts, idx)
            sig = self._hasher.signature(info.texts)
            for band in range(NUM_PERM // BAND_ROWS):
                self._buckets[(band, sig[band * BAND_ROWS:(band + 1) * BAND_ROWS].tobytes())].append(idx)

    def domain_hit(self, info: PageInfo) -> set[str]:
        """Training sources that share this page's registrable domain. A Mind2Web site name without a dot
        (`kohls`) also matches a web domain with that second-level label (`kohls.com`)."""
        if not info.domain:
            return set()
        hit = set(self.domains.get(info.domain, ()))
        if info.source.startswith("mind2web") and "." not in info.domain:
            hit |= self.slds.get(info.domain, set())
        return hit

    def app_hit(self, info: PageInfo) -> set[str]:
        out: set[str] = set()
        for form in app_forms(info.app) if info.app else ():
            out |= self.apps.get(form, set())
        return out

    def url_hit(self, info: PageInfo) -> set[str]:
        return set(self.urls.get(info.norm_url, ())) if info.norm_url else set()

    def near_duplicate(self, info: PageInfo, threshold: float = JACCARD) -> tuple[float, PageInfo | None]:
        """(best Jaccard, that training page) among training pages that share an LSH bucket (1.0 for an identical
        element-text set); (0.0, None) when the page is too small or nothing reaches `threshold`."""
        if len(info.texts) < MIN_TEXTS:
            return 0.0, None
        if info.texts in self._exact:
            return 1.0, self.sets[self._exact[info.texts]]
        sig = self._hasher.signature(info.texts)
        cands: set[int] = set()
        for band in range(NUM_PERM // BAND_ROWS):
            cands.update(self._buckets.get((band, sig[band * BAND_ROWS:(band + 1) * BAND_ROWS].tobytes()), ()))
        best, best_page = 0.0, None
        for i in cands:
            j = jaccard(info.texts, self.sets[i].texts)
            if j > best:
                best, best_page = j, self.sets[i]
        return (best, best_page) if best >= threshold else (0.0, None)


# ---- audit --------------------------------------------------------------------------------------------------


@dataclass
class SourceReport:
    name: str
    pages: int = 0
    domains: set[str] = field(default_factory=set)
    apps: set[str] = field(default_factory=set)
    domain_pages: int = 0
    url_pages: int = 0
    app_pages: int = 0
    near_pages: int = 0  # near-duplicate (Jaccard >= threshold) of any training page
    exact_pages: int = 0  # identical element-text set
    near_cross_pages: int = 0  # near-duplicate whose training page is on a different domain / app
    comparable_pages: int = 0  # pages with enough distinct texts for content matching
    overlap_domains: dict[str, set[str]] = field(default_factory=dict)  # domain -> training sources
    overlap_apps: dict[str, set[str]] = field(default_factory=dict)  # eval app -> matching training apps
    near_examples: list[tuple[str, str, float]] = field(default_factory=list)


def audit_source(name: str, infos: Iterable[PageInfo], index: TrainIndex, threshold: float = JACCARD) -> SourceReport:
    rep = SourceReport(name)
    for info in infos:
        rep.pages += 1
        if info.domain:
            rep.domains.add(info.domain)
        if info.app:
            rep.apps.add(info.app)
        dh = index.domain_hit(info)
        if dh:
            rep.domain_pages += 1
            rep.overlap_domains.setdefault(info.domain, set()).update(dh)
        if index.url_hit(info):
            rep.url_pages += 1
        ah = index.app_hit(info)
        if ah:
            rep.app_pages += 1
            rep.overlap_apps.setdefault(info.app, set()).update(ah)
        if len(info.texts) >= MIN_TEXTS:
            rep.comparable_pages += 1
        j, other = index.near_duplicate(info, threshold)
        if other is not None:
            rep.near_pages += 1
            rep.exact_pages += j == 1.0
            same = (info.domain and info.domain == other.domain) or (info.app and info.app == other.app)
            if not same:
                rep.near_cross_pages += 1
                if len(rep.near_examples) < 8:
                    rep.near_examples.append((info.url, other.url, round(j, 3)))
    return rep


def agent_task_report(tasks: list[dict], visited: dict[str, set[str]], index: TrainIndex) -> dict:
    """Per agent task: is its start domain / any visited domain / its start URL in training data?"""
    rows = []
    for t in tasks:
        start = t["start_url"]
        dom = registrable_domain(start)
        seen = visited.get(t["id"], set())
        visited_domains = {registrable_domain(u) for u in seen} | {dom}
        rows.append({
            "id": t["id"], "start_url": start, "domain": dom,
            "start_domain_in_train": dom in index.domains,
            "start_url_in_train": normalize_url(start) in index.urls,
            "visited_domains": sorted(visited_domains),
            "visited_domains_in_train": sorted(d for d in visited_domains if d in index.domains),
            "visited_urls": len(seen),
            "visited_urls_in_train": sum(normalize_url(u) in index.urls for u in seen),
        })
    return {"tasks": rows}


# ---- loading ------------------------------------------------------------------------------------------------


def iter_infos(source: str, dirs: Iterable[str], root: Path = ROOT) -> Iterator[PageInfo]:
    for d in dirs:
        for path in sorted(glob.glob(str(root / d / "**" / "*.json.gz"), recursive=True)):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                yield page_info(source, json.load(f))


def visited_urls(logs_glob: str, root: Path = ROOT) -> dict[str, set[str]]:
    """task_id -> every URL an agent saw (the `url:` line of a tool result) in the server logs."""
    out: dict[str, set[str]] = defaultdict(set)
    for path in sorted(glob.glob(str(root / logs_glob))):
        urls: set[str] = set()
        task = None
        with open(path, encoding="utf-8") as f:
            for line in f:
                ev = json.loads(line)
                if ev.get("type") == "call":
                    urls.update(_URL_LIKE.findall(ev.get("result") or ""))
                elif ev.get("type") == "summary":
                    task = ev.get("task_id")
                    if ev.get("final_url"):
                        urls.add(ev["final_url"])
        if task:
            out[task] |= {u for u in urls if u.startswith("http")}
    return out


# ---- report -------------------------------------------------------------------------------------------------


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}%" if d else "–"


def render(reports: list[SourceReport], tasks: dict, train_counts: dict[str, tuple[int, int]],
           train_domains: int, train_apps: int) -> str:
    L: list[str] = []
    L.append("# Train/eval leakage audit\n")
    L.append("Generated by `tools/leakage_audit.py` (plan v2.1, step 1). Training = web train+dev, macOS AX, Screen2AX, "
             "DeskForge train+dev, Mind2Web train. Percentages are of eval **pages** unless stated.\n")
    L.append("## Training data\n")
    L.append("| source | pages | distinct domains / apps |\n|---|---:|---:|")
    for k, (n, s) in train_counts.items():
        L.append(f"| {k} | {n:,} | {s:,} |")
    L.append(f"\n{train_domains:,} registrable domains and {train_apps:,} app names are written to "
             "`data/leakage/heldout_domains.txt` and `data/leakage/train_apps.txt` (domains to avoid when crawling held-out sites).\n")
    L.append("## Overlap per evaluation source\n")
    L.append("Kinds: **domain** = registrable domain (eTLD+1, as `grounder.data.crawl.registrable_domain`; Mind2Web site names "
             "are matched by name, or by second-level label against web domains); **URL** = normalised exact URL (no scheme/`www.`/"
             "fragment/trailing slash, sorted query); **app** = app name (bundle-id tail / `gnome-` prefix tolerant, >=4 chars); "
             f"**near-dup** = Jaccard >= {JACCARD} of the normalised element-text sets (MinHash LSH + exact check; "
             f"only pages with >= {MIN_TEXTS} distinct texts are comparable); **cross** = near-dup whose training page is on a different domain/app.\n")
    L.append("| eval source | pages | domains/apps | domain overlap | URL overlap | app overlap | near-dup | identical set | near-dup cross-domain |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in reports:
        if not r.pages:
            L.append(f"| {r.name} | 0 | – | – | – | – | – | – | – |")
            continue
        has_dom, has_app = bool(r.domains), bool(r.apps)
        units = len(r.domains) if has_dom else len(r.apps)
        hits = len(r.overlap_domains) if has_dom else len(r.overlap_apps)
        dom_cell = f"{r.domain_pages:,} ({pct(r.domain_pages, r.pages)}); {hits}/{units} domains" if has_dom else "n/a"
        app_cell = f"{r.app_pages:,} ({pct(r.app_pages, r.pages)}); {hits}/{units} apps" if has_app else "n/a"
        near = f"{r.near_pages:,} ({pct(r.near_pages, r.comparable_pages)} of {r.comparable_pages:,})"
        L.append(f"| {r.name} | {r.pages:,} | {units:,} | {dom_cell} | {r.url_pages:,} ({pct(r.url_pages, r.pages)}) | {app_cell} | "
                 f"{near} | {r.exact_pages:,} | {r.near_cross_pages:,} |")
    L.append("")
    for r in reports:
        if not r.pages:
            continue
        if r.overlap_domains:
            L.append(f"### {r.name}: overlapping domains ({len(r.overlap_domains)} of {len(r.domains)})\n")
            L.append(", ".join(f"`{d}` ({'+'.join(sorted(s))})" for d, s in sorted(r.overlap_domains.items())[:80]))
            if len(r.overlap_domains) > 80:
                L.append(f"… and {len(r.overlap_domains) - 80} more")
            L.append("")
        if r.overlap_apps:
            L.append(f"### {r.name}: overlapping apps ({len(r.overlap_apps)} of {len(r.apps)})\n")
            L.append("; ".join(f"`{a}` ≈ {', '.join(sorted(m))}" for a, m in sorted(r.overlap_apps.items())[:80]))
            L.append("")
        if r.near_examples:
            L.append(f"### {r.name}: cross-domain near-duplicates (examples)\n")
            for u, v, j in r.near_examples:
                L.append(f"- J={j}: `{u}` ≈ `{v}`")
            L.append("")
    rows = tasks["tasks"]
    n = len(rows)
    L.append("## Agent tasks (demo/tasks.yaml + URLs visited in runs/cli*/logs)\n")
    if n:
        sd = sum(r["start_domain_in_train"] for r in rows)
        su = sum(r["start_url_in_train"] for r in rows)
        anyv = sum(bool(r["visited_domains_in_train"]) for r in rows)
        allv = sum(set(r["visited_domains"]) == set(r["visited_domains_in_train"]) for r in rows)
        vu = sum(r["visited_urls"] for r in rows)
        vut = sum(r["visited_urls_in_train"] for r in rows)
        L.append(f"- {n} tasks. Start domain in training: **{sd}/{n} ({pct(sd, n)})**; start URL (exact) in training: {su}/{n} ({pct(su, n)}).")
        L.append(f"- Every visited domain in training: {allv}/{n} ({pct(allv, n)}); at least one: {anyv}/{n} ({pct(anyv, n)}).")
        L.append(f"- Visited URLs seen by agents: {vu:,}, of which {vut:,} ({pct(vut, vu)}) are exact training URLs.\n")
        L.append("| task | start domain | in train | start URL in train | visited URLs in train |\n|---|---|:-:|:-:|---:|")
        for r in rows:
            L.append(f"| {r['id']} | {r['domain']} | {'yes' if r['start_domain_in_train'] else 'no'} | "
                     f"{'yes' if r['start_url_in_train'] else 'no'} | {r['visited_urls_in_train']}/{r['visited_urls']} |")
        not_in = sorted({r["domain"] for r in rows if not r["start_domain_in_train"]})
        L.append(f"\nTask start domains **not** in training: {', '.join(not_in) or 'none'}.\n")
    L.append("## Caveats\n")
    L.append("- Screen2AX pages are anonymised (hashed app ids, no app names), so app overlap cannot be measured for them; "
             "only content near-duplicates can catch them.")
    L.append("- Mind2Web page URLs are `mind2web://{site}`; domain overlap there means the same **site**, which is by design for "
             "test_task and test_website only if the site was also in train.")
    L.append("- Content matching ignores pages with fewer than "
             f"{MIN_TEXTS} distinct element texts (tiny app windows share generic labels such as Close/Minimize).")
    L.append("- The agent-task block only has URLs, not page content, so only domain and URL overlap apply to it.")
    return "\n".join(L) + "\n"


def run(root: Path, out_md: Path, out_dir: Path) -> None:
    index = TrainIndex()
    train_counts: dict[str, tuple[int, int]] = {}
    for name, dirs in TRAIN_SOURCES.items():
        n, units = 0, set()
        for info in iter_infos(name, dirs, root):
            index.add(info)
            n += 1
            units.add(info.domain or info.app or info.site)
        train_counts[name] = (n, len(units))
        print(f"train {name}: {n} pages", file=sys.stderr)
    reports = []
    for name, dirs in EVAL_SOURCES.items():
        rep = audit_source(name, iter_infos(name, dirs, root), index)
        reports.append(rep)
        print(f"eval {name}: {rep.pages} pages", file=sys.stderr)
    import yaml

    tasks = yaml.safe_load((root / "demo/tasks.yaml").read_text())
    visited: dict[str, set[str]] = defaultdict(set)
    for g in sorted(glob.glob(str(root / "runs/cli*/logs"))):
        for k, v in visited_urls(str(Path(g).relative_to(root) / "*.server.jsonl"), root).items():
            visited[k] |= v
    task_rep = agent_task_report(tasks, visited, index)

    web_domains = sorted(d for d, s in index.domains.items() if s - {"mind2web"} or "." in d)
    m2w_sites = sorted(d for d, s in index.domains.items() if s == {"mind2web"} and "." not in d)
    app_names = sorted({a for names in index.apps.values() for a in names})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "heldout_domains.txt").write_text("\n".join(web_domains) + "\n")
    (out_dir / "train_apps.txt").write_text("\n".join(app_names) + "\n")
    (out_dir / "train_mind2web_sites.txt").write_text("\n".join(m2w_sites) + "\n")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render(reports, task_rep, train_counts, len(web_domains), len(app_names)))
    print(f"wrote {out_md}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--out-md", type=Path, default=ROOT / "docs/leakage.md")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data/leakage")
    a = p.parse_args()
    run(a.root, a.out_md, a.out_dir)


if __name__ == "__main__":
    main()
