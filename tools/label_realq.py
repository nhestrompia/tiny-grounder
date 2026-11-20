"""Label the real-query candidates (data/realq/candidates.jsonl) -> data/realq/labels.jsonl.

Interactive (default): shows each unlabelled query with its task goal, the run outcomes and the candidate list.
Answer with element ids (comma separated: several are accepted as equivalent targets), `n` = no matching element on
the page (a null example), `s` = skip, `q` = quit; add `?` to flag the query as ambiguous.

    uv run python tools/label_realq.py                       # interactive
    uv run python tools/label_realq.py show 0 20             # print items 0..19 compactly (no prompts)
    uv run python tools/label_realq.py grep 17 login         # elements of item 17's page matching a word
    uv run python tools/label_realq.py apply labels.txt      # bulk: lines `INDEX: 3,5 [?] # note`, `INDEX: none`, `INDEX: skip`

The target is judged against the goal and the page. The element the agent clicked is shown as a hint only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REALQ = ROOT / "data" / "realq"
_SRC = re.compile(r"bm25#(\d+)")


def load_candidates(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def load_labels(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    labels = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            lab = json.loads(line)
            labels[lab["key"]] = lab
    return labels


def save_label(path: Path, item: dict, target_ids: list[int], null: bool, ambiguous: bool, skip: bool, note: str,
               labeller: str) -> dict:
    lab = {"key": item["key"], "query": item["query"], "url": item["url"], "target_ids": target_ids, "null": null,
           "ambiguous": ambiguous, "skip": skip, "note": note, "labeller": labeller}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(lab, ensure_ascii=False) + "\n")
    return lab


def short_src(sources: list[str]) -> str:
    out = []
    for s in sources:
        m = _SRC.fullmatch(s)
        out.append(f"b{m.group(1)}" if m else {"agent-click": "CLICK", "agent-type": "TYPE", "agent-saw": "saw", "text-match": "txt"}.get(s, s))
    return ",".join(out)


def render(idx: int, item: dict, width: int = 70, max_cands: int = 12) -> str:
    runs = item["outcomes"]
    ok = sum(r["success"] for r in runs)
    lines = [f"#{idx} Q: {item['query']!r}  [{item['url'][:90]}]  runs {ok}/{len(runs)} ok, {item['n_elements']} els",
             f"   goal: {' || '.join(g[:160] for g in item['goals'])}"]
    if item["clicked"]:
        lines.append("   agent next: " + "; ".join(f"{c['verb']} {c['role']} {c['text'][:50]!r}" for c in item["clicked"][:3]))
    for c in item["candidates"][:max_cands]:
        ctx = f" ~{c['context'][:40]}" if c["context"] else ""
        lines.append(f"   {c['id']}|{c['role']}|{c['text'][:width]}{ctx} <{short_src(c['sources'])}>")
    return "\n".join(lines)


def render_compact(idx: int, item: dict, prev_goal: str | None, max_cands: int = 5) -> tuple[str, str]:
    """Dense form for bulk labelling: goal only when it changes, one short line per candidate."""
    goal = " || ".join(g[:110] for g in item["goals"])
    lines = [f"## goal: {goal}"] if goal != prev_goal else []
    ok = sum(r["success"] for r in item["outcomes"])
    click = "; ".join(f"{c['text'][:30]!r}" for c in item["clicked"][:2])
    lines.append(f"#{idx} {item['query']!r} ({ok}/{len(item['outcomes'])}ok n={item['n_elements']}) click:{click}")
    for c in item["candidates"][:max_cands]:
        mark = "*" if any(s.startswith("agent-") and s != "agent-saw" for s in c["sources"]) else ""
        ctx = f"~{c['context'][:22]}" if c["context"] else ""
        lines.append(f" {c['id']}{mark}|{c['role'][:4]}|{c['text'][:42]}{ctx}")
    return "\n".join(lines), goal


def grep_page(item: dict, word: str, pages_dir: Path = REALQ / "pages") -> list[str]:
    from grounder.data.pages import load_pages

    page = load_pages(pages_dir)[item["page_id"]]
    w = word.casefold()
    return [f"{e.id}|{e.role}|{e.text[:70]}~{e.context[:40]}" for e in page.elements if w in f"{e.text} {e.context} {e.href}".casefold()]


def parse_spec(spec: str) -> tuple[list[int], bool, bool, bool, str]:
    """`3,5 ? # note` / `none` / `skip` -> (ids, null, ambiguous, skip, note)."""
    spec, _, note = spec.partition("#")
    spec = spec.strip()
    ambiguous = "?" in spec
    spec = spec.replace("?", "").strip().casefold()
    if spec in ("s", "skip"):
        return [], False, ambiguous, True, note.strip()
    if spec in ("n", "none", "null"):
        return [], True, ambiguous, False, note.strip()
    ids = [int(x) for x in re.split(r"[,\s]+", spec) if x]
    if not ids:
        raise ValueError("empty label")
    return ids, False, ambiguous, False, note.strip()


def page_ids(pages_dir: Path) -> dict[str, set[int]]:
    from grounder.data.pages import load_pages

    return {pid: {e.id for e in p.elements} for pid, p in load_pages(pages_dir).items()}


def apply(path: Path, items: list[dict], labels_path: Path, labeller: str, pages_dir: Path | None = None) -> int:
    """Bulk-apply `INDEX: spec` lines. Ids must be elements of the item's page (not only listed candidates)."""
    ids_by_page = page_ids(pages_dir) if pages_dir else {}
    n = 0
    for lineno, line in enumerate(open(path, encoding="utf-8"), 1):
        line = line.strip()
        if not line or line.startswith("##"):
            continue
        idx_s, _, spec = line.partition(":")
        idx = int(idx_s)
        ids, null, amb, skip, note = parse_spec(spec)
        item = items[idx]
        valid = ids_by_page.get(item["page_id"]) or {c["id"] for c in item["candidates"]}
        if not set(ids) <= valid:
            raise ValueError(f"line {lineno}: ids {sorted(set(ids) - valid)} are not elements of item {idx}'s page")
        save_label(labels_path, item, ids, null, amb, skip, note, labeller)
        n += 1
    return n


def interactive(items: list[dict], labels_path: Path, labeller: str) -> None:
    done = load_labels(labels_path)
    for idx, item in enumerate(items):
        if item["key"] in done:
            continue
        print("\n" + render(idx, item))
        while True:
            ans = input("ids | n(one) | s(kip) | q(uit)  [add ? if ambiguous] > ").strip()
            if ans.casefold() == "q":
                return
            try:
                ids, null, amb, skip, note = parse_spec(ans)
            except ValueError as exc:
                print(exc)
                continue
            valid = {c["id"] for c in item["candidates"]}
            if not set(ids) <= valid:
                print(f"not candidates: {sorted(set(ids) - valid)}")
                continue
            save_label(labels_path, item, ids, null, amb, skip, note, labeller)
            break


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("cmd", nargs="?", default="label", choices=["label", "show", "grep", "apply", "stats"])
    p.add_argument("args", nargs="*")
    p.add_argument("--dir", type=Path, default=REALQ)
    p.add_argument("--labeller", default="human")
    a = p.parse_args()
    items = load_candidates(a.dir / "candidates.jsonl")
    labels_path = a.dir / "labels.jsonl"
    if a.cmd == "label":
        interactive(items, labels_path, a.labeller)
    elif a.cmd == "show":
        start, count = int(a.args[0]), int(a.args[1])
        prev = None
        for i in range(start, min(start + count, len(items))):
            text, prev = render_compact(i, items[i], prev)
            print(text)
    elif a.cmd == "grep":
        print("\n".join(grep_page(items[int(a.args[0])], a.args[1], a.dir / "pages")))
    elif a.cmd == "apply":
        print(f"applied {apply(Path(a.args[0]), items, labels_path, a.labeller, a.dir / "pages")} labels")
    else:
        labs = load_labels(labels_path)
        print(f"{len(labs)}/{len(items)} labelled; null {sum(l['null'] for l in labs.values())}; "
              f"ambiguous {sum(l['ambiguous'] for l in labs.values())}; skipped {sum(l['skip'] for l in labs.values())}")


if __name__ == "__main__":
    main()
