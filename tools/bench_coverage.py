"""Report per-task snapshot sizes and pilot top-level archive-miss coverage.

uv run python -m tools.bench_coverage --logs runs/bench-v3-pilot/logs
Coverage counts unique (task, defragmented URL) pairs, not repeated snapshot calls.
A covered URL has a successful HTML or plain-text response or a complete archived redirect chain.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlsplit

from tools.bench_v2 import ROOT, har_path, load_raw, pilot_misses
from demo.browser import parse_tasks
from grounder.data.crawl import registrable_domain


def archived_pages(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        har = json.loads(archive.read(next(n for n in archive.namelist() if n.endswith('.har'))))
    pages = {}
    for entry in har['log']['entries']:
        response = entry['response']
        url = urldefrag(entry['request']['url'])[0]
        if entry['request']['method'] == 'GET':
            pages[url] = response
    return pages


def covered(url: str, pages: dict, seen: frozenset = frozenset()) -> bool:
    url = urldefrag(url)[0]
    if url in seen or url not in pages:
        return False
    response = pages[url]
    status = response['status']
    if 300 <= status < 400:
        target = response.get('redirectURL') or next((h['value'] for h in response.get('headers', [])
                                                    if h['name'].lower() == 'location'), '')
        return bool(target) and covered(urljoin(url, target), pages, seen | {url})
    mime = response.get('content', {}).get('mimeType', '').lower()
    return 200 <= status < 300 and ('html' in mime or mime.startswith('text/plain'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks-file', type=Path, default=ROOT / 'demo/tasks_milestone.yaml')
    parser.add_argument('--logs', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'data/bench_v3/coverage.json')
    args = parser.parse_args()
    tasks = parse_tasks(load_raw(args.tasks_file))
    report = {'definition': 'unique same-site (task, defragmented pilot miss URL) pairs; successful HTML/plain text or complete redirects',
              'tasks': {}}
    total = hits = size = 0
    for task in tasks:
        path = har_path(task)
        pages = archived_pages(path)
        misses = [u for u in pilot_misses(args.logs, task.id)
                  if urlsplit(u).scheme in {'http', 'https'} and registrable_domain(u) == registrable_domain(task.start_url)]
        coverage = [{'url': u, 'archived': covered(u, pages)} for u in misses]
        total += len(coverage)
        hits += sum(c['archived'] for c in coverage)
        size += path.stat().st_size
        report['tasks'][task.id] = {'har': task.har, 'bytes': path.stat().st_size,
                                  'document_urls': sum(covered(u, pages) for u in pages), 'pilot_misses': coverage}
    report.update(total_bytes=size, pilot_miss_urls=total, archived_miss_urls=hits,
                  miss_coverage_rate=hits / total if total else None)
    args.out.write_text(json.dumps(report, indent=1) + '\n')
    lines = ["# Snapshot coverage", "", f"Total: {size:,} bytes across {len(tasks)} snapshots.", "",
             f"Pilot same-site miss coverage: {hits}/{total} unique task/URL pairs.", "",
             "| Task | Bytes | MiB | Document URLs including redirects | Pilot URLs archived |",
             "|---|---:|---:|---:|---:|"]
    for task_id, row in report['tasks'].items():
        misses = row['pilot_misses']
        count = sum(m['archived'] for m in misses)
        lines.append(f"| {task_id} | {row['bytes']:,} | {row['bytes'] / 2**20:.2f} | {row['document_urls']} | {count}/{len(misses)} |")
    lines.extend(["", "## Pilot URLs", "", "| Task | URL | Archived |", "|---|---|---|"])
    for task_id, row in report['tasks'].items():
        for miss in row['pilot_misses']:
            lines.append(f"| {task_id} | {miss['url']} | {'yes' if miss['archived'] else 'no'} |")
    args.out.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'tasks'}, indent=1))


if __name__ == '__main__':
    main()
