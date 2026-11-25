"""Evaluate the real observe digest, with paired site/seed bootstrap intervals.

Full run used (from the repository root):
    uv run python tools/next_step_recall.py --set mind2web_task --set mind2web_website \
      --set mind2web_domain --set bench-v2 --budgets 300 600 1200 \
      --rankers exact bm25 xs m 'blend[xs]' 'blend[m]' --device cpu --threads 4 --out runs/nextstep

Named Mind2Web sets expand to name|examples.jsonl|pages directories; arbitrary
sets use that same pipe-separated syntax. --model FAMILY=path,path,path overrides
checkpoint families. Alpha=8 is fixed, never fitted here. Stored pages supply no
landmarks. Budgets use observe's ceil(characters/4) estimate, not a tokenizer.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grounder.baselines.blend import blend
from grounder.data.crawl import registrable_domain
from grounder.data.pages import load_examples, load_page, write_examples, write_page
from grounder.ranking import Ranked, positive_ids
from grounder.records import Element, Example, action_repr, equiv_ids
from grounder_tool.digest import observe, estimate_tokens
from grounder_tool.rankers import make_ranker
from tools.compare_rankers import EvalSet, group_labels, load_pages_for, paired_bootstrap
from tools.harvest_queries import map_rows, parse_action, read_run


class FixedRanker:
    """Feed precomputed scores through observe without repeating neural inference."""
    name = 'precomputed'

    def __init__(self, ranked: Ranked):
        self.ranked = ranked

    def rank(self, query, elements, mode='goal', history=None):
        return self.ranked


def listed_ids(text: str) -> list[int]:
    return [int(m.group(1)) for m in re.finditer(r'^(\d+)\|', text, re.M)]


def hit(ids: list[int], example: Example) -> bool:
    return bool(set(ids) & positive_ids(example))


def build_set(spec: str) -> tuple[EvalSet, dict]:
    name, examples_path, directories = spec.split('|')
    raw = load_examples(examples_path)
    examples = [replace(e, mode='goal') for e in raw if e.target_id is not None and e.kind != 'null']
    pages = load_pages_for(directories.split(), {e.page_id for e in examples})
    kept = [e for e in examples if e.page_id in pages]
    return EvalSet(name, pages, kept), {'raw': len(raw), 'null': len(raw)-len(examples),
                                      'missing_pages': len(examples)-len(kept)}


def parse_steps(events: list[dict], start_url: str) -> list[dict]:
    """Capture the URL BEFORE each action; action result URLs describe the next page.

    Unparseable coordinate actions are retained as drops, since no reliable target
    label is present. Failed/refused actions are also counted but not used as labels.
    History includes successful earlier click/type actions, even if recovery fails.
    """
    url, history, steps = start_url, [], []
    history_indices = []
    tags = {'a': 'link', 'input': 'textbox', 'textarea': 'textbox', 'select': 'combobox'}
    for ev in events:
        result = ev.get('result') or ''
        if ev.get('name') in ('click', 'type', 'click_xy', 'type_text'):
            parsed = None if ev.get('is_error') else parse_action(result)
            step = {'url': url, 'history': list(history), 'action': parsed, 'args': ev.get('args', {}), 'history_indices': list(history_indices)}
            steps.append(step)
            if parsed:
                verb, role, text = parsed
                element = Element(0, tags.get(role, role), text, tag=role if role in tags else {'link': 'a', 'textbox': 'input', 'searchbox': 'input'}.get(role, ''))
                history_indices.append(len(steps)-1)
                history.append(action_repr(element, step['args'].get('text', '') if verb == 'type' else None))
            elif not ev.get('is_error'):
                # A coordinate action happened, but its label is unknowable from this log.
                history_indices.append(len(steps)-1)
                history.append('[unknown]  -> ' + ('TYPE: ' + ev.get('args', {}).get('text', '')
                                                 if ev['name'] in ('type', 'type_text') else 'CLICK'))
        urls = re.findall(r'^url:\s*(\S+)', result, re.M)
        if urls:
            url = urls[-1]
    return steps


def build_bench(runs_dir: str = 'runs/bench-v2', tasks_file: str = 'demo/tasks_v2.yaml',
                name: str | None = None, root: Path = ROOT) -> tuple[EvalSet, dict]:
    """Recover next-step labels from a benchmark run's server logs.

    Only successful runs yield labels, and only click/type steps whose target element is
    recoverable from the frozen page — the elements an agent actually used, not its failures.
    """
    from demo.browser import BrowserSession, load_tasks

    name = name or Path(runs_dir).name
    tasks = {t.id: t for t in load_tasks(root / tasks_file)}
    run_rows = [json.loads(l) for l in (root / runs_dir / 'runs.jsonl').read_text().splitlines() if l.strip()]
    runs = {}
    for r in run_rows:
        key = (r['agent'].replace(':', '-'), re.sub(r'[^a-zA-Z0-9_-]', '-', r.get('arm', r['config'])), r['task_id'], str(r['repeat']))
        runs[key] = r
    stats = Counter()
    pages, examples, seen, cached = {}, [], set(), {}
    failures, provenance = [], []
    cache_dir = root / 'data/nextstep/pages'
    for path in sorted((root / runs_dir / 'logs').glob('*.server.jsonl')):
        key = tuple(path.name.removesuffix('.server.jsonl').split('__'))
        run = runs.get(key)
        if not run or not run.get('success'):
            continue
        events, summary = read_run(str(path))
        task = tasks[run['task_id']]
        stats['successful_runs'] += 1
        resolved_history = {}
        for index, step in enumerate(parse_steps(events, task.start_url)):
            stats['action_steps'] += 1
            reason = ''
            if not step['action']:
                reason = 'unlabelled_or_failed_action'
            else:
                url = step['url']
                # Key pages by archive as well as task+URL: two benchmarks can share task ids
                # and URLs while freezing different page contents.
                har = getattr(task, 'har', '') or f'data/bench_v2/har/{task.id}.zip'
                pid = 'next-' + hashlib.sha256(f'{har}\0{task.id}\0{url}'.encode()).hexdigest()[:20]
                if pid not in cached:
                    matches = list(cache_dir.glob(f'*/{pid}.json.gz'))
                    try:
                        if matches:
                            page = load_page(matches[0])
                        else:
                            with BrowserSession(url, har=root / har) as browser:
                                page = browser.extract()
                                if not page.elements or page.title.lower() == 'page not archived':
                                    raise ValueError('empty/unarchived page')
                            page.page_id, page.site = pid, task.site or run.get('site', '') or registrable_domain(url)
                            write_page(page, cache_dir)
                        cached[pid] = page
                        print(f'bench page {len(cached)}: {task.id} {url}', flush=True)
                    except Exception as exc:
                        cached[pid] = None
                        failures.append({'task': task.id, 'url': url, 'error': str(exc)})
                page = cached[pid]
                if page is None:
                    reason = 'page_unrecoverable'
                else:
                    verb, role, text = step['action']
                    role = {'a': 'link', 'input': 'textbox', 'textarea': 'textbox', 'select': 'combobox'}.get(role, role)
                    ids = map_rows(page.elements, [{'role': role, 'text': text}], by_context=False)
                    if not ids:
                        reason = 'element_unrecoverable'
                    else:
                        # Recover the server's tag-based history from the actual extracted element.
                        element = next(e for e in page.elements if e.id == ids[0])
                        resolved_history[index] = action_repr(element, step['args'].get('text', '') if verb == 'type' else None)
                        # map_rows deliberately allows text-only fallback; retain all its matches.
                        targets = sorted(set(ids) | {j for i in ids for j in equiv_ids(page.elements, i)})
                        ident = (task.id, url, tuple(targets))
                        stats['recoverable_steps'] += 1
                        if ident in seen:
                            stats['duplicates'] += 1
                            continue
                        seen.add(ident)
                        pages[pid] = page
                        examples.append(Example(pid, 'goal', 'goal', summary.get('goal') or task.goal,
                                                [resolved_history.get(i, h) for i, h in zip(step['history_indices'], step['history'])],
                                                targets[0], targets, task.id))
                        provenance.append({'log': path.name, 'step': index, 'task': task.id, 'url': url,
                                           'page_id': pid, 'target_ids': targets})
            if reason:
                stats['dropped_steps'] += 1
                stats[reason] += 1
    stats['usable_unique'] = len(examples)
    target = root / 'data/nextstep'
    target.mkdir(parents=True, exist_ok=True)
    write_examples(examples, target / f'{name}.jsonl')
    (target / f'{name}-build.json').write_text(json.dumps({'counts': dict(stats), 'page_failures': failures,
                                                          'provenance': provenance}, indent=2))
    return EvalSet(name, pages, examples), dict(stats)


def rankings(s: EvalSet, ckpt: str, out: Path, device: str) -> list[Ranked]:
    from grounder.baselines.model import ModelRanker

    fingerprint = hashlib.sha256()
    fingerprint.update(Path(ckpt).read_bytes())
    for e in s.examples:
        fingerprint.update(json.dumps(e.to_dict(), sort_keys=True).encode())
        fingerprint.update(json.dumps(s.pages[e.page_id].to_dict(), sort_keys=True).encode())
    cache = out / 'cache' / f'{fingerprint.hexdigest()}.pkl'
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    result = ModelRanker(ckpt, device).rank_all(s.pages, s.examples)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(result))
    return result


def evaluate(s: EvalSet, models: dict, budgets: list[int], names: list[str], out: Path,
             device: str, group_by: str, draws: int) -> dict:
    if not s.examples:
        raise ValueError(f'{s.name}: no examples')
    ranked, families = {}, {}
    lexical = {}
    for name in ('exact', 'bm25'):
        ranker = make_ranker(name)
        lexical[name] = [ranker.rank(e.query or s.pages[e.page_id].title, s.pages[e.page_id].elements,
                                     'goal', e.history) for e in s.examples]
        if name in names:
            ranked[name], families[name] = [lexical[name]], None
    for family, paths in models.items():
        if family not in names and f'blend[{family}]' not in names:
            continue
        seeds = []
        for path in paths:
            seeds.append(rankings(s, path, out, device))
            print(f'{s.name}: {family} {path} ranked', flush=True)
        if family in names:
            ranked[family], families[family] = seeds, family
        name = f'blend[{family}]'
        if name in names:
            ranked[name] = [[blend(m, b, 8.0) for m, b in zip(seed, lexical['bm25'])] for seed in seeds]
            families[name] = family
    groups = group_labels(s, group_by)
    metrics, counts = {}, {}
    for name, seeds in ranked.items():
        values = {f'recall@{b}': [] for b in budgets} | {f'recall@k{k}': [] for k in (5, 10, 20)}
        counts[name] = {str(b): [] for b in budgets}
        for seed in seeds:
            cur = {m: [] for m in values}
            for e, r in zip(s.examples, seed):
                for b in budgets:
                    text = observe(s.pages[e.page_id], e.query, FixedRanker(r), budget=b, history=e.history)
                    assert estimate_tokens(text) <= b
                    ids = listed_ids(text)
                    cur[f'recall@{b}'].append(hit(ids, e))
                    counts[name][str(b)].append(len(ids))
                for k in (5, 10, 20):
                    cur[f'recall@k{k}'].append(hit(r.ids[:k], e))
            for m in values:
                values[m].append(cur[m])
        metrics[name] = {m: np.array(v, dtype=bool) for m, v in values.items()}
    result = {'n': len(s.examples), 'n_groups': len(set(groups)), 'rankers': {n: {} for n in ranked}}
    for metric in next(iter(metrics.values())):
        # Binary synthetic ranks expose arbitrary recall via the imported top1 bootstrap.
        # A fixed all-miss reference gives the absolute recall CI with the same draws.
        arrays = {n: np.where(v[metric], 1, 10**9) for n, v in metrics.items()}
        arrays['_zero'] = np.full((1, len(s.examples)), 10**9)
        pairs = [(n, '_zero') for n in ranked]
        pairs += [(n, b) for n in ranked for b in ('exact', 'bm25') if b in ranked and n != b]
        pairs += [(a, b) for a, b in [('xs', 'm'), ('blend[xs]', 'blend[m]')] if a in ranked and b in ranked]
        boot = paired_bootstrap(arrays, families, groups, pairs, draws=draws)
        for n in ranked:
            row = dict(boot[n, '_zero']['top1'])
            row['mean'] = row.pop('diff')
            row['vs'] = {b: boot[a, b]['top1'] for a, b in pairs if a == n and b != '_zero'}
            if metric.startswith('recall@') and metric[7:].isdigit():
                row['mean_k'] = float(np.mean(counts[n][metric[7:]]))
            result['rankers'][n][metric] = row
    return result


def fmt(v: dict, delta: bool = False) -> str:
    key = 'diff' if delta else 'mean'
    return f"{100*v[key]:.1f} [{100*v['lo']:.1f}, {100*v['hi']:.1f}]"


def markdown(report: dict) -> str:
    lines = ['# Next-step recall', '',
             'Recall and paired deltas in percentage points [95% CI]. Site/source clusters and neural seeds '
             f"resampled with compare_rankers.paired_bootstrap, {report['setup']['draws']:,} draws, seed 0. Alpha fixed at 8.", '',
             'The real observe formatter is used without landmarks. Tokens = ceil(characters/4). Hits use target plus '
             'equivalent ids; null examples are excluded. Recall@k uses the raw ranking without a token budget. '
             'Neural point estimates average three seeds; family seeds are shared with their blends within each draw.', '',
             'Caveats: trajectories contain self-selected actions from successful runs; coordinate-only and failed '
             'actions are dropped. Role-only log histories use recovered element tags where possible and inferred '
             'HTML tags otherwise. map_rows allows text-only fallback and accepts all matching targets. '
             'Mind2Web test_domain shares 16/53 domains with training; see docs/leakage.md.', '']
    for name, s in report['sets'].items():
        lines += [f"## {name}: {s['n']} examples, {s['n_groups']} groups", '',
                  '| Ranker | Metric | Recall [CI] | Mean K | Δ vs exact [CI] | Δ vs BM25 [CI] | Δ XS vs M [CI] |',
                  '|---|---|---|---|---|---|---|']
        for ranker, metrics in s['rankers'].items():
            for metric, v in metrics.items():
                other = 'm' if ranker == 'xs' else 'blend[m]'
                delta = lambda base: fmt(v['vs'][base], True) if base in v['vs'] else '-'
                shown = f"{v['mean_k']:.1f}" if 'mean_k' in v else '-'
                lines.append(f"| {ranker} | {metric} | {fmt(v)} | {shown} | "
                             f"{delta('exact')} | {delta('bm25')} | {delta(other) if ranker in ('xs', 'blend[xs]') else '-'} |")
        lines.append('')
    lines += ['## Data recovery', '', '```json', json.dumps(report['data'], indent=2), '```', '',
              f"Runtime: {report['runtime_seconds']:.1f} seconds. CPU deterministic inference; all checkpoints retained.", '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--set', action='append', required=True)
    parser.add_argument('--budgets', nargs='+', type=int, default=[300, 600, 1200])
    parser.add_argument('--rankers', nargs='+', default=['exact', 'bm25', 'xs', 'm', 'blend[xs]', 'blend[m]'])
    parser.add_argument('--model', action='append', default=[])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--group-by', choices=['page_site', 'source'], default='page_site')
    parser.add_argument('--draws', type=int, default=2000)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    models = {f: [f'runs/train/{f}-s{i}/model.pt' for i in range(3)] for f in ('xs', 'm')}
    for spec in args.model:
        family, paths = spec.split('=', 1)
        models[family] = paths.split(',')
    allowed = {'exact', 'bm25', *models, *(f'blend[{f}]' for f in models)}
    if set(args.rankers) - allowed:
        parser.error('unknown rankers: ' + str(set(args.rankers)-allowed))
    args.out.mkdir(parents=True, exist_ok=True)
    report = {'setup': vars(args) | {'out': str(args.out), 'models': models, 'alpha': 8}, 'sets': {}, 'data': {}}
    for spec in args.set:
        if spec == 'bench-v2':
            s, stats = build_bench()
        elif spec.startswith('bench:'):
            # bench:<runs_dir>|<tasks_file>[|<name>] — any benchmark log dir becomes a label source.
            parts = spec.removeprefix('bench:').split('|')
            if len(parts) < 2:
                parser.error('bench: sets need bench:<runs_dir>|<tasks_file>[|<name>]')
            s, stats = build_bench(parts[0], parts[1], parts[2] if len(parts) > 2 else None)
        else:
            if '|' not in spec:
                split = spec.removeprefix('mind2web_')
                spec = f'mind2web_{split}|data/mind2web/test_{split}.jsonl|data/mind2web/pages/test_{split}'
            s, stats = build_set(spec)
        print(f'{s.name}: {len(s.examples)} examples; {stats}', flush=True)
        report['data'][s.name] = stats
        report['sets'][s.name] = evaluate(s, models, args.budgets, args.rankers, args.out,
                                         args.device, args.group_by, args.draws)
        report['runtime_seconds'] = time.perf_counter()-started
        (args.out / 'report.json').write_text(json.dumps(report, indent=2))
        (args.out / 'report.md').write_text(markdown(report))
    print(f"Finished in {report['runtime_seconds']:.1f}s", flush=True)


if __name__ == '__main__':
    main()
