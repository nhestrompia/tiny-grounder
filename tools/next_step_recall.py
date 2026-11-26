"""Evaluate the real observe digest, with paired task/seed bootstrap intervals.

The primary metric is recall@3 over distinct observe-equivalent elements. Budget recall and
site/app-clustered intervals remain secondary reports, as fixed by docs/model-uniqueness-prereg.md.

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
from grounder.records import Element, Example, Page, action_repr, equiv_ids, equiv_key
from grounder_tool.digest import observe, estimate_tokens, _clip, _line
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
    action_ordinal = 0
    tags = {'a': 'link', 'input': 'textbox', 'textarea': 'textbox', 'select': 'combobox'}
    for ev in events:
        result = ev.get('result') or ''
        if ev.get('name') in ('click', 'type', 'fill', 'go_back', 'click_xy', 'type_text', 'scroll'):
            action_ordinal += 1
        if ev.get('name') in ('click', 'type', 'click_xy', 'type_text'):
            parsed = None if ev.get('is_error') else parse_action(result)
            step = {'url': url, 'history': list(history), 'action': parsed, 'args': ev.get('args', {}),
                    'history_indices': list(history_indices), 'name': ev.get('name'), 'call_n': ev.get('n'),
                    'is_error': bool(ev.get('is_error')), 'action_step': action_ordinal}
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
                name: str | None = None, root: Path = ROOT, out_root: Path | None = None,
                cache_dir: Path | None = None) -> tuple[EvalSet, dict]:
    """Recover next-step labels from a benchmark run's server logs.

    Only successful runs yield labels, and only click/type steps whose target element is
    recoverable from the frozen page — the elements an agent actually used, not its failures.
    """
    from demo.browser import BrowserSession, load_tasks

    out_root = out_root or root
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
    cache_dir = cache_dir or (out_root / 'data/nextstep/pages')
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
                        provenance.append({'log': path.name, 'run_id': path.name.removesuffix('.server.jsonl'),
                                           'step': step.get('action_step', index), 'call_n': step.get('call_n'), 'task': task.id, 'url': url,
                                           'page_id': pid, 'target_ids': targets, 'action': verb,
                                           'type_text': step['args'].get('text', '') if verb == 'type' else '',
                                           'agent': run.get('agent', ''), 'arm': run.get('arm', run.get('config', '')),
                                           'run_success': bool(run.get('success')), 'site': page.site})
            if reason:
                stats['dropped_steps'] += 1
                stats[reason] += 1
    stats['usable_unique'] = len(examples)
    target = out_root / 'data/nextstep'
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


PRIMARY_METRIC = 'recall@3'
STRUCTURAL_BUDGET = 300
TAG_ORDER = ('literal', 'goal-only', 'icon', 'untagged')


def distinct_ids(ids: list[int], elements: list[Element]) -> list[int]:
    """Ranked ids with observe's equivalence deduplication, preserving rank order."""
    by_id = {e.id: e for e in elements}
    seen = set()
    out = []
    for identifier in ids:
        key = equiv_key(by_id[identifier])
        if key in seen:
            continue
        seen.add(key)
        out.append(identifier)
    return out


def hit_at_distinct(ranked: Ranked, example: Example, elements: list[Element], k: int = 3) -> bool:
    return hit(distinct_ids(ranked.ids, elements)[:k], example)


def whole_page_fits(page: Page, budget: int = STRUCTURAL_BUDGET) -> bool:
    """Ranker-free structural check: can observe fit every distinct element line plus header/footer?

    This reproduces observe's no-landmark, no-passage formatting. Duplicate targets collapse by `equiv_key`;
    the first extraction-order representative of each equivalence class supplies the line.
    """
    cap = max(0, budget) * 4
    total = len(page.elements)

    def footer(k: int) -> str:
        return f"showing {k} of {total} elements; call find(query) for others or inspect(id) for detail"

    header_cap = max(0, cap - len(footer(0)) - 1)
    if header_cap < len("url: \ntitle: "):
        return not page.elements
    url_cap = min(200, (header_cap - 13) // 2)
    title_cap = min(120, header_cap - 13 - url_cap)
    lines = [f"url: {_clip(page.url, url_cap)}", f"title: {_clip(page.title, title_cap)}"]
    seen = set()
    distinct = []
    for element in page.elements:
        key = equiv_key(element)
        if key in seen:
            continue
        seen.add(key)
        distinct.append(element)
    element_lines = [_line(element, None) for element in distinct]
    digest = "\n".join([*lines, *element_lines, footer(len(element_lines))])
    return len(digest) <= cap


def metric_order(budgets: list[int]) -> list[str]:
    return [PRIMARY_METRIC, *(f'recall@{b}' for b in budgets), *(f'recall@k{k}' for k in (5, 10, 20))]


def measure(s: EvalSet, ranked: dict[str, list[list[Ranked]]], budgets: list[int]) -> tuple[dict, dict]:
    """Binary per-example hit arrays for every ranker/seed, plus digest element counts."""
    metrics, counts = {}, {}
    names = metric_order(budgets)
    for name, seeds in ranked.items():
        values = {m: [] for m in names}
        counts[name] = {str(b): [] for b in budgets}
        for seed in seeds:
            cur = {m: [] for m in names}
            cur_counts = {str(b): [] for b in budgets}
            for e, r in zip(s.examples, seed):
                page = s.pages[e.page_id]
                cur[PRIMARY_METRIC].append(hit_at_distinct(r, e, page.elements, 3))
                for b in budgets:
                    text = observe(page, e.query, FixedRanker(r), budget=b, history=e.history)
                    assert estimate_tokens(text) <= b
                    ids = listed_ids(text)
                    cur[f'recall@{b}'].append(hit(ids, e))
                    cur_counts[str(b)].append(len(ids))
                for k in (5, 10, 20):
                    cur[f'recall@k{k}'].append(hit(r.ids[:k], e))
            for m in values:
                values[m].append(cur[m])
            for b in cur_counts:
                counts[name][b].append(cur_counts[b])
        metrics[name] = {m: np.array(v, dtype=bool) for m, v in values.items()}
        counts[name] = {b: np.array(v, dtype=int) for b, v in counts[name].items()}
    return metrics, counts


def bootstrap_pairs(names: list[str]) -> list[tuple[str, str]]:
    pairs = [(n, '_zero') for n in names]
    pairs += [(n, b) for n in names for b in ('exact', 'bm25') if b in names and n != b]
    pairs += [(a, b) for a, b in [('xs', 'm'), ('blend[xs]', 'blend[m]')] if a in names and b in names]
    return pairs


def slice_report(metrics: dict, counts: dict, families: dict, groups: np.ndarray, mask: np.ndarray,
                 budgets: list[int], draws: int, metric_names: list[str] | None = None) -> dict:
    """Absolute and paired bootstrap rows for one slice of examples."""
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    sliced_groups = groups[mask]
    result = {'n': n, 'n_groups': len(set(sliced_groups)) if n else 0, 'rankers': {n_: {} for n_ in metrics}}
    if not n:
        return result
    metric_names = metric_names or metric_order(budgets)
    budget_metrics = {f'recall@{b}' for b in budgets}
    names = list(metrics)
    pairs = bootstrap_pairs(names)
    for metric in metric_names:
        # Binary synthetic ranks expose arbitrary recall via the imported top1 bootstrap.
        # A fixed all-miss reference gives the absolute recall CI with the same draws.
        arrays = {n_: np.where(v[metric][:, mask], 1, 10**9) for n_, v in metrics.items()}
        arrays['_zero'] = np.full((1, len(sliced_groups)), 10**9)
        boot = paired_bootstrap(arrays, families, sliced_groups, pairs, draws=draws)
        for name in names:
            row = dict(boot[name, '_zero']['top1'])
            row['mean'] = row.pop('diff')
            row['vs'] = {b: boot[a, b]['top1'] for a, b in pairs if a == name and b != '_zero'}
            if metric in budget_metrics:
                row['mean_k'] = float(np.mean(counts[name][metric.removeprefix('recall@')][:, mask]))
            result['rankers'][name][metric] = row
    return result


def tag_masks(examples: list[Example]) -> dict[str, np.ndarray]:
    tags = [(e.tags or []) for e in examples]
    out = {tag: np.array([tag in row for row in tags], dtype=bool) for tag in TAG_ORDER[:-1]}
    out['untagged'] = np.array([not row for row in tags], dtype=bool)
    return out


def gate_report(overall: dict, tags: dict) -> dict:
    """The amended pre-registered gate: recall@3 overall CI plus the literal point-estimate check."""
    rows = overall.get('rankers', {})
    required = {'blend[xs]', 'exact', 'bm25'}
    missing = sorted(required - set(rows))
    if missing or any(PRIMARY_METRIC not in rows.get(n, {}) for n in required - set(missing)):
        return {'pass': False, 'metric': PRIMARY_METRIC, 'computable': False, 'missing': missing}
    baseline = max(('exact', 'bm25'), key=lambda n: rows[n][PRIMARY_METRIC]['mean'])
    primary = rows['blend[xs]'][PRIMARY_METRIC]['vs'][baseline]
    literal = tags.get('literal', {'n': 0, 'rankers': {}})
    literal_rows = literal.get('rankers', {})
    literal_diff = None
    if literal.get('n') and 'blend[xs]' in literal_rows and 'exact' in literal_rows:
        literal_diff = (literal_rows['blend[xs]'][PRIMARY_METRIC]['mean']
                        - literal_rows['exact'][PRIMARY_METRIC]['mean'])
    first_pass = primary['lo'] > 0
    second_pass = literal_diff is not None and literal_diff >= -0.01
    return {
        'pass': bool(first_pass and second_pass),
        'metric': PRIMARY_METRIC,
        'computable': True,
        'blend_minus_max_lexical': {
            'baseline': baseline,
            'baseline_mean': rows[baseline][PRIMARY_METRIC]['mean'],
            'blend_mean': rows['blend[xs]'][PRIMARY_METRIC]['mean'],
            'diff': primary['diff'], 'lo': primary['lo'], 'hi': primary['hi'],
            'pass': bool(first_pass),
        },
        'literal_blend_minus_exact': {
            'n': literal.get('n', 0),
            'diff': literal_diff,
            'threshold': -0.01,
            'pass': bool(second_pass),
        },
    }


def evaluate(s: EvalSet, models: dict, budgets: list[int], names: list[str], out: Path,
             device: str, group_by: str = 'source', draws: int = 2000) -> dict:
    if not s.examples:
        raise ValueError(f'{s.name}: no examples')
    if STRUCTURAL_BUDGET not in budgets:
        raise ValueError(f'{s.name}: budget {STRUCTURAL_BUDGET} is required for the discriminating-items report')
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
    metrics, counts = measure(s, ranked, budgets)
    all_mask = np.ones(len(s.examples), dtype=bool)
    task_groups = group_labels(s, 'source')
    site_groups = group_labels(s, 'page_site')
    names_order = metric_order(budgets)
    overall_task = slice_report(metrics, counts, families, task_groups, all_mask, budgets, draws, names_order)
    overall_site = slice_report(metrics, counts, families, site_groups, all_mask, budgets, draws, names_order)

    fits = np.array([whole_page_fits(s.pages[e.page_id], STRUCTURAL_BUDGET) for e in s.examples], dtype=bool)
    discriminating = slice_report(metrics, counts, families, task_groups, ~fits, budgets, draws,
                                  [f'recall@{STRUCTURAL_BUDGET}'])
    tag_results = {
        tag: slice_report(metrics, counts, families, task_groups, mask, budgets, draws, names_order)
        for tag, mask in tag_masks(s.examples).items()
    }
    structural_items = [
        {'index': i, 'source': e.source, 'page_id': e.page_id, 'target_id': e.target_id, 'fits': bool(fit)}
        for i, (e, fit) in enumerate(zip(s.examples, fits))
    ]
    result = {
        'n': len(s.examples),
        'n_groups': len(set(task_groups)),
        'n_groups_page_site': len(set(site_groups)),
        'primary_cluster': 'source',
        'secondary_cluster': 'page_site',
        'rankers': overall_task['rankers'],
        'rankers_page_site': overall_site['rankers'],
        'structural': {
            'budget': STRUCTURAL_BUDGET,
            'n_fitting': int(fits.sum()),
            'n_discriminating': int((~fits).sum()),
            'items': structural_items,
        },
        'discriminating': discriminating,
        'tags': tag_results,
    }
    result['gate'] = gate_report({'rankers': result['rankers']}, tag_results)
    return result


def fmt(v: dict, delta: bool = False) -> str:
    key = 'diff' if delta else 'mean'
    return f"{100*v[key]:.1f} [{100*v['lo']:.1f}, {100*v['hi']:.1f}]"


def metric_label(metric: str) -> str:
    if metric == PRIMARY_METRIC:
        return 'recall@3 distinct'
    if metric.startswith('recall@k'):
        return 'raw recall@' + metric.removeprefix('recall@k')
    return metric


def gate_line(gate: dict) -> str:
    if not gate.get('computable'):
        return '**Gate: FAIL** — not computable for the selected arms.'
    first = gate['blend_minus_max_lexical']
    second = gate['literal_blend_minus_exact']
    first_text = (f"recall@3 blend[xs] − max(exact, bm25) = {100*first['diff']:+.1f} pp "
                  f"[{100*first['lo']:+.1f}, {100*first['hi']:+.1f}] against {first['baseline']}")
    if second['diff'] is None:
        second_text = 'literal recall@3 blend[xs] − exact not computable (no literal items)'
    else:
        second_text = (f"literal recall@3 blend[xs] − exact = {100*second['diff']:+.1f} pp "
                       f"(n={second['n']}, threshold −1.0 pp)")
    status = 'PASS' if gate['pass'] else 'FAIL'
    return f"**Gate: {status}** — {first_text}; {second_text}."


def markdown(report: dict) -> str:
    lines = ['# Next-step recall', '',
             'Recall and paired deltas in percentage points [95% CI]. The primary metric is recall@3: the target or an '
             'equivalent id within the first 3 distinct elements after observe-style `equiv_key` deduplication. '
             'Task (`Example.source`) clusters and neural seeds are resampled with compare_rankers.paired_bootstrap, '
             f"{report['setup']['draws']:,} draws, seed 0. Site/app (`page_site`) intervals are secondary. Alpha fixed at 8.", '',
             'The real observe formatter is used without landmarks. Tokens = ceil(characters/4). Budget hits use target plus '
             'equivalent ids; null examples are excluded. `recall@300`, `recall@600` and `recall@1200` are token-budget '
             'digests; raw recall@5/10/20 uses the unbudgeted ranking as before. Neural point estimates average three seeds; '
             'family seeds are shared with their blends within each draw.', '',
             'The structural discriminating-items check runs no ranker: it sizes every distinct element line plus the observe '
             'header/footer at B=300. Per-tag rows use the automatic `literal`, `goal-only` and `icon` metadata; tags can '
             'overlap, and examples with none of the stored tags are shown as untagged.', '',
             'Caveats: trajectories contain self-selected actions from successful runs; coordinate-only and failed '
             'actions are dropped. Role-only log histories use recovered element tags where possible and inferred '
             'HTML tags otherwise. map_rows allows text-only fallback and accepts all matching targets. '
             'Mind2Web test_domain shares 16/53 domains with training; see docs/leakage.md.', '']
    for name, s in report['sets'].items():
        lines += [f"## {name}: {s['n']} examples, {s['n_groups']} task clusters ({s['n_groups_page_site']} site/app clusters)", '',
                  gate_line(s['gate']), '',
                  '### Overall metrics, task-clustered intervals', '',
                  '| Ranker | Metric | Recall [CI] | Mean K | Δ vs exact [CI] | Δ vs BM25 [CI] | Δ XS vs M [CI] |',
                  '|---|---|---|---|---|---|---|']
        for ranker, metrics in s['rankers'].items():
            for metric, v in metrics.items():
                other = 'm' if ranker == 'xs' else 'blend[m]'
                delta = lambda base: fmt(v['vs'][base], True) if base in v['vs'] else '-'
                shown = f"{v['mean_k']:.1f}" if 'mean_k' in v else '-'
                lines.append(f"| {ranker} | {metric_label(metric)} | {fmt(v)} | {shown} | "
                             f"{delta('exact')} | {delta('bm25')} | {delta(other) if ranker in ('xs', 'blend[xs]') else '-'} |")
        lines += ['', '### Secondary site/app-clustered absolute intervals', '',
                  '| Ranker | Metric | Recall [page_site CI] |', '|---|---|---|']
        for ranker, metrics in s['rankers_page_site'].items():
            for metric, v in metrics.items():
                lines.append(f"| {ranker} | {metric_label(metric)} | {fmt(v)} |")
        structural = s['structural']
        discriminating = s['discriminating']
        lines += ['', '### Discriminating items at B=300', '',
                  f"Whole page fits the B=300 digest for {structural['n_fitting']} of {s['n']} items; "
                  f"the non-fitting discriminating slice has {structural['n_discriminating']} items.", '',
                  '| Ranker | B=300 recall on non-fitting items [task CI] |', '|---|---|---|']
        for ranker, metrics in discriminating['rankers'].items():
            v = metrics.get(f"recall@{STRUCTURAL_BUDGET}")
            lines.append(f"| {ranker} | {fmt(v) if v else '-'} |")
        lines += ['', '### Per-tag breakdown, task-clustered intervals', '']
        budget_metrics = [m for m in next(iter(s['rankers'].values())) if 'mean_k' in next(iter(s['rankers'].values()))[m]]
        header = '| Tag | N | Ranker | recall@3 distinct [CI] | ' + ' | '.join(f'{m} [CI]' for m in budget_metrics) + ' |'
        lines += [header, '|---|---|---|---|' + '---|' * len(budget_metrics)]
        for tag, data in s['tags'].items():
            if not data['n']:
                lines.append(f"| {tag} | 0 | - | - | " + ' | '.join('-' for _ in budget_metrics) + ' |')
                continue
            for ranker, metrics in data['rankers'].items():
                cells = [fmt(metrics[PRIMARY_METRIC])]
                cells += [fmt(metrics[m]) for m in budget_metrics]
                lines.append(f"| {tag} | {data['n']} | {ranker} | " + ' | '.join(cells) + ' |')
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
    parser.add_argument('--draws', type=int, default=2000)
    args = parser.parse_args()
    args.budgets = sorted(set(args.budgets) | {300, 600, 1200})
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
                                         args.device, draws=args.draws)
        report['runtime_seconds'] = time.perf_counter()-started
        (args.out / 'report.json').write_text(json.dumps(report, indent=2))
        (args.out / 'report.md').write_text(markdown(report))
    print(f"Finished in {report['runtime_seconds']:.1f}s", flush=True)


if __name__ == '__main__':
    main()
