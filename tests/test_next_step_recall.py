"""Next-step evaluation uses actual digest ids, goal history and pre-action URLs."""
import json

import numpy as np

from grounder.data.pages import write_examples, write_page
from grounder.ranking import Ranked
from grounder.records import Element, Example, Page
from grounder_tool.digest import observe
import tools.next_step_recall as next_step
from tools.next_step_recall import (
    PRIMARY_METRIC, FixedRanker, build_set, distinct_ids, evaluate, gate_report, hit,
    hit_at_distinct, listed_ids, parse_steps, whole_page_fits,
)
from tools.compare_rankers import EvalSet


def page():
    return Page('p', 'https://example.org/', 'example.org', 'Title', [1280, 800], [1280, 800],
                [Element(1, 'link', 'First'), Element(2, 'link', 'Next'), Element(3, 'link', 'Next')])


def test_build_set(tmp_path):
    p = page()
    write_page(p, tmp_path / 'pages')
    e = Example('p', 'goal', 'goal', 'Find next', ['[link] First -> CLICK'], 2, [2, 3])
    write_examples([e, Example('p', 'null', 'do', 'absent'),
                    Example('missing', 'goal', 'goal', 'Find', target_id=1)], tmp_path / 'examples.jsonl')
    s, stats = build_set(f'tiny|{tmp_path}/examples.jsonl|{tmp_path}/pages')
    assert s.examples == [e]
    assert stats == {'raw': 3, 'null': 1, 'missing_pages': 1}


def test_hit_equivalent_and_complete_digest_lines():
    e = Example('p', 'goal', 'goal', 'Next', target_id=3, equiv_ids=[2, 3])
    text = observe(page(), e.query, FixedRanker(Ranked([2, 3, 1], [1., .5, 0.], 0.)), budget=300)
    assert listed_ids(text) == [2, 1]  # duplicate labels deduplicated by observe
    assert hit(listed_ids(text), e)
    assert not hit([1], e)
    assert not hit([], e)
    assert listed_ids('url: 42|link|Next\ntitle: 3|link|Next\n2|link|Next|') == [2]


def test_bench_synthetic_jsonl(tmp_path):
    rows = [
        {'type': 'call', 'name': 'snapshot', 'result': 'url: https://example.org/\ntitle: Home'},
        {'type': 'call', 'name': 'click', 'args': {'id': 2},
         'result': "clicked link 'Next'\nurl: https://example.org/next"},
        {'type': 'call', 'name': 'type', 'args': {'id': 3, 'text': 'abc'},
         'result': "typed into textbox 'Search' and pressed Enter\nurl: https://example.org/results"},
        {'type': 'call', 'name': 'click', 'args': {'id': 4}, 'is_error': True, 'result': 'refused'},
    ]
    path = tmp_path / 'tiny.server.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    from tools.harvest_queries import read_run
    events, _ = read_run(str(path))
    steps = parse_steps(events, 'https://example.org/')
    assert steps[0]['url'] == 'https://example.org/'
    assert steps[1]['url'] == 'https://example.org/next'
    assert steps[1]['history'] == ['[a]  Next -> CLICK']
    assert steps[2]['history'][-1] == '[input]  Search -> TYPE: abc'
    assert steps[2]['action'] is None
    assert steps[2]['url'] == 'https://example.org/results'


def test_tag_actions_and_coordinate_drops():
    steps = parse_steps([
        {'name': 'click', 'result': "clicked a 'Next'\nurl: https://example.org/next"},
        {'name': 'click_xy', 'result': 'clicked at (1, 2)'},
        {'name': 'type', 'result': "typed into input 'Query'", 'args': {'text': 'x'}},
    ], 'https://example.org/')
    assert steps[1]['action'] is None
    assert steps[1]['history'] == ['[a]  Next -> CLICK']
    assert steps[2]['history'][-1] == '[unknown]  -> CLICK'


def test_bootstrap_recall_and_k(tmp_path):
    p = page()
    s = EvalSet('tiny', {'p': p}, [Example('p', 'goal', 'goal', 'Next', target_id=3, equiv_ids=[2, 3])])
    result = evaluate(s, {}, [300], ['exact', 'bm25'], tmp_path, 'cpu', 'page_site', 20)
    row = result['rankers']['exact']['recall@300']
    assert row['mean'] == row['lo'] == row['hi'] == 1
    assert row['mean_k'] == 2
    assert np.isclose(row['vs']['bm25']['diff'], 0)


def test_recall_at3_collapses_equivalent_ids_before_counting():
    elements = [Element(1, 'link', 'Same'), Element(2, 'link', 'Same'),
                Element(3, 'button', 'Other'), Element(4, 'button', 'Target')]
    ranked = Ranked([1, 2, 3, 4], [1., .9, .8, .7], 0.)
    example = Example('p', 'goal', 'goal', 'Target', target_id=4, equiv_ids=[4])
    assert distinct_ids(ranked.ids, elements) == [1, 3, 4]
    assert hit_at_distinct(ranked, example, elements, 3)
    assert not hit(ranked.ids[:3], example)


def test_whole_page_fits_sizes_complete_digest_and_deduplicates():
    tiny = Page('p', 'https://e/', 'e', 'T', [1, 1], [1, 1], [Element(1, 'link', 'Next')])
    assert whole_page_fits(tiny, 29)
    assert not whole_page_fits(tiny, 28)

    duplicates = [Element(i, 'link', 'Same', 'context') for i in range(80)]
    many = Page('p', 'https://e/', 'e', 'T', [1, 1], [1, 1], duplicates)
    assert whole_page_fits(many, 300)

    distinct = [Element(i, 'link', f'Label {i}', 'x' * 80) for i in range(20)]
    large = Page('p', 'https://e/', 'e', 'T', [1, 1], [1, 1], distinct)
    assert not whole_page_fits(large, 300)


def test_task_clusters_are_primary_and_site_clusters_secondary(monkeypatch, tmp_path):
    pages = {
        'p1': Page('p1', 'https://same/', 'same.site', 'One', [1, 1], [1, 1], [Element(1, 'link', 'Next')]),
        'p2': Page('p2', 'https://same/2', 'same.site', 'Two', [1, 1], [1, 1], [Element(1, 'link', 'Next')]),
    }
    examples = [Example('p1', 'goal', 'goal', 'Next', target_id=1, equiv_ids=[1], source='task-a'),
                Example('p2', 'goal', 'goal', 'Next', target_id=1, equiv_ids=[1], source='task-b')]
    s = EvalSet('tiny', pages, examples)
    real = next_step.paired_bootstrap
    calls = []

    def spy(arrays, families, groups, pairs, **kwargs):
        calls.append(list(groups))
        return real(arrays, families, groups, pairs, **kwargs)

    monkeypatch.setattr(next_step, 'paired_bootstrap', spy)
    result = evaluate(s, {}, [300], ['exact', 'bm25'], tmp_path, 'cpu', draws=2)
    assert calls[0] == ['task-a', 'task-b']
    assert ['same.site', 'same.site'] in calls
    assert result['n_groups'] == 2
    assert result['n_groups_page_site'] == 1
    assert result['primary_cluster'] == 'source'
    json.dumps(result)


def gate_rows(blend_lo=.01, literal_diff=-.005):
    overall = {'rankers': {
        'exact': {PRIMARY_METRIC: {'mean': .50}},
        'bm25': {PRIMARY_METRIC: {'mean': .60}},
        'blend[xs]': {PRIMARY_METRIC: {'mean': .70, 'vs': {
            'exact': {'diff': .20, 'lo': .05, 'hi': .30},
            'bm25': {'diff': .10, 'lo': blend_lo, 'hi': .20},
        }}},
    }}
    literal_exact = .80
    tags = {'literal': {'n': 2, 'rankers': {
        'exact': {PRIMARY_METRIC: {'mean': literal_exact}},
        'blend[xs]': {PRIMARY_METRIC: {'mean': literal_exact + literal_diff}},
    }}}
    return overall, tags


def test_gate_logic_uses_best_lexical_ci_and_literal_point_estimate():
    gate = gate_report(*gate_rows())
    assert gate['pass']
    assert gate['blend_minus_max_lexical']['baseline'] == 'bm25'
    assert gate['blend_minus_max_lexical']['pass']
    assert gate['literal_blend_minus_exact']['pass']

    assert not gate_report(*gate_rows(blend_lo=0.0))['pass']
    assert not gate_report(*gate_rows(literal_diff=-.02))['pass']
