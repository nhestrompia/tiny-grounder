"""Next-step evaluation uses actual digest ids, goal history and pre-action URLs."""
import json

import numpy as np

from grounder.data.pages import write_examples, write_page
from grounder.ranking import Ranked
from grounder.records import Element, Example, Page
from grounder_tool.digest import observe
from tools.next_step_recall import FixedRanker, build_set, hit, listed_ids, parse_steps, evaluate
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
