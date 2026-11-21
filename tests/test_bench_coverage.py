from tools.bench_v2 import page_url, pilot_misses, neighbour_links
from tools.bench_coverage import covered


def test_page_candidates_allow_subdomains_and_skip_unsafe_files():
    assert page_url('https://docs.example.org/guide/#intro', 'example.org') == 'https://docs.example.org/guide/'
    for url in ('https://other.org/guide', 'mailto:x@example.org', 'https://example.org/a.tar.xz', 'https://example.org/archive.tgz',
                'https://example.org/logout?next=/', 'https://example.org/submit', 'https://example.org/a.woff2', 'https://example.org/schedule?form=donate',
                'https://example.org/cdn-cgi/content?id=x'):
        assert page_url(url, 'example.org') is None


def test_neighbours_keep_page_order_and_remove_duplicates():
    class Page:
        url = 'https://example.org/'

        def evaluate(self, script):
            return ['https://example.org/#x', 'https://docs.example.org/a#x', 'https://docs.example.org/a#y',
                    'https://other.org/b', 'https://example.org/a.pdf', 'https://example.org/b']
    assert neighbour_links(Page(), 0) == []
    assert neighbour_links(Page(), 2) == ['https://docs.example.org/a', 'https://example.org/b']


def test_pilot_misses_are_task_scoped_and_deduplicated(tmp_path):
    import json
    rows = [{'result': 'url: https://example.org/a#x\ntitle: page not archived'},
            {'result': 'clicked link\nurl: https://example.org/a#y\ntitle: page not archived'},
            {'result': 'url: https://example.org/b\ntitle: Normal page'}]
    (tmp_path / 'agent__arm__task__0.server.jsonl').write_text('\n'.join(map(json.dumps, rows)))
    (tmp_path / 'agent__arm__other__0.server.jsonl').write_text(json.dumps(rows[0]))
    assert pilot_misses(tmp_path, 'task') == ['https://example.org/a']


def test_coverage_requires_html_or_complete_redirect_chain():
    pages = {'https://x.org/a': {'status': 301, 'redirectURL': '/b'},
             'https://x.org/b': {'status': 200, 'content': {'mimeType': 'text/html'}},
             'https://x.org/c': {'status': 404, 'content': {'mimeType': 'text/html'}},
             'https://x.org/d': {'status': 301, 'redirectURL': '/missing'},
             'https://x.org/e': {'status': 200, 'content': {'mimeType': 'application/pdf'}},
             'https://x.org/loop': {'status': 301, 'redirectURL': '/loop'}}
    assert covered('https://x.org/a#x', pages)
    for suffix in ('c', 'd', 'e', 'loop', 'missing'):
        assert not covered('https://x.org/' + suffix, pages)


def test_a11y_oracle_supports_menu_buttons_and_nested_link_text(tmp_path):
    import json
    from tools.verify_tasks_v3 import verify

    home = "https://oracle.test/"
    bodies = {
        home: '<button onclick="document.getElementById(\'links\').hidden=false">Menu</button>'
              '<div id="links" hidden><a href="/next"><span>Next</span></a></div>',
        home + 'next': '<h1>The answer is 42</h1>',
    }
    entries = []
    for url, body in bodies.items():
        entries.append({
            'startedDateTime': '2026-01-01T00:00:00.000Z', 'time': 0,
            'request': {'method': 'GET', 'url': url, 'httpVersion': 'HTTP/1.1', 'headers': [],
                        'queryString': [], 'cookies': [], 'headersSize': -1, 'bodySize': 0},
            'response': {'status': 200, 'statusText': 'OK', 'httpVersion': 'HTTP/1.1',
                         'headers': [{'name': 'Content-Type', 'value': 'text/html'}], 'cookies': [],
                         'content': {'size': len(body), 'mimeType': 'text/html', 'text': body},
                         'redirectURL': '', 'headersSize': -1, 'bodySize': len(body)},
            'cache': {}, 'timings': {'send': 0, 'wait': 0, 'receive': 0},
        })
    har = tmp_path / 'oracle.har'
    har.write_text(json.dumps({'log': {'version': '1.2', 'creator': {'name': 'test', 'version': '1'},
                                     'entries': entries}}))
    result = verify({'id': 'oracle', 'start_url': home, 'goal': 'Find the answer', 'max_steps': 5,
                     'check': {'url_contains': '/next'}, 'answer_regex': '42', 'answer': '42', 'har': str(har),
                     'path': [{'click': {'role': 'button', 'name': 'Menu'}},
                              {'click': {'text': 'Next', 'exact': True}}]})
    assert result['ok'], result
    assert len(result['refs']) == 2
