# grounder_tool

A compact UI-grounding tool for agents: `find(query, k=5)` and `snapshot(goal, k=20)` return a handful of
`id|role|text|context|score=` lines instead of the whole page. The ranker is one flag:

| `--ranker` | what |
|---|---|
| `exact` | exact label match first, then BM25 (default; no model files; best for agents' literal queries) |
| `bm25` | BM25 over text + context |
| `blend:<dir>:<alpha>` | model log-probability + alpha * normalised BM25 (alpha 8 by default); best on human, descriptive queries |
| `model:<dir>` | ONNX model in `<dir>` (e.g. `dist/xs-s0`); int8 if present, `--fp32` forces fp32 |
| `router:<dir>:<tau>` | lexical coverage >= tau -> exact-match + BM25, otherwise the model |

Why `exact` is the default: see `docs/results-v2.md` (agents quote labels; the neural blend did not raise agent success).

## Install

```bash
uv sync --extra tool            # mcp, playwright, rank-bm25 (plus the base onnxruntime stack)
uv run playwright install chromium
```

Installed as a package, the console script is `grounder-tool`. From a checkout use `uv run grounder-tool`.

## Python

```python
from playwright.sync_api import sync_playwright
from grounder_tool import UIGrounder, format_lines
from grounder_tool.extract import from_playwright      # afrom_playwright for async, from_macos_ax for macOS

g = UIGrounder("model:dist/xs-s0")                     # or "exact" (default), "blend:dist/xs-s0:8"
with sync_playwright() as p:
    page = p.chromium.launch().new_page()
    page.goto("https://example.com")
    hits = g.find(from_playwright(page), "more information link", k=5)
    print(format_lines(hits))                          # 0|link|Learn more||score=0.912
    goal_hits = g.snapshot(from_playwright(page), "find the docs", k=20, history=None)
```

`find` / `snapshot` accept a `records.Page` or a list of `records.Element`. Each `Hit` has `id, role, text, context,
score, box`. `from_macos_ax(app=None)` dumps an app's front window (frontmost app by default); it needs the Swift
`axdump` binary (`swift build -c release` in `grounder/extract/macos`) and the Accessibility permission.

## MCP server

Tools: `open(url)`, `find(query, k=5)`, `snapshot(goal, k=20)`, `click(id)`, `type(id, text, press_enter)`,
`go_back()`, `read_text(max_chars)`, `observe(goal="", budget=600)` returns URL, title, live headings, goal-ranked element
lines and a count footer. Live pages also include relevant text passages after the headings. An empty goal uses the page title. Budgets use one estimate,
ceil(characters / 4). Both MCP servers clamp the budget to 100-2000 tokens.
Elements are ranked in goal mode with the last three actions. Duplicate role, case-folded
text and href combinations appear once. Observe clips text to 80 characters and context to 40.

The shared DOM helper collects visible h1-h3 headings and named navigation, main, aside
and footer landmarks in document order. Names are deduplicated, clipped to 60 characters
and capped at 30 entries. The outline uses at most 15% of the budget and reports omissions
with `+N more`. Stored Pages have no landmarks, so they omit the outline.

`inspect(id=42)` returns element properties and up to ten neighbours with the same context.
`inspect(region="Install")` matches a heading prefix without regard to case and selects
controls between that heading and the next heading by document y. Without heading positions,
it falls back to context-prefix matching. The Python API accepts `landmarks` and `line_suffix`
on both methods, plus `heading_positions` on inspect.

Use `--observe-ranker blend:dist/xs-s0:8` to choose a separate observation ranker.
The default is the find ranker. In the benchmark, `find+observe@exact` uses exact for
both tools; `find+observe@exact+blend:dist/xs-s0:8` chooses them independently.
The benchmark's empty observation goal uses the task goal. Each action returns its
status line followed by a fresh 400-token digest of the new page; an action refused
for a stale or missing element list returns the error followed by the same digest,
so the ids to retry with arrive in the same call. `find+observe` also exposes
`fill(fields=[{id, text}], submit_id)`: several form fields in one call (text fields
are typed into, native selects pick the option by label), with the usual
click-safety checks and one digest for the whole call.

### Attach to an existing browser

Start Chrome with remote debugging enabled, for example:

```sh
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --remote-debugging-port=9222 --user-data-dir=/tmp/grounder-chrome
uv run grounder-tool --cdp-endpoint http://localhost:9222 --ranker exact
```

For a browser shared with Playwright MCP, give both servers the same CDP endpoint:

```sh
npx @playwright/mcp --cdp-endpoint http://localhost:9222
uv run grounder-tool --cdp-endpoint http://localhost:9222 \
  --cdp-page-url-substring example.com --observe-ranker exact
```

Attach mode selects the last page of the first context, or the last page whose URL
contains `--cdp-page-url-substring`. It registers perception tools only and re-extracts
the page on each call. Element lines append `@x,y`, the center in viewport CSS pixels
with the current scroll offset subtracted. Disconnecting leaves the browser open.
Use Playwright MCP for actions, disable automatic screenshots in your agent setup,
and call `observe` after navigation to get the compact page view.


### Read facts from page text

Use `read(query, budget=400)` to find facts on the current page. Both MCP modes and the
benchmark `find+observe` arm expose this tool. The budget is clamped to 100-2000 tokens.
Each result uses `> text  [§ heading]`, followed by `showing K of N passages`.
Only passages with positive lexical scores appear. Ties keep document order. When
fewer than three passages match, `read` also falls back to the context of the
elements lexical matching picked: their container text, plus short digit-bearing
blocks near them (prices, totals, counts).

The shared live extractor returns paragraphs, list items, definitions, table rows,
code blocks, quotations, captions and headings. It skips hidden blocks and navigation,
headers, footers and sidebars when the page has other content. Blocks contain at most
300 characters. The extractor returns at most 2,000 blocks.

Leaf blocks outside any larger passage — a `div` or `span` holding a lone short
number, for example — are also collected, so standalone prices and counts reach
the digest and `read`. `observe` keeps at most the ten digit-bearing blocks nearest
to the elements it shows, within 120 CSS pixels on the page.

Passage ranking always uses the existing exact/BM25 scorer, regardless of the element
ranker. Observe uses at most 30% of its total budget for text. Element lines fill the
rest. Action digests in `find+observe` include fresh passages. Stored Pages retain their
existing output unless the caller supplies `passages` to `UIGrounder.observe`.
