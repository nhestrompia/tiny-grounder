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
`go_back()`, `read_text(max_chars)`. Ids refer to the latest find/snapshot. Risky clicks (pay, delete, form submit,
password fields) are refused. Flags: `--ranker`, `--fp32`, `--threads N`, `--headed`.

Claude Code:

```bash
claude mcp add grounder -- uv run --directory /path/to/tiny-grounder grounder-tool --ranker exact
# user-wide: add `-s user` before the name; headed browser: append --headed
```

Codex:

```bash
codex mcp add grounder -- uv run --directory /path/to/tiny-grounder grounder-tool --ranker exact
```

or in `~/.codex/config.toml`:

```toml
[mcp_servers.grounder]
command = "uv"
args = ["run", "--directory", "/path/to/tiny-grounder", "grounder-tool", "--ranker", "model:dist/xs-s0"]
```

## Numbers

`uv run python tools/bench_tool.py` (latency of `find` including extraction, and package sizes against the gates:
p95 <= 150 ms on 500 elements, package <= 3 MB).
