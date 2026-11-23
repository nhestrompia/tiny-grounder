# Competitor benchmark arms

The binaries are `@playwright/mcp@0.0.84` (published 2026-10-31) and
`agent-browser@0.41.2` (published 2026-11-03). Both predate the 2026-11-21
implementation by more than two weeks. The implementation checked their shipped
README and `--help`. It checks installed versions before serving tools.

Install once, then run offline:

```sh
npm install --prefix .cache/competitors --save-exact @playwright/mcp@0.0.84 agent-browser@0.41.2
uv sync --extra demo --inexact
uv run python -m tools.verify_competitors
uv run pytest -q
```

No benchmark agent is called by verification. Results are written only to
`data/bench_v3/verify_competitors.json`; call logs are under
`.cache/competitors/verification/`. Existing archive and verification files stay unchanged.

| Arm | Binary arguments | View |
|---|---|---|
| `pw-mcp` | `--cdp-endpoint ENDPOINT --timeout-settle 0` | Stock tools, snapshots and image responses. |
| `pw-mcp-compact` | `--cdp-endpoint ENDPOINT --timeout-settle 0 --snapshot-mode none` | Prompt recommends `browser_snapshot(depth=3)`, subtree snapshots, and `browser_find(text)`. Explicit snapshots still work. |
| `agent-browser` | `--config EMPTY_JSON --session UNIQUE --cdp ENDPOINT` | `snapshot -i -c`; click/fill by `@ref`, open, back, and `get text @ref`. |
| `pw-mcp+observe` | `--cdp-endpoint ENDPOINT --timeout-settle 0 --snapshot-mode none` | `find`, `observe`, `inspect`, `read` with the same exact keyword ranker and handlers as `find+observe@exact`. The middleware adapter accepts our ids as Playwright action targets and returns a fresh goal digest after actions. |

`ENDPOINT` is a loopback Chromium CDP port allocated per run. The agent-browser
configuration file contains `{}`. Its environment clears inherited `AGENT_BROWSER_*`
settings, assigns a unique namespace/socket directory, and sets
`AGENT_BROWSER_NO_VERSION_CHECK=1` to prevent the CLI's Chrome-version network check.
Successful CLI stdout is forwarded verbatim, including its final newline. CLI failures
return stdout, or stderr if stdout is empty, with MCP `isError=true`.

## Browser ownership and replay

The Python benchmark still launches Chromium, owns the context and installs the HAR
fallback routes before opening the page. External binaries attach to that browser.
The owning Playwright thread keeps dispatching events while the child runs; otherwise
HAR requests from an external CDP client can stall. All browser calls remain on that
thread. Grounder perception reads the same owned page, which the external tool drives
through CDP. It does not launch another browser.

An explicit new-tab probe found that the pinned MCP binary's new tabs can bypass the
owner's HAR routes. All new arms therefore enforce single-tab operation. `window.open`
and links with `target` navigate the existing tab; `browser_tabs(action="new")` and
isolated-context creation are rejected. Browser-run-code calls that request new pages,
contexts or browser connections are rejected too. Tool names and schemas remain stock.
This guard supports normal navigation, not adversarial JavaScript sandboxing.
Chromium also uses the existing closed-port proxy and blocked service workers. Missing
navigations get the existing local stub; other HAR misses abort. The Open MPI verification
checks a missing navigation through every arm and a redirected popup through each MCP arm.
It requires zero live responses and zero proxy-network failures.

## Proxy and measurement

A Python MCP client starts the pinned Playwright MCP stdio child. The outer server
forwards tools/list definitions and tools/call results unchanged for the two direct
Playwright arms, subject to the single-tab and element safety checks. It adds the benchmark's `done(answer)`.
Every Playwright arm omits `browser_run_code` and `browser_run_code_unsafe`, since arbitrary
code would bypass the element safety checks the arms share.
Middleware omits `browser_snapshot`, `browser_find` and `browser_take_screenshot` from
its list and strips any remaining `### Snapshot` sections from results as a fallback.
The supported snapshot-mode flag handles ordinary action responses.

Every forwarded result has a call event in the existing JSONL log, including errors,
arguments, elapsed time, character count and full MCP result. Text, structured content,
and resource descriptions count toward returned characters. Images count by decoded
bytes and pixels, using the existing `ceil(width*height/750)` token estimate. Text uses
the existing `ceil(chars/4)` estimate. These are estimates, not model tokenizer counts.
Local middleware calls and appended action digests retain `extract_ms` and `rank_ms`. Run summaries and CLI run
records include the competitor package, verified version, exact flags and replay counters.

The owner waits for load and then 500 ms only after successful page-changing calls,
matching the existing click, type and history rule. Read-only calls have no owner wait.
`demo/competitor_specs.py` classifies every tool in the pinned Playwright bundle and
all six commands exposed by our agent-browser wrapper. Tab listing is a read, while
other tab operations are actions. Arbitrary evaluation/code and unknown names are
conservative actions. Logs record `page_changing` and `unknown_tool`.
Playwright runs with `--timeout-settle 0`, so its native settle does not duplicate the
owner wait. Its native actionability, request and load handling still run.

All arms log `safety_ms`, `settle_ms`, `proxy_ms` and `harness_overhead_ms` in call
records and run summaries. Run totals include the initial owner settle, reported as
`startup_settle_ms`, which occurs before the first call. Safety timing includes refusals.
The CLI run JSON and printed summary include the same totals and
`wall_seconds_without_harness`. Raw `wall_seconds` remains unchanged. Proxy timing is
the elapsed forwarding wrapper time minus child runtime, safety, settle and appended
digest work. It includes owner queue delays, but excludes final serialization/logging.
Child runtime remains in adjusted latency. Browser/binary startup outside the measured
settle, model time, extraction and ranking remain too. These additive fields do not
change existing tool responses or action behavior.

## Interpretation

The scripted verifier uses the YAML next-link label as an oracle, selects a label/ref
actually returned by that arm, and calls its real action tool. Middleware passes the id returned by `find` directly as `browser_click(target=id)`.
Before clicking, verification checks that the stamped node is the exact oracle node,
so a same-label duplicate cannot silently substitute for it. It then calls `done`
with the reference answer. This proves interface navigation and grading, not LLM success
or answer discovery. Three FastAPI tasks share their start page; this is a small sanity
sample, not a performance estimate for all 134 tasks.

The JSON reports first-observation tokens, per-navigation tokens, total tool tokens
and calls. A navigation measurement includes its action result and, for arms without
automatic views, an explicit observation. Search results used to choose a link count in
total tool tokens, but not in that navigation-only column. Isolation probes also count
in totals, but not navigation means.

Playwright MCP sometimes returns snapshot file links for large action snapshots.
These remain stock and count as the link text the agent receives. Agents have no file
access, but can use `browser_find` or an explicit snapshot. This explains small direct
MCP navigation responses and is a material limitation when interpreting token sizes.
The CLI wrapper exposes a smaller command set than agent-browser's full shell interface;
agents cannot chain shell commands. Middleware intentionally changes Playwright's view.

## Middleware adapter

This is the adapter a real integration would ship. Extraction stamps each exact DOM
node with `data-tg="tg-DOCUMENT-NODE"`. The document prefix is eight hex characters,
and the node counter is stable for surviving nodes across extractions. These ids cannot
collide with Playwright refs such as `e5` or `f1e48`. All element lines in `observe`,
`find` and `inspect` use these ids; `inspect(id=...)` accepts them. `read` returns text
passages without element ids.

The proxy rewrites a current, uniquely resolving id to `[data-tg="ID"]` in `target`
arguments, drag `startTarget`/`endTarget`, and nested form target/ref fields. Stale ids,
unknown ids and stock refs pass through unchanged. It does not invent label selectors.
Remote tool names and schemas remain stock. Only local `inspect` extends its id input
to accept strings. Call logs retain both `args` and `forwarded_args`.

After every successful forwarded action, including DOM-only changes, the middleware
keeps Playwright's stock summary, strips snapshot sections and appends the same goal
digest format and 400-token budget as `find+observe@exact`. Longer adapter ids count
inside that budget. The combined response counts toward tool tokens. No extra observe
call is needed. Errors do not append a digest.

## Element safety and remaining differences

The refusal predicates still use `ph.blocked_text`, `ph.verdict`, disabled state,
form-submit facts and password facts. Existing arms retain their original checks.
Playwright refs use the tool's human-readable `element` argument, with corresponding
`startElement` and `endElement` descriptions for drag targets. Agent-browser uses the
label cached from its most recent forwarded snapshot. Actions invalidate this cache.
No safety call takes a snapshot, extracts the full page, calls `browser_evaluate`, or
starts an extra agent-browser CLI process.

One owner-side DOM read supplements a ref label with facts from exact-label matching
controls. It checks every matching duplicate conservatively. CSS targets and our stamped
ids resolve directly. Our ids also check their cached extracted text, but not the
extractor's wider context, so every arm gets the same heading context from the DOM read. Native
Playwright/agent-browser actions retain their own actionability checks. Logs report
`checked` when DOM facts are available, or `label checked; DOM facts unavailable`.

Precision is lower for child refs because labels do not identify an exact node across
clients. A misleading supplied description, stale snapshot, frames, shadow roots or
accessible names outside the cheap label rule can prevent DOM facts from resolving.
In those cases the blocked-word label check still runs, but submit/password/context
facts can be unavailable. Duplicate labels can cause conservative refusals. Cheap DOM
context includes the nearest section/form heading, rather than a fresh full extractor
context. There is no extra owner scroll or position probe. These are documented
coverage differences, not new refusal predicates.

Element checks cannot cover arbitrary JavaScript, global key presses, coordinate-only
mouse tools, dialog acceptance or actions without an element target. Targeted code can
act on other nodes, e.g. through `browser_evaluate`. This wrapper is not a JavaScript security sandbox.

Direct MCP actions can return inaccessible snapshot file links; their prompts tell
agents to call `browser_snapshot()` for the view. Compact MCP and agent-browser need
explicit observations after actions. The middleware automatically returns its digest.
Agent-browser exposes only the wrapper's command subset. Verification follows oracle
paths and supplies reference answers, so it establishes tool navigation and replay
isolation, not autonomous agent success.


## Earlier verification after adapter fixes

All 20 runs succeeded, with zero live requests and zero proxy-network failures.
All 48 scripted path clicks logged `safety: checked`. Token counts are estimates.

| Task | Arm | Success | Live requests | First-view tokens | Calls |
|---|---|---|---:|---:|---:|
| open-mpi-index-3 | pw-mcp | yes | 0 | 3469 | 8 |
| asciidoctor-document-attributes-ref | pw-mcp | yes | 0 | 11645 | 7 |
| fastapi-tutorial-devserver | pw-mcp | yes | 0 | 14648 | 7 |
| fastapi-async-explainer | pw-mcp | yes | 0 | 14635 | 5 |
| fastapi-deploy-vague | pw-mcp | yes | 0 | 14679 | 5 |
| open-mpi-index-3 | pw-mcp-compact | yes | 0 | 350 | 11 |
| asciidoctor-document-attributes-ref | pw-mcp-compact | yes | 0 | 228 | 11 |
| fastapi-tutorial-devserver | pw-mcp-compact | yes | 0 | 411 | 11 |
| fastapi-async-explainer | pw-mcp-compact | yes | 0 | 412 | 8 |
| fastapi-deploy-vague | pw-mcp-compact | yes | 0 | 393 | 8 |
| open-mpi-index-3 | agent-browser | yes | 0 | 1260 | 8 |
| asciidoctor-document-attributes-ref | agent-browser | yes | 0 | 1520 | 8 |
| fastapi-tutorial-devserver | agent-browser | yes | 0 | 2180 | 8 |
| fastapi-async-explainer | agent-browser | yes | 0 | 2185 | 6 |
| fastapi-deploy-vague | agent-browser | yes | 0 | 2176 | 6 |
| open-mpi-index-3 | pw-mcp+observe | yes | 0 | 571 | 9 |
| asciidoctor-document-attributes-ref | pw-mcp+observe | yes | 0 | 586 | 8 |
| fastapi-tutorial-devserver | pw-mcp+observe | yes | 0 | 583 | 8 |
| fastapi-async-explainer | pw-mcp+observe | yes | 0 | 597 | 6 |
| fastapi-deploy-vague | pw-mcp+observe | yes | 0 | 576 | 6 |

| Arm | Mean first-view tokens | Mean per-navigation tokens |
|---|---:|---:|
| pw-mcp | 11815.2 | 84.5 |
| pw-mcp-compact | 358.8 | 391.3 |
| agent-browser | 1864.2 | 2256.1 |
| pw-mcp+observe | 582.6 | 629.5 |

Middleware first-observation cost changed from 594.0 to 582.6 estimated tokens;
per-navigation cost changed from 604.4 to 629.5. Its calls per task changed as follows,
with task order matching the verifier: Open MPI 11 → 9, Asciidoctor 11 → 8,
FastAPI tutorial 11 → 8, async 8 → 6, deployment 8 → 6. The saved call is the explicit
observe after each path navigation. Open MPI totals include the isolation probes.
Longer ids and their selectors in Playwright summaries also count toward returned text.


## Latency parity verification

The verifier preserves the original logs in
`.cache/competitors/verification-before-parity/` and writes current logs to
`.cache/competitors/verification/`. The JSON contains per-tool nearest-rank p50/p95,
safety p95 and child runtime p50/p95 under `latency_before` and `latency_after`.
It also runs `find+observe@exact` on the same five frozen paths with no agent.
An absent native-arm baseline is not reconstructed. Native observations before the
change come from the local tools in the middleware logs. Sample sizes are small:
five first views and twelve clicks per arm. CLI cold starts affect snapshot tails.

All 25 runs passed: 20 competitor runs and five native runs. All competitor runs
reported zero live requests and zero network failures. Milliseconds below are
p50 / p95, with nearest-rank percentiles. A dash means no recorded baseline.

| Arm | Tool | Before, ms | After, ms | n |
|---|---|---:|---:|---:|
| agent-browser | snapshot | 590 / 699 | 77 / 180 | 17 |
| agent-browser | click | 695 / 768 | 552 / 571 | 12 |
| find+observe | observe | — | 40 / 59 | 5 |
| find+observe | find | — | 14 / 28 | 12 |
| find+observe | click | — | 530 / 629 | 12 |
| pw-mcp+observe | observe | 78 / 101 | 77 / 103 | 5 |
| pw-mcp+observe | find | 17 / 44 | 13 / 30 | 12 |
| pw-mcp+observe | browser_click | 1699 / 1800 | 654 / 820 | 12 |
| pw-mcp-compact | browser_snapshot | 546 / 587 | 40 / 82 | 17 |
| pw-mcp-compact | browser_find | 525 / 547 | 14 / 28 | 12 |
| pw-mcp-compact | browser_click | 2152 / 2280 | 593 / 680 | 12 |
| pw-mcp | browser_snapshot | 587 / 598 | 76 / 82 | 5 |
| pw-mcp | browser_click | 2195 / 2288 | 635 / 701 | 12 |
| pw-mcp | browser_find | 524 / 546 | 22 / 27 | 7 |

Action safety p95 was 4.48 ms for stock Playwright, 5.57 ms for compact,
6.61 ms for middleware and 5.08 ms for agent-browser. Native safety p95 was
1.66 ms. All meet the 20 ms target on this sample. Forwarded read latency matches
the measured child runtime to rounding, with no owner settle.

Click median differences versus native were 105 ms for stock Playwright,
63 ms for compact, 124 ms for middleware and 22 ms for agent-browser. Thus compact
and agent-browser meet the roughly 100 ms goal, stock is near it, and middleware
exceeds it. The remaining difference comes from the binary's native locator,
request/load and response handling, plus middleware extraction/ranking for the
appended digest. Middleware click p95 remains 191 ms above native. These costs
remain in adjusted latency. The former duplicated binary and owner settle is gone.

The prior logs lacked safety and child-runtime timers. Their zero-valued timer
columns in the JSON mean unmeasured, not zero overhead. Calls not exercised by
the scripted paths, such as console/network reads and get_text, have classification
and wrapper coverage but no latency sample here.

## Pre-registered full comparison (2026-11-21, before any full-run result)

- **Run:** 134 milestone tasks × 4 arms × 4 agents (Claude Haiku 5.5, Claude Sonnet 5.5, GPT-6 Luna, GPT-6.1 Terra),
  one run each, arm order shuffled per task (seeded). Output: `runs/bench-v4`.
- **Arms:**
  - `find+observe@exact` (B, tiny-grounder standalone);
  - `pw-mcp+observe` (M, tiny-grounder as the view for Playwright MCP);
  - `pw-mcp-compact`;
  - `agent-browser`.

  Default `pw-mcp` was dropped after the pilot. It was the worst arm, and it is not anyone's best configuration.
- **Analysis:** pooled over agents and paired by (task, agent), with a site-cluster bootstrap (4,000 draws, seed 0,
  95% percentile CIs). These are the same methods as the Plan v3 milestone.
- **Primary comparisons:** B vs `pw-mcp-compact` and B vs `agent-browser`. "Beats the existing compact tools" is
  claimed only if B passes **both**.
- **Middleware comparison:** M vs `pw-mcp-compact`. This tests whether tiny-grounder improves a Playwright MCP
  workflow as a drop-in view.
- **Gates, per comparison:**
  - success Δ, lower CI > −10 pp;
  - total input tokens per success ratio, upper CI < 1.0;
  - median paired wall-time ratio, upper CI ≤ 1.10.
- **Secondary (no gates):** tool-output tokens, uncached input, Claude cost per success, wall time per success, p90
  latency, tool calls, harness overhead (safety, settle, proxy), per-agent tables, and harness failures (runs without
  a server summary are reported and excluded pairwise).
- The pilot tasks (`runs/bench-v4-pilot`) are part of the 134 and are rerun; pilot runs are not pooled in.
