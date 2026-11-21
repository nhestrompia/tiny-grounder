# Benchmark v4 hard slice — frozen shop/app tasks (Plan v4 step 2)

`demo/tasks_v4.yaml` — 21 frozen tasks on 5 non-docs sites. v2/v3 were all
developer docs where keyword matching suffices; this slice targets the cases
where picking the right element is hard.

## What is in it

| Site | Tasks | What it adds |
|------|-------|--------------|
| saucedemo.com | 8 | Login-gated shop; icon-only cart link and burger menu; sort `<select>`; end-to-end checkout (up to 13 steps) |
| demoblaze.com | 3 | Store with API-driven category pages (read-only tasks only — see below) |
| tutorialsninja.com/demo | 5 | OpenCart store: icon-only search/cart/wishlist buttons, cart flow |
| the-internet.herokuapp.com | 5 | App controls: forms, dynamic content, keyboard, DOM state |

- **Long sessions:** 6 tasks run 8–15 steps (checkouts + remove-from-cart +
  login/logout/re-login) so re-sent snapshots pile up — the failure mode the
  114k-token Playwright MCP figure comes from.
- **Icon-only controls:** 10 tasks tagged `icon` (cart icon, burger menu,
  search magnifier, wishlist heart).
- **Vague goals:** 10 tasks tagged `vague` ("buy the cheapest item",
  "take one back out").
- **Tags:** same set as v2/v3 plus `app` (app-style pages).
- **New step type:** `select: {locator, option: {label|value|index}}` in
  `tools/bench_v2.py::run_steps` for `<select>` controls (saucedemo sort).

## What is deliberately not in it

- **demoblaze cart/checkout flows**: the cart lives server-side behind POSTs
  whose bodies carry a per-session cookie, so the recorded archive cannot
  serve replays (Playwright HAR matching includes POST data). Demoblaze tasks
  are read-only browsing/fact tasks.
- **demoqa.com bookstore**: the SPA's book-detail route renders an empty page
  under HAR replay even when fully archived (client-side routing); dropped.
- **Dialog-dependent flows** are kept minimal — Playwright auto-dismisses
  `alert()` but cart state still needs the server.

## Freeze and verify

```bash
uv run python -m tools.bench_v2 record   --tasks-file demo/tasks_v4.yaml --out data/bench_v4 --branches 0
uv run python -m tools.bench_v2 verify   --tasks-file demo/tasks_v4.yaml --out data/bench_v4
uv run python -m tools.bench_v2 manifest --tasks-file demo/tasks_v4.yaml --out data/bench_v4
```

Current state: **21/21 verified** (all four scripted checks per task, zero
live requests; residual misses are analytics pixels only). Freeze hashes in
`data/bench_v4/manifest.json`. `tests/test_bench_v4_tasks.py` asserts the
slice properties (≥5 sessions of 8+ steps, ≥5 `icon` and `vague` tags,
domains disjoint from v2/v3 docs and from training domains).

To fold into the step-1 competitor run, add `--tasks-file demo/tasks_v4.yaml`
to the same `demo.cli_bench` invocation (or merge files) — arms, agents and
analysis stay identical.
