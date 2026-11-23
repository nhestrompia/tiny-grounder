# Benchmark v4 hard slice — frozen shop/app tasks (Plan v4 step 2)

`demo/tasks_v4.yaml` — 23 frozen tasks on 7 non-docs sites. v2/v3 were all
developer docs where keyword matching suffices; this slice targets the cases
where picking the right element is hard.

## What is in it

| Site | Tasks | What it adds |
|------|-------|--------------|
| saucedemo.com | 8 | Login-gated shop; icon-only cart link and burger menu; sort `<select>`; checkout to order overview (up to 13 steps) |
| demoblaze.com | 3 | Store with API-driven category pages (read-only tasks only — see below) |
| tutorialsninja.com/demo | 5 | OpenCart store: icon-only search/cart/wishlist buttons, cart flow |
| the-internet.herokuapp.com | 4 | App controls: login, forms, dynamic content, keyboard |
| blazedemo.com | 2 | Real flight-booking flow: destination `<select>`s, results table, multi-field purchase form (14 steps) |
| news.ycombinator.com | 1 | Real news aggregator: front page to a comments thread |

- **Long sessions:** 7 tasks run 8–14 steps (the five checkouts,
  remove-from-cart, and the flight booking) so re-sent snapshots pile up — the failure mode the
  114k-token Playwright MCP figure comes from.
- **Icon-only controls:** 10 tasks tagged `icon` (cart icon, burger menu,
  search magnifier, wishlist heart).
- **Vague goals:** 14 tasks tagged `vague` ("buy the cheapest item",
  "take one back out", "report the rent").
- **Tags:** same set as v2/v3 plus `app` (app-style pages).
- **New step type:** `select: {locator, option: {label|value|index}}` in
  `tools/bench_v2.py::run_steps` for `<select>` controls (saucedemo sort).
- **Row-pinning goals where POST bodies are matched:** `blazedemo-book-flight`
  names the flight (Virgin America, 1:43 AM) because the archived purchase POST
  only replays for the recorded row — any other row navigates to a page the
  archive lacks.
- **`allow_actions: true`** on the 11 tasks whose flow needs a login, a form
  submit, or a blocked-word control (checkout/remove). It relaxes the
  click-safety refusals (blocked words, form submits, password fields) for
  that run and swaps the agent's "never log in" prompt line for a
  demo-site-actions-permitted one. Off by default.
- **Checks verify the basket, not the banner:** saucedemo checkout paths end
  on the `checkout-step-two` overview where the cart contents are listed, so
  `text_contains` requires the actual purchased item — an empty or wrong
  basket can't pass.
- **Price questions grade the answer, not the route:** a price read off a
  listing page counts the same as one read off the product page, so question
  tasks accept either page. Tutorialsninja prices accept the displayed
  (tax-inclusive) price as well as the ex-tax one.

## What is deliberately not in it

- **demoblaze cart/checkout flows**: the cart lives server-side behind POSTs
  whose bodies carry a per-session cookie, so the recorded archive cannot
  serve replays (Playwright HAR matching includes POST data). Demoblaze tasks
  are read-only browsing/fact tasks.
- **demoqa.com bookstore**: the SPA's book-detail route renders an empty page
  under HAR replay even when fully archived (client-side routing); dropped.
- **amazon.com / ebay.com / etsy.com / gymshark.com**: probed and dropped —
  bot interstitials (amazon `_sec/verify`, ebay 0.15 MB captcha page),
  hydration/API-driven pages that replay incompletely, or search APIs whose
  rotating request params defeat HAR matching (craigslist: verified once,
  then its results fetch started missing — dropped). allbirds.com (Shopify)
  verified on freeze but now intercepts every click for headless browsers
  (bot wall) so it can no longer be recorded or played — replaced with
  news.ycombinator.com, which renders server-side and replays cleanly.
- **Dialog-dependent flows** are kept minimal — Playwright auto-dismisses
  `alert()` but cart state still needs the server.
- **the-internet login–logout–login cycle**: the end state is identical to
  `internet-login`'s, so the cycle itself is unverifiable by a final-page
  check; dropped.

## Freeze and verify

```bash
uv run python -m tools.bench_v2 record   --tasks-file demo/tasks_v4.yaml --out data/bench_v4 --branches 0
uv run python -m tools.bench_v2 verify   --tasks-file demo/tasks_v4.yaml --out data/bench_v4
uv run python -m tools.bench_v2 manifest --tasks-file demo/tasks_v4.yaml --out data/bench_v4
```

Current state: **23/23 verified** (all four scripted checks per task, zero
live requests; residual misses are analytics pixels only). Freeze hashes in
`data/bench_v4/manifest.json`. `tests/test_bench_v4_tasks.py` asserts the
slice properties (≥5 sessions of 8+ steps, ≥5 `icon` and `vague` tags,
domains disjoint from v2/v3 docs and from training domains).

To fold into the step-1 competitor run, add `--tasks-file demo/tasks_v4.yaml`
to the same `demo.cli_bench` invocation (or merge files) — arms, agents and
analysis stay identical.
