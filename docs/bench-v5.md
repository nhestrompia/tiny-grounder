# Benchmark v5 slice — harder frozen shop/app tasks (Plan v4 step 2)

`demo/tasks_v5.yaml` — 30 frozen tasks on 5 sites. v4's 23 tasks already score
22–23/23 on every arm, so v5 keeps the same sites and replay machinery but
pushes the difficulty up: sessions of 8–15 **actions** (click/type/select/
press/goto — `wait`/`scroll` steps don't count) where re-sent snapshots pile
up, icon-only controls, vaguer goals, and archives that cover plausible
alternative routes, not only the scripted path. v4 is unchanged; v5 is a
separate slice (`data/bench_v5`).

## What is in it

| Site | Tasks | What it adds over v4 |
|------|-------|----------------------|
| saucedemo.com | 8 | Cart surgery mid-checkout (add/remove/cancel up to 15 actions); form-validation questions with `answer_regex` on the error text; `text_not_contains` checks for items that must be gone; the locked-out user |
| demoblaze.com | 6 | Compare-shop chains across categories (8 product/category hops each); a second-page product; a cross-category flagship comparison |
| tutorialsninja.com/demo | 7 | Three-product carts, a cart quantity edit, a 3-way product comparison, the currency dropdown + compare chain, a registration-page navigation task, a two-round contact enquiry |
| the-internet.herokuapp.com | 5 | Fail-then-pass login + logout, a controls tour across two pages, a four-code status tour through the redirector, both dynamic-loading examples, all three JS dialogs |
| blazedemo.com | 4 | Three full 12-action bookings (cheapest / last / priciest row), a vacation-page detour into a flight search |

### Action counts (not path length)

Scripted-action histogram: `{2: 1, 3: 2, 4: 1, 5: 1, 6: 2, 8: 10, 9: 6,
11: 1, 12: 4, 13: 1, 15: 2}` — 24 of 30 scripted paths run 8–15 actions.
**Min** is the fewest actions a correct agent needs for the goal (login =
type+type+click, reads are free). Minimum histogram: `{2: 2, 3: 2, 4: 7,
5: 1, 6: 5, 7: 1, 8: 3, 9: 5, 11: 1, 12: 3, 13: 1}` — **12 of 30 tasks
require 8+ actions**; the rest are shorter by nature of their goal.

| Task | Actions | Min |
|------|---------|-----|
| saucedemo-checkout-swap-items | 15 | 13 |
| saucedemo-checkout-priciest | 11 | 11 |
| saucedemo-checkout-cancel-redo | 15 | 15 |
| saucedemo-cheapest-and-priciest | 9 | 5 |
| saucedemo-checkout-missing-name | 8 | 8 |
| saucedemo-checkout-missing-postal | 9 | 9 |
| saucedemo-cart-remove-keep-two | 13 | 9 |
| saucedemo-locked-out-user | 6 | 6 |
| demoblaze-asus-monitor-price | 8 | 3 |
| demoblaze-priciest-phone | 8 | 4 |
| demoblaze-priciest-laptop | 8 | 4 |
| demoblaze-monitor-under-250 | 8 | 4 |
| demoblaze-next-page-dell | 8 | 3 |
| demoblaze-flagship-priciest | 8 | 6 |
| tn-cart-three-products | 12 | 7 |
| tn-long-checkout-macbook-pro | 9 | 8 |
| tn-compare-three-products | 9 | 5 |
| tn-cart-update-quantity | 9 | 9 |
| tn-currency-euro-lp3065 | 8 | 4 |
| tn-register-fill-form | 2 | 2 |
| tn-contact-enquiry | 6 | 6 |
| internet-login-fail-then-pass | 8 | 7 |
| internet-controls-tour | 6 | 6 |
| internet-status-codes-tour | 9 | 9 |
| internet-dynamic-loading-both | 4 | 4 |
| internet-js-alerts-all | 3 | 3 |
| blazedemo-book-cheapest | 12 | 12 |
| blazedemo-book-last-flight | 12 | 12 |
| blazedemo-book-priciest | 12 | 12 |
| blazedemo-vacation-then-book | 5 | 5 |

- **Icon-only controls:** 13 tasks tagged `icon` (cart icon, burger menu,
  search magnifier, compare, currency and account dropdowns, cart remove).
- **`wrong` routes:** 7 alternative routes recorded alongside the paths, so
  archives cover reasonable detours: the tutorialsninja search page
  (`route=product/search&search=…`) on two tasks, the demoblaze ASUS Full HD
  product page, a blazedemo purchase all the way to `confirmation.php`, a
  saucedemo low-to-high sort + cart, and the-internet's 404 status page (twice).
- **`allow_actions` / `answer_regex` / row-pinning rules:** as in v4 — tasks
  that need logins, form submits, password fields or 'remove'/'checkout'
  clicks opt in via `allow_actions`; price questions anchor to whole amounts
  (`(?<![\d,.])N(?:\.00)?(?!\.?\d)`); bookings name the exact flight where the
  archived POST only replays for the recorded row.
- **Known replay limits:** Playwright's `route_from_har` serves the first
  matching (method, URL) entry, so checks that depend on a same-URL page
  changing mid-path (e.g. a cart losing its last item) are not expressible.
  Replay also ignores POST bodies, so tasks whose only distinguishing state
  is a POST parameter cannot be checked. Concretely, on `blazedemo-book-*`
  **row choice is not verified**: booking the wrong row still replays the
  same `Amount 555 USD` confirmation (the wrong route is archived so the
  booking *runs*, the check just can't tell them apart). Their Min counts
  the full booking form, which every route must still fill.
  `internet-controls-tour` has the same shape: the site clears the checkbox
  message when the input form toggles, so no end-state check can verify the
  remove/add round-trip — the goal was shortened to the enable/disable
  round-trip the check actually verifies (`A checkbox` + `It's enabled!`).
- **Arm-intrinsic refusals:** click-safety refuses password fields, form
  submits and blocked words unless the task opts in with `allow_actions`; all
  tasks that need them do.

## Freeze and verify

```bash
uv run python -m tools.bench_v2 record   --tasks-file demo/tasks_v5.yaml --out data/bench_v5 --branches 6
uv run python -m tools.bench_v2 verify   --tasks-file demo/tasks_v5.yaml --out data/bench_v5
uv run python -m tools.bench_v2 manifest --tasks-file demo/tasks_v5.yaml --out data/bench_v5
```

Recording adds `--headed` for tutorialsninja: the site bot-walls the headless
UA, so `record` sends a desktop-Chrome `RECORD_UA` and can run headed when it
still insists. `sanitize` also drops 304-revalidation entries that have a full
sibling — an empty revalidation body otherwise poisons the replay.

`--branches 6` also archives the first same-site links of each start page.
Freeze hashes in `data/bench_v5/manifest.json`;
`tests/test_bench_v5_tasks.py` asserts the slice properties (≥20 tasks, ≥20
sessions of 8–15 actions, ≥6 `icon` and `vague` tags, ≥5 `wrong` routes,
domains disjoint from v2/v3 docs and from training domains).
