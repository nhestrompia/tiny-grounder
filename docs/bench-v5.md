# Benchmark v5 slice — harder frozen shop/app tasks (Plan v4 step 2)

`demo/tasks_v5.yaml` — 30 frozen tasks on 6 sites. v4's 23 tasks already score
22–23/23 on every arm, so v5 keeps the same sites and replay machinery but
pushes the difficulty up: more 8–15-step sessions where re-sent snapshots pile
up, more icon-only controls, vaguer goals, and archives that cover plausible
alternative routes, not only the scripted path. v4 is unchanged; v5 is a
separate slice (`data/bench_v5`).

## What is in it

| Site | Tasks | What it adds over v4 |
|------|-------|----------------------|
| saucedemo.com | 7 | Cart surgery mid-checkout (add/remove/back-out up to 15 steps); form-validation questions with `answer_regex` on the error text; the locked-out user |
| demoblaze.com | 4 | Price questions that require the product page (the listing's do-nothing run must fail); vague "cheapest/priciest" goals |
| tutorialsninja.com/demo | 8 | Longer cart flows (two products, remove-to-empty), the compare control, the currency dropdown (€ evidence), the icon-only account menu, a guest checkout through the category |
| the-internet.herokuapp.com | 6 | Multi-stage dynamic controls (remove + enable), hidden-element loading, failed login, JS alert result, status-code and redirect navigation |
| blazedemo.com | 4 | Two more full 12-step bookings (cheapest row by price, last row), a price question, the vacation link |
| news.ycombinator.com | 1 | The newest-submissions page |

- **Long sessions:** 8 tasks run 8–15 steps (cart surgery, two-product carts,
  guest checkout, both flight bookings, remove-to-empty).
- **Icon-only controls:** 13 tasks tagged `icon` (cart icon, burger menu,
  search magnifier, compare, currency and account dropdowns, cart remove).
- **Vague goals:** 13 tasks tagged `vague` ("the most expensive thing the store
  sells", "the last flight on the list", "take the phone back out").
- **`wrong` routes:** 8 alternative routes recorded alongside the paths, so
  archives cover reasonable detours: the tutorialsninja search page
  (`route=product/search&search=…`) on three tasks, the demoblaze ASUS Full HD
  product page, a blazedemo purchase all the way to `confirmation.php`, a
  saucedemo low-to-high sort + cart, and the-internet's 404 status page.
- **`allow_actions` / `answer_regex` / row-pinning rules:** as in v4 — price
  questions anchor to whole amounts, and bookings name the exact flight where
  the archived POST only replays for the recorded row.

## Freeze and verify

```bash
uv run python -m tools.bench_v2 record   --tasks-file demo/tasks_v5.yaml --out data/bench_v5 --branches 6
uv run python -m tools.bench_v2 verify   --tasks-file demo/tasks_v5.yaml --out data/bench_v5
uv run python -m tools.bench_v2 manifest --tasks-file demo/tasks_v5.yaml --out data/bench_v5
```

`--branches 6` also archives the first same-site links of each start page.
Freeze hashes in `data/bench_v5/manifest.json`;
`tests/test_bench_v5_tasks.py` asserts the slice properties (≥20 tasks, ≥7
sessions of 8+ steps, ≥6 `icon` and `vague` tags, ≥5 `wrong` routes, domains
disjoint from v2/v3 docs and from training domains).
