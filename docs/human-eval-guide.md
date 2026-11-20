# Human evaluation set: guide for writers (people and agents)

Plan v2.1, step 3. The human set is written by people (and Claude/GPT writer agents) who have **not seen the model's outputs**, on **held-out sites** that are not in training. It is frozen before any model choice is made. Two splits:

| split | who writes | size | use |
|---|---|---:|---|
| **select** | Claude / GPT writer agents | about 250, at least 200 of them *hard* | choose the ranker (plan step 4) |
| **confirm** | the user | about 150 | untouched until the release decision (step 7) |

Pages for the two splits come from **different sites** (split by site, not by page). The lists are `data/heldout/confirm_pages.txt` and `data/heldout/select_pages.txt`.

## What to write

Look at a page screenshot and write what *a person would tell a helper to do* on it, then name the one element that does it.

- Write naturally, like telling a friend next to you. No element ids, no "click the element with…".
- **Vary the style** across a page and across the set:
  - **literal**: uses the words on the element ("click Sign in", "open the pricing page");
  - **paraphrase**: same meaning, different words ("log me in", "how much does it cost");
  - **goal**: an aim, not a control ("I want to find something cheaper", "change how dark the screen is");
  - **icon / position**: the target has no (or hidden) text: "the magnifier at the top right", "the little cart", "the hamburger menu".
- Do **not** copy an element's text unless that is what you would naturally say. Do not read the element list to pick wording for a *hard* item; write from the screenshot.
- One instruction = one target. If several elements do the same thing (same role, text and link), any of them counts; the tooling records the equivalents.
- Be specific enough that a careful person would pick the same element. Skip an item rather than write something ambiguous.
- English only, one sentence, no personal data, nothing you would not say aloud.

### Absent target

Some instructions must be about something **the page cannot do** ("track my order" on a docs page). Write something plausible for that kind of site and mark the target as *not on this page* (`target_id: null`). The model should answer "none".

## Quotas

Targets for the whole human set; the tags are `literal` / `paraphrase` / `goal` (style) and `icon` / `position` / `vague` (the UI checkboxes). **These numbers are proposals** (the plan fixes the dimensions and the select hard-slice floor, not the exact mix); change them here before writing starts, not after.

| dimension | select (about 280) | confirm (about 150) |
|---|---|---|
| platform | web from these pages; macOS items written the same way on AX pages | same |
| **hard** items (paraphrase, icon/unlabelled, or goal) | **at least 200** (about 75%) | at least 90 (60%) |
| literal | at least 25% (control group; the cheap rankers should win here) | at least 25% |
| paraphrase | at least 35% | at least 25% |
| goal | at least 15% | at least 15% |
| text target vs icon/unlabelled target | at least 20% icon/unlabelled | at least 15% |
| absent target | 8–10% | 8–10% |
| per page | 1–3 items, mixed styles; at most 4 | 1–3 |
| per site | no more than 8 items | no more than 8 |

"Hard" in the plan means paraphrase, icon, or goal. Everything is also tagged afterwards by the analysis scripts (lexical overlap with the target text decides literal vs paraphrase), so a writer's tag is a hint, not the final label.

## For agent writers (select split)

1. Read `data/heldout/packets/{page_id}.json` for each page you are given. A packet has `url`, `title`, `screenshot` (path to the full-page 1280-px-wide PNG, look at it), `n_elements` and `elements`: one line per element, `id|role|text|context|x,y,w,h` (box in screenshot pixels).
2. Write items following the rules above. Use the screenshot to write; use the element list only to find the **id** of the element you mean and to check that the instruction is unambiguous.
3. Output one JSON object per line (JSONL), no extra text:

   ```json
   {"page_id": "…", "query": "the little cart at the top right", "target_id": 17, "style": "goal", "tags": ["icon", "position"]}
   {"page_id": "…", "query": "check on my order status", "target_id": null, "style": "paraphrase", "tags": []}
   ```

   `style` is `literal`, `paraphrase` or `goal`; `tags` is a subset of `icon`, `position`, `vague`. The merge step turns these into `records.Example` lines (kind `hand`, mode `do`, `equiv_ids` from `records.equiv_ids`, `split` `select`, `annotator` = the agent's name).
4. Writers must not see model predictions, scores or other writers' items for the same page. Use at least two different writer models and let each cover different pages, so one model's style does not define the set.
5. Tall pages are long images; if a part of the page is unreadable at the shown resolution, skip those elements.

## For people (confirm split): the labelling UI

```bash
uv run python tools/label_web.py --pages data/heldout/pages/test \
    --assign data/heldout/confirm_pages.txt --out data/heldout/confirm.jsonl --target 150 --annotator user
```

It opens <http://127.0.0.1:8765/>. For each page:

1. Type what you would tell someone to do on the page, in the box.
2. Click the element it refers to on the screenshot (the box under the pointer lights up; click the same spot again to pick a larger box underneath). Or tick **Not on this page**.
3. Optionally tick the tags (refers to an icon / a position / is vague).
4. **Enter** saves and clears the box so you can write another for the same page (it moves on by itself after 3 entries). **Next page** when you are done with a page; **Skip** for broken or empty pages; **Undo** removes your last entry.

Keys: `Enter` save · `Alt+→` next page · `Alt+S` skip · `Alt+Z` undo · `Alt+0` not on this page · `Alt+1/2/3` tags · `Alt+N` show element numbers · `Esc` clear the click.

Every entry is written to the JSONL at once, so you can stop any time (`Ctrl-C`) and run the same command to resume; finished and skipped pages are remembered in `data/heldout/confirm.jsonl.pages.jsonl`. Progress (`n / target`) is shown at the top of the side panel. Do not look at the model's predictions, and do not write the confirm items after seeing select results.

## How the pages were made

- `data/heldout/sites.txt`: about 150 English, publicly browsable sites (government and public services including non-US, universities, museums and libraries, small shops, local news, open-source and developer-docs sites, recipes and hobby sites, nonprofits, transit and travel, SaaS marketing, search/filter-heavy pages). Built by `tools/heldout_sites.py`: every registrable domain is absent from `data/leakage/heldout_domains.txt` (which lists the *training* domains), absent from `demo/tasks.yaml`, passes the blocklist, the adult-keyword check and the Cloudflare family DNS check, and answers HTTP 200 within 15 s.
- Crawl: `uv run python -m grounder.data.crawl --sites data/heldout/sites.txt --out data/heldout/pages --links 2 --screenshots-for test --force-split test --concurrency 6` (`--force-split test` puts every site in `test`; the normal hash split would put 90% in train). Start page plus up to two same-site pages (one search-result page if the home page has a GET search form).
- `uv run python -m tools.heldout_prepare`: drops pages without a screenshot, with fewer than 10 interactive elements, or that collide with training (domain, URL, or element-text near-duplicate, Jaccard ≥ 0.9, as in `docs/leakage.md`); splits the rest by site into the two lists and writes the select packets. Dropped pages and reasons: `data/heldout/dropped_pages.jsonl`.
- Extractor recall is **not** measured here: the element list is what the extractor found, so an item whose true target the extractor missed cannot be written. Report extractor recall separately (plan step 3).
