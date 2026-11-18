# Collecting macOS pages (step 13)

## One-time setup

```
cd grounder/extract/macos && swift build -c release
```

System Settings > Privacy & Security, enable the terminal app you run from (Terminal, iTerm, Ghostty, ...) under:

- **Accessibility** (always, axdump reads the UI tree through it)
- **Screen Recording** (only for `--split test`, for the window screenshots; without it pages are still saved and a warning is shown)

Restart the terminal after granting.

## Collect

```
uv run python tools/ax_collect.py --out data/pages_ax --split train
uv run python tools/ax_collect.py --out data/pages_ax --split test     # held-out apps only
uv run python tools/ax_collect.py --out data/pages_ax --stats          # apps x pages x split
```

Press Enter, then bring the window you want to front within the countdown (`--delay`, default 3 s); `q` quits.
Take 3-5 windows per app in different states (different views, a settings pane, a dialog). Pages are written to
`data/pages_ax/{split}/{app}/{page_id}.json.gz`. Skipped with a message: fewer than `--min-elements` (default 5)
elements, a near-duplicate (Jaccard >= 0.9 on text) of a page already saved for that app, and any app that already
has pages in the other split.

## Next steps

```
# teacher data from the training apps
uv run python -m grounder.data.teacher --pages data/pages_ax --split train --out data/examples_ax_train.jsonl ...

# hand-label the held-out apps (screenshot beside each page, boxes drawn from the window-relative `box`)
uv run python tools/label.py --pages data/pages_ax/test --out hand_desktop.jsonl
```

Aim for 50 instructions on the test apps.

## Suggested apps

Train (about 25): Finder, Safari, Notes, Mail, Calendar, Messages, Reminders, Preview, TextEdit, System Settings,
Terminal, Activity Monitor, Calculator, Contacts, Maps, Photos, Music, Books, Stocks, Weather, Xcode, Visual Studio Code,
Slack, Spotify, Chrome.

Test, held out (5): Numbers, Pages, Keynote, Font Book, Disk Utility. Pick apps unlike the training ones in layout
where possible, and keep them out of `--split train` entirely (the collector refuses the mix).
