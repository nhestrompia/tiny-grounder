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

## Automatic collection

```
uv run python tools/ax_auto.py --out data/pages_ax --split train --dry-run   # print the plan, do nothing
uv run python tools/ax_auto.py --out data/pages_ax --split train
uv run python tools/ax_auto.py --out data/pages_ax --split test              # Numbers, Pages, Keynote, Font Book, Disk Utility
uv run python tools/ax_auto.py --out data/pages_ax --apps "Notes,Mail" --max-per-app 4 --settle 3
```

For each installed app (about 30 stock Apple apps by default, or `--apps`) it launches the app, dumps the front window,
then opens and dumps a few safe views and undoes each: settings window (cmd+,), new window (cmd+n, only in apps where it creates no data — Finder, Safari, TextEdit, Terminal, Maps and the like; closed with cmd+w,
"Don't Save" via cmd+d if a save sheet appears), open and print panels (cmd+o, cmd+p) and the find bar (cmd+f), each
closed with Escape. System Settings is walked pane by pane through `x-apple.systempreferences:` URLs. Only those
keystrokes are ever sent, never a click, cmd+q or delete. Apps it launched are quit afterwards, except Finder, Terminal
and the app you run it from (which is skipped). Dumps of the wrong frontmost app are dropped. Saving, near-duplicate
skipping, the train/test split guard and test screenshots are the same as the interactive collector; a final table
counts saved, duplicate, too-few, wrong-app and error scenes per app.

The terminal needs **Accessibility** (to send keystrokes and for axdump) and, for `--split test`, **Screen Recording**.
Do not touch the keyboard or mouse while it runs; keystrokes go to whatever is in front.


## Goal-walk collection and reverse task synthesis

`--mode walks` is the bounded random-but-safe collector used for the second macOS next-step data build. Its app
membership and per-app restrictions are in `configs/apps_policy.yaml`; the hard click denylist, typing boundary,
sign-in/payment refusal and privacy scrubber are code in `tools/mac_safety.py` and cannot be weakened from YAML.

```
uv run python tools/ax_auto.py --mode walks --out data/mac_walks --split train --dry-run
uv run python tools/ax_auto.py --mode walks --out data/mac_walks --split train
uv run python tools/ax_auto.py --mode walks --out data/mac_walks --split test
```

The train split collects only the policy's train apps; the test split collects only test_v2 apps. frozen_test_v1,
excluded apps, unknown apps and cross-split requests are refused before launch. A locked or curtained login session
(`loginwindow` frontmost) exits with status 4 because no real window can be inspected safely. Screen Recording is
probed once; when unavailable the run continues without screenshots. When available, each novel state has a window PNG
beside its page JSON and icon-only elements are cropped to 64 px under `crops/`.

Every step is recorded as `pre_page_id -> action -> post_page_id`. Novel states are exact/near-duplicate filtered with
Jaccard >= 0.9. Pages are privacy-scrubbed before being written; history, account/profile and policy-denied views are
never saved. Logic Pro is collected only through a writable temp-folder copy of its factory blank Quickstart project;
the copy and folder are deleted after Logic is quit. Final Cut Pro has no factory blank library package on disk, so it
is skipped rather than risk opening a user library. The collector writes
`data/mac_walks/{split}/collection-report.json` with per-app states, walks, skip reasons, privacy counts, screenshot
status and safety denials.

After GUI exploration is complete, synthesize goals with the already-downloaded local Gemma GGUF. This phase starts
`llama-server` itself and stops only the server it started:

```
uv run --group local python tools/mac_goals.py --walk-root data/mac_walks --split train --out data/nextstep --name mac_train_v2
uv run --group local python tools/mac_goals.py --walk-root data/mac_walks --split test --out data/nextstep --name mac_test_v2
```

It emits goal-mode examples in the same schema as `data/nextstep/d_macos.jsonl`, per-set build reports, pages under
`pages_<name>/`, and frozen SHA-256 manifests. Never run a ranker or `next_step_recall` on `mac_test_v2`.

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
