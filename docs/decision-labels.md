# Decision labels

`tools/decision_labels.py` turns benchmark agent logs into five decision datasets under `data/decisions/`. Each row contains the goal, action history, page id, label, provenance, source rule, split, and a small feature object. Pages use the existing compressed `Page` serialization under `data/decisions/pages/<site>/`.

The extractor uses bench v2 and the v3 milestone only. The v4/v5 archives are not present on the data machine, the cli runs used live pages, and `demo/tasks_desktop.yaml` tasks are macOS applications rather than replayable web pages.

## Row schema

Every row has these fields:

- `decision`: `show`, `next`, `enough`, `worked`, or `done`.
- `goal`: the task goal shown to the agent.
- `history`: click and type history in the existing goal-mode format.
- `page_id`: the id of the replayed `Page` file.
- `label`: the decision-specific label.
- `provenance`: run id, run directory, task id, action step, call number, agent, arm, run success, site, and URL.
- `rule`: the versioned extraction rule.
- `label_reason`: the exact branch that produced the label.
- `features.arm`: the arm is also a feature. It is not hidden in provenance only.
- `split`: `train`, `dev`, or `test`.

`provenance.step` is the server's committed-action count. `provenance.call_n` is the tool-call number in the server log.

## Label rules

### SHOW: `show.next_action_success_v1`

Use successful runs only. For a successful click or type action, label the recovered target element and every equivalent element as the elements the agent needed. The replay path is `tools.next_step_recall.build_bench`: parse the action result, recover the pre-action page from the task's HAR archive, map the action text and role with `map_rows`, and add `equiv_ids`.

The label is a proxy. It records the element the agent acted on, not every element the agent needed to read. Scrolling, `go_back`, `done`, `fill`, failed actions, and coordinate actions do not produce SHOW rows.

### NEXT: `next.click_type_success_v1`

Use the same successful click/type steps as SHOW. The label is the action type, the recovered target element, equivalent target ids, and typed text for `type`.

The label does not include confidence because the logs do not contain calibrated agent confidence. The extraction confidence is exact only when the replayed page contains a unique matching element. Text-only fallback matches and duplicate equivalent elements are retained as label noise.

### ENOUGH: `enough.screenshot_or_failure_v1`

Use all runs. At every page state with a later committed action or `done` call:

1. Label positive when the agent called `screenshot` before the commitment.
2. Label positive when the previous action returned a screenshot automatically in the vision arm.
3. Otherwise label negative when the run succeeded.
4. Otherwise label positive when the run failed.

This is the weakest v1 label. Failure after staying in a text view does not prove that a screenshot would have helped. The vision arm makes screenshot use nearly mandatory because its action results include screenshots, while text arms do not offer the same vision tool. Treat arm as a first-class feature and never pool the arms without checking that confound.

### ENOUGH v2: `enough.paired_arm_v2`

Files live under `data/decisions/enough_v2/`; rows keep `decision: "enough"` and carry this v2 rule name. This is a task-level counterfactual label applied to steps. It does not use the agent's step-level screenshot habit as evidence.

Group runs by `(task_id, agent)`. A pair qualifies when that pair has at least one text-only run (`a11y-cap` or an arm whose name starts with `find`) and at least one vision-arm run. Pairing is task-level across bench v2, bench v2 smoke, and the v3 milestone; this is the only way `find+observe` tasks can have a vision counterfactual. For every individual text-arm run in a qualifying pair:

1. If the text-arm run succeeded, every state of that run is negative (`text_arm_succeeded`): text was enough.
2. If the text-arm run failed and at least one vision run for the pair succeeded, every state of that failed text run is positive (`text_failed_and_vision_succeeded`).
3. If the text-arm run failed and no vision run succeeded, drop the state as `both_text_and_vision_failed`.
4. Vision-arm states, text states without a paired vision run, and states whose page cannot be replayed are dropped.

The vision-success policy is deliberately “any successful vision run” because the question is whether the task is visually solvable, not whether a particular repeat happened to succeed. The rule remains noisy: a failed text run can reflect agent mistakes rather than missing visual information, and a successful text run can be lucky. Applying one task-level inference to every step also ignores the fact that only some steps may need visual detail.

### WORKED: `worked.error_recheck_retry_v1`

Use all runs. For a state after a committed browser action and before the next committed action or `done`:

1. Label negative when the previous action returned a tool error.
2. Label negative when the agent immediately called `snapshot`, `observe`, `find`, `inspect`, `read`, or `screenshot` before committing again.
3. Label negative when the next commitment retries the same action on the same target. Element ids must match exactly; coordinates may differ by at most two CSS pixels.
4. Otherwise label positive.

Run success does not enter the v1 label. The label measures the agent's behavior, not the true environment state. A re-check can be careful rather than corrective, and proceeding can hide a failed action. In a11y and `find+observe` arms, the action result already contains a fresh text view, so explicit re-observation is stronger evidence than it first appears. In the vision arm, a screenshot is part of the normal loop, so this rule is arm-biased.

### WORKED v2: `worked.progress_v2`

Files live under `data/decisions/worked_v2/`; rows keep `decision: "worked"` and carry this v2 rule name. The rule labels evidence of environment progress rather than the agent's re-observation habit.

For a state after a committed browser action:

1. Label negative when that action returned a tool error.
2. Label negative when the same action targets the same element again within the next two committed actions. Semantic click/type targets are compared by normalized role and visible text; raw element ids are a same-page fallback. Coordinate retries must be within two CSS pixels.
3. Label positive when the next committed action is `done` and the run succeeds.
4. Label positive when the next committed click/type targets an element with no plausible role/text match in the replayed pre-action page. This is evidence that the UI changed enough to expose a new target.
5. Otherwise drop the state. Immediate `find`, `snapshot`, `observe`, `inspect`, `read`, or `screenshot` calls never create a negative label by themselves. Re-observed states without independent progress evidence are counted under `reobservation_without_progress_evidence`.

This rule is intentionally conservative and drops many ambiguous states. It still has noise. A new target can appear for unrelated reasons; a persistent navigation link can look like the same element across pages; replaying a URL loses same-URL DOM mutations, focus, scroll, open menus, and form contents. A successful `done` can also be correct because the task was already complete before the action being evaluated.

### DONE: `done.successful_done_next_v1`

Use all runs. At every page state with a later committed action or `done` call:

1. Label positive when the next commitment is `done` and the run is finally successful.
2. Label negative when the next commitment is `done` but the run fails.
3. Label negative when the next commitment is another action.

Run success corrects some premature or wrong `done` calls. It also creates noise: a state can already satisfy the goal while the agent does unnecessary extra work, and a run can fail for an answer-format reason after reaching the right page.

## Shared noise and biases

- Agent actions are self-selected. Successful agents show what worked for them, not what was required.
- Arms change what agents can see and which tools exist. Arm is a feature, not a hidden control.
- Replaying a URL resets scroll position, focus, open menus, form contents, and other same-URL DOM state. This is most damaging after actions that do not navigate.
- Pages outside the frozen archive replay as `page not archived` and are dropped.
- SHOW/NEXT rows are deduplicated across repeated runs of the same page state. ENOUGH/WORKED/DONE rows are not deduplicated because each row is a separate agent decision. Repeated tasks and agents therefore shape the base rates.
- WORKED v2 cannot infer semantic progress for most coordinate and scroll actions, so vision-arm states are more likely to be dropped as ambiguous. Per-arm base rates are therefore conditioned on different action mixes as well as different behavior.
- Text logs are truncated to 4,000 characters per call. The action status line and URL survive, but long snapshots do not.

## Splits and exclusions

Splits are by site. A deterministic greedy assignment uses the v1 row counts to approximate a 70/15/15 train/dev/test division while keeping every site in exactly one split. The v2 files reuse that assignment so the v1 split and manifest cannot drift. `data/decisions/test-manifest.json` freezes the v1 test rows and referenced pages by SHA-256. `data/decisions/test-manifest-v2.json` separately freezes the v2 ENOUGH/WORKED test rows and referenced pages.

These frozen benchmark sites are excluded from every split: saucedemo.com, demoblaze.com, tutorialsninja.com, the-internet.herokuapp.com, and blazedemo.com. macOS desktop tasks are excluded for the same reason as live cli tasks: they do not have replayable bench v2/v3 milestone web archives.

## Regeneration

From the worktree, with the read-only data checkout at `/path/to/tiny-grounder`:

```bash
uv run --extra demo --with tldextract python tools/decision_labels.py extract \
  --source-root /path/to/tiny-grounder \
  --out data/decisions \
  --workers 6
```

The extractor reads the other checkout only. It writes pages, rows, `summary.json`, `screenshot-audit.json`, and the test manifests in this checkout. When a v1 manifest already exists, extraction preserves it rather than rewriting the frozen file.

## Review verdict (2026-11-26, before any model is trained)

- **SHOW, NEXT, DONE (v1): accepted** for training and gating.
- **ENOUGH (v1 and v2): not trainable from these logs.**
  - v1 encodes "run failed" or "vision arm".
  - In v2, 62 of 231 (task, agent) pairs carry both labels: the same agent on the same task passes some text runs and
    fails others, and each failure is labelled "screenshot needed". So most positives are agent variance.
  - The test split has only 14 positive runs, and every heuristic has AUROC < 0.55.
  - Plan: ENOUGH ships as a structural rule (the target or the top-ranked elements have no visible text, or a high
    empty-text ratio, plus the ranker's null probability). A learned version waits for a source with a real
    counterfactual: the same state with and without pixels, e.g. icon-only targets in the Mac walks.
- **WORKED (v2): label valid, but a model isn't needed yet.**
  - URL change alone reaches AUROC 0.90 (P 0.988, R 0.978), because v2 positives are mostly navigation or `done`.
  - Same-URL interactions (forms, SPAs, desktop dialogs), where a model would add value, are mostly dropped for lack
    of evidence.
  - Plan: ship WORKED as a deterministic before/after page diff checked against the action's intent. Revisit with the
    Mac walks, which store pre- and post-action dumps for same-window interactions.
