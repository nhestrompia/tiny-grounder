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

This is the weakest label. Failure after staying in a text view does not prove that a screenshot would have helped. The vision arm makes screenshot use nearly mandatory because its action results include screenshots, while text arms do not offer the same vision tool. Treat arm as a first-class feature and never pool the arms without checking that confound.

### WORKED: `worked.error_recheck_retry_v1`

Use all runs. For a state after a committed browser action and before the next committed action or `done`:

1. Label negative when the previous action returned a tool error.
2. Label negative when the agent immediately called `snapshot`, `observe`, `find`, `inspect`, `read`, or `screenshot` before committing again.
3. Label negative when the next commitment retries the same action on the same target. Element ids must match exactly; coordinates may differ by at most two CSS pixels.
4. Otherwise label positive.

Run success does not enter the label. The label measures the agent's behavior, not the true environment state. A re-check can be careful rather than corrective, and proceeding can hide a failed action. In a11y and `find+observe` arms, the action result already contains a fresh text view, so explicit re-observation is stronger evidence than it first appears. In the vision arm, a screenshot is part of the normal loop, so this rule is arm-biased.

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
- Text logs are truncated to 4,000 characters per call. The action status line and URL survive, but long snapshots do not.

## Splits and exclusions

Splits are by site. A deterministic greedy assignment uses row counts to approximate a 70/15/15 train/dev/test division while keeping every site in exactly one split. `data/decisions/test-manifest.json` freezes the test rows and referenced pages by SHA-256.

These frozen benchmark sites are excluded from every split: saucedemo.com, demoblaze.com, tutorialsninja.com, the-internet.herokuapp.com, and blazedemo.com. macOS desktop tasks are excluded for the same reason as live cli tasks: they do not have replayable bench v2/v3 milestone web archives.

## Regeneration

From the worktree, with the read-only data checkout at `/path/to/tiny-grounder`:

```bash
uv run --extra demo --with tldextract python tools/decision_labels.py extract \
  --source-root /path/to/tiny-grounder \
  --out data/decisions \
  --workers 6
```

The extractor reads the other checkout only. It writes pages, rows, `summary.json`, `screenshot-audit.json`, and the test manifest in this checkout.
