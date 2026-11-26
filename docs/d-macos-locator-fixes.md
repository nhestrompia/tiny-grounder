# D macOS locator fixes

This table is the audit trail for the frozen held-out macOS recording. `goal` text, task ids, apps, tags, task order, and the task set were not changed. Step numbers are 1-based positions in the YAML path after the documented path edits.

## Accessibility evidence used

- Font Book: the search control is `AXTextField` with subrole `AXSearchField` and text/placeholder `Search`. Font families are actionable `AXGroup`s emitted as role `button` with labels such as `Show Courier New`. View controls are `AXRadioButton`s named `View as Grid`, `View as Samples`, and `View as List`; the information control is `AXCheckBox`/`checkbox` named `Show info pane`.
- Dictionary: the search control is an initially blank `AXTextField` whose value becomes the typed query, so it has no stable label. Sources are `AXRadioButton`s such as `English Thesaurus`. Navigation controls move into the `more toolbar items` overflow in the recording window; the Wikipedia source is under `more dictionaries`.
- Script Editor: document windows contain separate text areas whose nonvisual labels are exactly `script source` and `script description`. Both contain the substring `Script`, so the original substring locator was ambiguous/wrong.
- Grapher: `2D Graph` and `3D Graph` are exposed as `AXRadioButton`s inside a tab group and are emitted as role `tab`; the template confirmation is a `Choose` button. Equation creation is exposed through the document's nonvisual `add` menu and `New Equation` menu item.
- Automator: the library search field is an `AXTextField` initially exposed with `Name`; result rows are editable-text-like `AXTextField` elements named for the action.
- Clock: the mode selectors are `AXRadioButton`s named `World Clock`, `Alarms`, `Stopwatch`, and `Timers`.
- Disk Utility: volume outline rows are actionable `AXRow`s emitted as role `other`; `Macintosh HD volumes` is extracted from the row's child static text. The toolbar has a read-only `Info` button alongside destructive controls that were not used.

## Locator and path changes

| task | step(s) | old locator/path | new locator/path | axdump evidence |
|---|---:|---|---|---|
| all five Font Book tasks | search click/type steps | `{role: searchbox, label: Search}` | `{role: textbox, label: Search}` | control is `AXTextField` with subrole `AXSearchField`; axdump role is `textbox` |
| fontbook-code-preview | 3 | `{role: other, label: Courier New}` | `{role: button, label: Show Courier New}` | actionable `AXGroup` emitted as `button`, label `Show Courier New` |
| fontbook-code-preview | 4 | `{role: radio, label: Sample}` | `{role: radio, label: View as Samples}` | toolbar radio `View as Samples` |
| fontbook-times-repertoire | 3 | `{role: other, label: Times}` | `{role: button, label: Show Times New Roman}` | searching `Times` exposes `Times New Roman`, not a bare `Times` row |
| fontbook-times-repertoire | 4 | `{role: radio, label: Repertoire}` | `{role: radio, label: View as Grid}` | closest real character-oriented view control is `View as Grid` |
| fontbook-custom-preview | 3 | `{role: other, label: Avenir}` | `{role: button, label: Show Avenir}` | actionable family row emitted as `button`, label `Show Avenir` |
| fontbook-custom-preview | 4 | `{role: radio, label: Custom}` | `{role: radio, label: View as Samples}` | no `Custom` control is exposed; Samples view exposes editable sample controls |
| fontbook-custom-preview | 5-6 | `{role: textbox, label: Type your own preview text}` | `{role: button, label: Edit Avenir sample}` | Samples view exposes typeface-specific controls such as `Edit Avenir Oblique sample`, not a bare `Edit Avenir sample`; no separate preview textbox was exposed before editing |
| fontbook-grid-view-icon | 3 | `{role: other, label: Helvetica}` | `{role: button, label: Show Helvetica}` | actionable family row emitted as `button`, label `Show Helvetica` |
| fontbook-grid-view-icon | 4 | `{role: button, description: Grid View}` | `{role: radio, label: View as Grid}` | real control is radio `View as Grid` |
| fontbook-grid-view-icon | 5 | `{role: button, description: Repertoire View}` | `{role: radio, label: View as Samples}` | no `Repertoire View` control is exposed; `View as Samples` is the visible sample/repertoire-style control |
| fontbook-font-info | 3 | `{role: other, label: Menlo}` | `{role: button, label: Show Menlo}` | actionable family row emitted as `button`, label `Show Menlo` |
| fontbook-font-info | 4 | `{role: button, description: Font Info}` | `{role: checkbox, label: Show info pane}` | real toolbar control is checkbox `Show info pane` |
| fontbook-font-info | 5 | `{role: button, description: Close}` | `{role: checkbox, label: Show info pane}` | same checkbox toggles the info pane closed; no separate `Font Info` close button was exposed |
| all five Font Book tasks | after search click, before type | direct typing into a persistent search field | added `cmd+f` on the same field before typing | Font Book preserves search text across launches/tasks; its toolbar field does not reliably take focus from a coordinate click, while `cmd+f` focuses it and selects the prior query |
| all five Dictionary tasks | after search click, before type | direct typing into a persistent search field | added `cmd+a` on the same field before typing | Dictionary preserves the prior query; select-all prevents concatenating queries |
| all five Dictionary tasks | search click/type/key steps | `{role: searchbox, label: Search}` | `{role: textbox, nth: 1}` | initial `AXTextField` is blank; after typing its value is the query, so role+nth is the only stable locator |
| dictionary-thesaurus-stream | 4 | `{role: tab, label: Thesaurus}` | `{role: radio, label: English Thesaurus}` | source selector is radio `English Thesaurus` |
| dictionary-back-forward | 5-6 | menu/overflow assumptions from an earlier narrow probe | direct `{role: button, description: Back}` and `{role: button, description: Forward}` | in the actual recording-sized window both are direct `AXButton`s; the overflow popup contains `Text Size`, not Back/Forward |
| dictionary-wikipedia-hopper | 4 | `{role: tab, label: Wikipedia}` | click `{role: button, label: more dictionaries}`, then `{role: menuitem, label: Wikipedia}` | Wikipedia is a menu item inside `more dictionaries`, not a visible tab |
| dictionary-serendipity-entry | 4 | `{role: link, label: serendipity}` | `{role: text, label: serendipity}` | result-list entry is static text (`AXStaticText`, role `text`), not a link |
| all five Script Editor tasks | text-area click/type/key steps | `{role: textbox, description: Script}` | `{role: textbox, description: script source}` | real nonvisual labels include `script source` and `script description`; exact `script source` selects the document editor |
| all five Grapher tasks | 1 | `{role: button, label: 2D Cartesian}` | `{role: tab, label: 2D Graph}` | template choice is a radio inside a tab group, emitted as role `tab`, label `2D Graph` |
| all five Grapher tasks | equation creation click | click `{role: textbox, description: Equation}` | click `{role: button, description: add}`, then `{role: menuitem, label: New Equation}` | document exposes an `add` menu and `New Equation` menu item; no accessible equation textbox was exposed |
| all five Automator tasks | library search click/type steps | `{role: searchbox, label: Search}` | `{role: textbox, label: Name}` | library search control is `AXTextField`, initially exposed with `Name` |
| automator-ask-text-workflow | action row click/key steps | `{role: other, label: Ask for Text}` | `{role: textbox, label: Ask for Text}` | result row is emitted as `textbox` |
| automator-new-folder-workflow | action row click/key steps | `{role: other, label: New Folder}` | `{role: textbox, label: New Folder}` | result row is emitted as `textbox` |
| automator-rename-workflow | action row click/key steps | `{role: other, label: Rename Finder Items}` | `{role: textbox, label: Rename Finder Items}` | result row is emitted as `textbox` |
| automator-application-text | action row click/key steps | `{role: other, label: Get Specified Text}` | `{role: textbox, label: Get Specified Text}` | result row is emitted as `textbox` |
| automator-quick-action-speech | action row click/key steps | `{role: other, label: Speak Text}` | `{role: textbox, label: Speak Text}` | result row is emitted as `textbox` |
| all five Automator tasks | action insertion | one Return on the selected row | two Return steps on the selected row | live probing showed the first Return focused/selected the row and a second Return inserted it |
| clock-stopwatch-lap, clock-stopwatch-two-laps, clock-stopwatch-reset-immediately | 1 | `{role: button, label: Stopwatch}` | `{role: radio, label: Stopwatch}` | mode selector is `AXRadioButton` |
| clock-timer-one-minute, clock-timer-five-minute-pause | 1 | `{role: button, label: Timers}` | `{role: radio, label: Timers}` | mode selector is `AXRadioButton` |

## Recorder/extractor defects fixed during recording preparation

- `SystemEventsDriver.click_at` now closes both nested AppleScript `tell` blocks. The missing second `end tell` caused error `-2741` for every coordinate click.
- `SystemEventsDriver.type_text` now uses `type_text_scripts`: quotes/backslashes are escaped, lines are split, and explicit Return key codes are emitted between lines.
- Cleanup and accessibility-dump failures are contained to the current task instead of discarding the entire run. Applications are launched with `ApplePersistenceIgnoreState` so killed throwaway documents cannot block the next launch through state restoration, task-owned processes are terminated only after graceful quit fails, and a missing target is re-dumped twice over one second before it is counted as a genuine absence.
- `axdump` now keeps standard control roles even when an application omits their action list, treats `AXShowDefaultUI`/`AXShowAlternateUI` as actionable, and lifts row/cell labels from child static text. These fixes expose real controls such as Automator template buttons and Disk Utility volume rows without weakening the read-only safety rules.

## Genuine remaining blockers

- Grapher creates an equation row through `add` -> `New Equation`, but the row editor is not exposed as an accessible text element. Live typing after selecting the row did not change the accessibility tree, so the original `Equation` textbox genuinely does not exist in this macOS version.
- Clock's timer editor exposes hour/minute/second sliders plus `Cancel` and `Start`; there are no `1 min` or `5 min` preset controls.
- Disk Utility's read-only Info panel exposes only unlabeled window-control buttons. The close control has no stable `Close` label or description, and the task schema has no safe window-close step for it.
- Font Book's custom preview flow exposes typeface-specific sample-edit buttons rather than the original generic Avenir preview textbox. Coordinate input also left keyboard focus in the search field during live probing, so the flow was not recorded as a fabricated success.
