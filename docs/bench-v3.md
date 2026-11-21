# Benchmark v3 freeze

The frozen set contains 93 new tasks on 32 held-out sites. Each site has at most four tasks.
Together with the 18 existing v2 multistep tasks, this gives 111 multistep tasks for Plan v3.

There are 56 fact questions and 37 page-navigation tasks. All 93 have the multistep tag,
81 have the vague tag, and 50 have the long tag. No task needs an icon-only control.
The long tag means at least one visited page has more than 12,000 visible characters.

| Navigations | Tasks |
|---|---:|
| 2 | 20 |
| 3 | 71 |
| 4 | 2 |
| 5 | 0 |

Every path changes pages at each step. The final link is absent from the start page.
Each task allows its path length plus three steps. All final URL checks are anchored.
Fact questions also require an answer regex that matches the frozen page.

The 93 snapshots occupy 67,019,562 bytes, or 63.91 MiB. The manifest records SHA-256
hashes of the task YAML and every snapshot. Snapshots and verification records are in
`data/bench_v3`, which the repository ignores in Git.

## Verification

The authoring script checked robots rules before each navigation and recorded the route with Playwright.
The authoring audit stores page URLs, titles, visible character counts, and robots response statuses.
Offline replay uses the existing HAR replay implementation with a dead proxy.
Missing advertisement, analytics, or video requests receive local replay responses.
The page paths and answers do not depend on those requests.

The deterministic accessibility verifier uses the YAML path to identify the next link.
It reads the actual paginated a11y-cap snapshots, finds the corresponding exposed ref,
and invokes the baseline click tool. It also checks that question answers appear in
that baseline output. This proves interface solvability, not model performance.
No model calls, observe-arm runs, or agent subscriptions were used.

All 93 tasks pass that baseline verification. The standard offline replay also checks
that doing nothing fails, the path reaches the required page, and the answer matches.
Full pytest result: 592 passed in 146.56 seconds.

Repeat the checks from the repository root:

```sh
uv run python -m tools.bench_v2 verify --tasks-file demo/tasks_v3.yaml --out data/bench_v3
uv run python -m tools.verify_tasks_v3
uv run python -m tools.bench_v2 manifest --tasks-file demo/tasks_v3.yaml --out data/bench_v3
uv run pytest -q
```

The recorder now accepts `--out` and `--har-dir`. Its v2 defaults stay the same.
Relative task-file arguments also work when creating manifests.

## Rejected candidates

Of 108 authored candidates, 15 were rejected. They are excluded from the YAML and snapshot directory.
The full rejection record is `data/bench_v3/dropped.json`.

| Task | Reason |
|---|---|
| `fossil-scm-home` | The handbook served a Server Overload page while recording. |
| `scons-index` | The live home page moved documentation into hidden dropdown navigation. static crawl and browser layout differ. |
| `scons-scons-user` | The live home page moved documentation into hidden dropdown navigation. static crawl and browser layout differ. |
| `taskfile-styleguide` | The style-guide link was absent from the rendered guide navigation. |
| `hledger-import-csv` | The broad home page already linked the answer, and the narrower manual did not expose the scripted navigation. |
| `hledger-currency-conversion` | The broad home page already linked the answer, and the narrower manual did not expose the scripted navigation. |
| `hledger-quicken` | The broad home page already linked the answer, and the narrower manual did not expose the scripted navigation. |
| `quarto-notebook` | Hydrated links or duplicate navigation labels changed the offline route. repeated replay failures. |
| `quarto-notebook-1` | Hydrated links or duplicate navigation labels changed the offline route. repeated replay failures. |
| `quarto-running` | Hydrated links or duplicate navigation labels changed the offline route. repeated replay failures. |
| `pantsbuild-tagging-docker-images` | Client-side navigation recorded route payloads but no standalone HTML documents. offline paths failed repeatedly. |
| `pantsbuild-overview` | Client-side navigation recorded route payloads but no standalone HTML documents. offline paths failed repeatedly. |
| `pantsbuild-remote-caching-and-execution` | Client-side navigation recorded route payloads but no standalone HTML documents. offline paths failed repeatedly. |
| `jsonnet-fractal-configuration` | Rendered navigation hides the tutorial and case-study links. no reliable 2-5-navigation route without menu-only actions. |
| `jsonnet-fractal-operations` | Rendered navigation hides the tutorial and case-study links. no reliable 2-5-navigation route without menu-only actions. |

Prose lint score: 0.84 findings per 100 words.

## Limitation

All 32 sites are developer-documentation sites, mostly for C libraries and command-line tools. These were the held-out sites with stable, static and crawlable navigation. The v2 set adds universities, museums and news sites, but only for its 18 multistep tasks. Results on v3 therefore describe docs-style navigation (sidebars, indexes, long reference pages) and may not carry over to commerce or app-like sites.
