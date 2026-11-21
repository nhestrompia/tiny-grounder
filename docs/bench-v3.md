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

Before coverage widening, the 93 snapshots occupied 67,019,562 bytes, or 63.91 MiB. The manifest records SHA-256
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
Initial freeze pytest result: 592 passed in 146.56 seconds.

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

Initial freeze prose lint score: 0.84 findings per 100 words.

## Limitation

All 32 sites are developer-documentation sites, mostly for C libraries and command-line tools. These were the held-out sites with stable, static and crawlable navigation. The v2 set adds universities, museums and news sites, but only for its 18 multistep tasks. Results on v3 therefore describe docs-style navigation (sidebars, indexes, long reference pages) and may not carry over to commerce or app-like sites.

## Coverage widening

All 134 milestone tasks were re-recorded sequentially from live sites with `--neighbours 40`
and `--from-logs runs/bench-v3-pilot/logs`. The recorder collects links from every scripted page,
including the final page. It visits same-site links in page order, includes subdomains, and skips
fragments, duplicates, binary downloads, and form actions. Known pilot misses take priority.
Extra visits respect robots.txt, wait 0.25 seconds between pages, and stop at 60 extra pages per task.
The default v2 branch recorder is unchanged.

| Snapshot set | Tasks | Bytes | MiB |
|---|---:|---:|---:|
| v3, `har/` | 93 | 755,634,701 | 720.63 |
| milestone v2, `har_v2/` | 41 | 1,002,513,213 | 956.07 |
| Total | 134 | 1,758,147,914 | 1,676.70 |

Median sizes are 2.21 MiB for v3 and 17.46 MiB for milestone v2.
The largest snapshot is `quarto-docusaurus`, at 98.96 MiB. Images account for much of the larger archives.
The per-task sizes and every pilot miss URL are in `data/bench_v3/coverage.md` and `coverage.json`.

All 29 unique same-site pilot miss task/URL pairs are now archived, for 100% coverage.
These comprise eight Asciidoctor URLs, twenty Open MPI URLs, and one OCaml URL.
Coverage requires a successful HTML or plain-text response, or a complete archived redirect chain.
The unrelated GitHub URL in the Open MPI logs is outside that task's site and outside this denominator.
This measures known navigation misses, rather than all resource misses such as analytics or video requests.

All 134 tasks pass scripted offline replay and answer checks, and the a11y-cap verifier.
The verifier now supports text inside links and menu buttons that open without changing the URL.
Three transient recording failures passed on retry. No task paths, checks, answers, or goals changed, and no tasks were dropped.
Only the 41 milestone v2 HAR fields changed. Their new snapshots are under `data/bench_v3/har_v2/`.
The original `data/bench_v2` files remain intact.

The manifest hashes both task YAML files and all 134 snapshots. Repeat its update and coverage report with:

```sh
uv run python -m tools.bench_v2 manifest --tasks-file demo/tasks_v3.yaml --also-tasks-file demo/tasks_milestone.yaml --out data/bench_v3
uv run python -m tools.bench_coverage --logs runs/bench-v3-pilot/logs
```

Full validation: `uv run pytest -q` passed with 645 tests in 139.32 seconds.

Coverage section prose lint score: 0.94 findings per 100 words.
