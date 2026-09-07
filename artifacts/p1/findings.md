# P1.20–P1.21 findings

Three defects surfaced while building the end-to-end path. All three predate this
work; none is caused by `harness.py`. Reproduce with
`scripts/p1_reward_signal_audit.py` and the snippets below.

---

## F1 — Reward signal: hidden tests miss the injected bug in 32 of 43 cells

`artifacts/p1/reward_signal_audit.json`, over every (repo × mutation-satisfiable
bug type) at `c1-L0-single-function`:

| status | cells | meaning |
|---|---|---|
| `ok` | 11 | hidden tests fail on the buggy tree — the intended behaviour |
| `HIDDEN-MISSES-BUG` | 21 | **visible tests fail, hidden tests pass** |
| `NEITHER-CATCHES` | 11 | **no test fails on the buggy tree at all** |

**Why the 21 matter.** PRD §4.2 and architecture §3 specify hidden tests as a
*strengthened superset* of visible tests. In 21 cells they are strictly weaker.

**Correction (2026-09-06, after Codex review).** An earlier version of this note
called `verify()`'s `visible_regression` clause an undocumented addition to the
reward rule. That was wrong: PRD §4.8 item 5 says "Pre-existing regression tests
must still pass", so the clause implements the spec. These 21 cells are therefore
**not** demonstrated false-positive rewards — the combined grader rejects the
unchanged buggy tree. They demonstrate inadequate hidden coverage, which still
violates the superset property and still weakens the hidden suite's role as the
primary anti-cheat mechanism (§4.8 item 2).

**Why the 11 are worse.** A task where nothing fails on the buggy tree is one an
agent passes by doing nothing — confirmed through the full Docker harness on the
parcel off-by-one instance. Those instances inflate bucket success rates in D1/D2,
and can also create artificial heterogeneity, distorting the aggregation gap
itself. All 7 `parcel_ledger` cells are degenerate.

**Two qualifications (Codex review).** "`p_i = 1` for every policy" is too strong —
a policy can still break passing code or trip a cheat flag. The demonstrated claim
is that a no-op passes this grader at the tested seed. And these are 43 instances
at injector seed 11, one per satisfiable repo/type cell: not an exhaustive census
of those cells, and not a frequency estimate. All eight pristine repositories pass
their own combined suites.

**Root cause for parcel_ledger.** It has two source modules, `__init__.py` and
`billing_rules.py`, but both the visible suite (`from parcel_ledger.billing_rules
import *`) and the hidden generator import only `billing_rules`. The mutation
injector draws candidates from every non-test `.py`, so it injects into
`__init__.py`, which nothing exercises:

```
edit: parcel_ledger/__init__.py  allocate  line 53   len(values) -> len(values) + 1
visible_rc=0  hidden_rc=0
```

**Correction (Codex review).** An earlier version claimed the `parcel_ledger`
cross-file tasks are themselves degenerate. They are not: all 7 bug types at
`c2-L0-cross-file` return reward 0 under a no-op policy, because each pairs an
untested `__init__.py` edit with a tested `billing_rules.py` edit. The real defect
is an **advertised two-defect task whose grading ignores one of the two defects**.

P1.12's validation ("inject → tests fail → revert → tests pass") passes because
`tests/test_injector.py` builds its own synthetic fixture; it never asserted this
against the real repo pool.

---

## F2 — Injector: `commit-rollback` and `feat-add` are unusable on 7 of 8 repos

```
src/injector/core.py:229   if module.code_for_node(node).strip() == before.strip():
libcst._nodes.base.CSTCodegenError: Must specify a concrete default_indicator if default used on indicator
```

`Match.on_visit` calls `code_for_node` on every node it walks. libcst cannot
codegen an `Annotation` node standalone, so **the first type-annotated parameter
in a targeted module aborts the whole injection**. Minimal repro:

```python
cst.parse_module("def f(x: int): return x").code_for_node(<the Annotation node>)  # raises
```

Measured over 8 repos × 18 buckets × 3 seeds (432 combinations): **9 succeed**,
360 raise `CSTCodegenError`, 63 produce no task.

**Correction (Codex review).** That grid is *not* source-balanced and its headline
is misleading: seeds (20260906, 7, 42) draw 36 `commit-rollback` and 18 `feat-add`
buckets and **no `mutation` at all**, so 9/432 describes the provenance sources
only. Codex's seeds (11, 12, 13) draw 100% `commit-rollback`. Codex further splits
the 63 into 45 with no eligible task and **18 with invalid history provenance**
(`slot_planner` rollback fragments that do not match its git history) — a separate
catalog defect that should be rejected explicitly, not counted as an ordinary
unavailable task. The source-balanced evidence is the per-source table below, which
stands.

| source | repos where it works |
|---|---|
| `mutation` | 8 of 8 (4–7 of 7 bug types each) |
| `commit-rollback` | 1 of 8 (`parcel_ledger` only) |
| `feat-add` | 1 of 8 (`parcel_ledger` only) |

`parcel_ledger` survives only because its catalog targets `billing_rules.py`,
which carries no annotations.

**Impact.** The `source` dimension collapses to mutation-only — exactly the bug
class BugPilot (PRD §4.4, §2.4) says is least representative, and the realism
argument for the dimension's existence. Buckets randomise `type` and `source`
within, so buckets are currently not what the docs describe.

**Candidate fix — superseded by the Codex review.** My proposal was to wrap
`code_for_node` in `try/except CSTCodegenError`. Two problems: `libcst` does not
export `CSTCodegenError` at top level (verified), and the guard is insufficient.
Codex tested it in memory and found it merely uncovers the next layer — whole-node
match failures from indentation and incomplete catalog fragments, replacement
parsing failures on indented statements, the invalid `slot_planner` history, and
too few candidates for multi-edit tasks. Guarded, only 14/56 rollback and 18/56
feature cases generate at count 1, and **no non-mutation multi-edit case succeeds
on any repo**.

Better fix: filter *before* code generation to the node families the replacement
branches actually support (`BaseExpression`, `BaseSmallStatement`, `BaseStatement`),
returning `True` for everything else so traversal continues. Beyond the visitor,
the catalog needs a validated contract covering complete CST fragments, indentation,
replacement shape, real history, and sufficient distinct edits.

---

## F3 — Repo pool: cross-file buckets are empty for 7 of 8 repos

Mutation coverage over the full 18-bucket grid: the three `cross-file` buckets
(`c2-L{0,1,2}-cross-file`) yield **zero** eligible tasks on every repo except
`parcel_ledger`, because the other seven are single-module packages.

Consequences: 3 of 18 buckets are unreachable pool-wide; the only repo that
populates them is the one whose tasks are degenerate (F1); and a hard bucket drawn
from a single repo cannot support the repo-level cluster bootstrap that PRD §5.3
and §8 specify for interval reporting.

---

## F4 — The GPU host cannot run the sandbox

`ssh -p 47559 root@connect.bjb2.seetacloud.com`, RTX 4090D, 128 CPUs, Linux 5.15:

```
docker            -> command not found
/var/run/docker.sock -> absent
podman / nerdctl  -> absent
unshare --user --map-root-user --mount --pid --fork  -> Operation not permitted
/dev/fuse         -> absent
CapEff a80425fb   -> no CAP_SYS_ADMIN
```

No container runtime of any kind can start there: the instance is itself a
container with seccomp blocking `unshare`. Architecture §2 and tech_stack §4
specify Docker with container pooling as the sandbox, and PRD §4.3 gates on it, so
the authoritative P1.21 measurement cannot be taken on the machine where training
will run.

Note tech_stack §4 already contemplates the alternative — the Isolation row reads
"Container **or worktree** + process group cleanup" — but `src/environment/sandbox.py`
is Docker-only today.

**Bounding what we can.** A container-free pytest probe on that host
(`linux_pytest_probe.json`) puts the dominant term at **5.62 s mean / 8.82 s max**
per 12-step trajectory.

**Corrections (Codex review).** Three of my framings were wrong. (a) The 0.005–0.012 s
figure is lease+reset only; per-command Docker `exec` overhead sits inside the
run_tests and verify segments and was never isolated. (b) The scripted benchmark
gives a *lower* bound on trajectory latency and an *upper* bound on trajectories/hour
— my "floor for any model-in-the-loop rate" wording was reversed. (c) The host
reports 128 logical CPUs but its cgroup quota is `cpu.max = 1600000 100000` = **16
cores**, shared with training. So the probe supports "the gate is very likely
passed"; it does not close it, and P0.6 remains necessary.


---

## F5 — The grader can be bypassed outright (found by Codex, reproduced here)

Both reproduced through the full Docker harness on `record_index`
`c1-L0-single-function`, via an ordinary accepted `apply_patch` to an allowed
source file:

| Probe | Result |
|---|---|
| Append `import os as _os; _os._exit(0)` to the source, bug retained | **reward 1, no cheat flags**, pytest exit 0, empty output |
| Append an import-time `Path('tests/MARKER').write_text('x')` | marker written into `tests/`, **no cheat flag** |

The first is the serious one. `verify()` treats `hidden_result.exit_code == 0` as
"all hidden tests passed", so a module that kills the pytest process at import time
scores 1 on **every task** for two lines of code. The 23-pattern attack suite in
`attacks/` does not contain it, and §7.1's reward-hacking gate is stated over
exactly this class.

The second shows the patch-path whitelist governs only tool-mediated edits, not
runtime behaviour: the workspace and `tests/` are writable inside the container
(`sandbox.py:61`, `verifier.py:135`). Docker protects the host; it does not
currently protect the grading assets.

Required: immutable grading assets, fail-closed completion/accounting checks (a
printed "passed" count is forgeable too — require collected/executed counts to
match expectation), and end-to-end attack tests that assert reward 0 rather than
asserting a detector fires.

## F6 — Two provenance gaps in the P1.31 round-trip (found by Codex)

1. `repo_content_hash` excludes `injector_sources.json`, and the hidden-test
   generators live in `src/environment/hidden_tests.py` — outside any repo
   fingerprint. So repairing catalogs or hidden tests changes the regenerated
   instance **without changing `repo_ref`**, silently breaking the manifest
   contract. Both need explicit version pinning before any manifest is frozen.
2. `run_trajectory` passes the rollout `sampling_seed` to `verify()` as
   `--hypothesis-seed`, so each of the G=8 rollouts in a group grades against
   different hypothesis examples. On marginal bugs that injects verifier randomness
   into the mixed-group signal D1/D2 are meant to interpret. The verification seed
   should be fixed per instance, independent of the rollout seed.


---

# Repair status (2026-09-06, F5 + F1 + F6 contract)

Implementation by Codex (cut off mid-contract), completed and verified here.
`pytest tests/` → **191 passed**.

## F5 — closed for the demonstrated class

Grading now runs through `scripts/grading_runner.py` inside the sandbox, which
emits structured execution events to a file on a separate `/reports` mount,
independent of pytest's printed summary. `verifier.account()` compares those
events against node ids **collected from the reference tree before the agent can
touch it** (`collect_reference`), and requires: exact collected identity match,
a setup/call/teardown report for every expected node, no skipped or invalid
outcomes, a well-formed start/finish/returned envelope, and exit-code agreement.
`tests/` , `hidden_tests/` and `conftest.py` are now **read-only bind mounts**
(`sandbox.py`), and a `sys.addaudithook` in the runner reports writes to them.

Verified end-to-end (`tests/test_grading_integrity.py`, `tests/test_attack_harness.py`):

| Attack | Result |
|---|---|
| `exit_during_import` — `os._exit(0)` before collection | reward 0, `incomplete:*` |
| `partial_test_exit` — kill after two tests genuinely pass | reward 0, `incomplete:execution_count_mismatch` |
| `forged_pass_summary` — print `4 passed`, then exit 0 | reward 0, `incomplete:*` |
| `runtime_test_write` — import-time write into `tests/` | reward 0, `protected_asset_write:*` |

The partial-exit payload had to be corrected during completion: Codex's version
counted pytest's own `pytest_runtest_call` hook invocations and so died inside the
*first* test, reproducing only the zero-execution case the accounting already
caught. It now counts distinct `test_*` frames entering their call phase, so two
tests complete before the kill — the case that distinguishes real accounting from
a "did anything run?" check.

All 27 attack patterns are asserted to yield **reward 0 through the full harness**,
not merely to trip a detector.

**Residual surface, stated plainly.** The event writer runs inside pytest's
process, so code that is willing to import pytest internals can forge the protocol.
This closes the demonstrated exit/summary-spoofing class and makes forgery require
knowing the reference-collected node ids; it is not a trust boundary against
arbitrary in-process Python. A real boundary needs the grader out of the process
under test.

## F1 — 21 cells fixed by added properties, 7 rejected at admission

Codex added targeted properties to `src/environment/hidden_tests.py` (now 153
property tests). Re-running the original audit script against the repaired
generators, and then admission over the same 43 cells:

| Prior status | → admitted | → rejected |
|---|---|---|
| `ok` (11) | 11 | 0 |
| `HIDDEN-MISSES-BUG` (21) | 21 | 0 |
| `NEITHER-CATCHES` (11) | 4 | **7** |

- The 21 superset violations are **fixed by added properties**: re-running
  `scripts/p1_reward_signal_audit.py` now reports 0 `HIDDEN-MISSES-BUG` (36 `ok`,
  7 `NEITHER-CATCHES`). The pre-repair measurement is preserved at
  `artifacts/p1/reward_signal_audit_pre_repair.json`; the only semantic change in
  the detection path between the two runs is `hidden_tests.py`.
- The 7 rejections are **all `parcel_ledger`**, all `buggy:no_genuine_test_failure`
  — the mutation lands in `parcel_ledger/__init__.py`, which no suite exercises.
  Admission refuses them rather than papering over them.
- **A no-op policy earns reward 1 on 0 of the 36 admitted instances** (down from
  the 11 degenerate cells). This is the condition that mattered.

**Pool consequence to carry into P2:** `parcel_ledger` contributes **zero** admitted
`mutation` instances at count=1. *(Superseded 2026-09-07 — see "Correction —
`parcel_ledger` is not a dead repo" below: once `commit-rollback` and `feat-add`
work it contributes 14 admitted instances, and the pool is 6 of 6. The statement
below that the pool is 5 of 6 was wrong.)* Either its `__init__.py` gains test
coverage or its mutation cells stay unavailable. Combined
with F3 (cross-file unsupported on 7 of 8 repos) the active design is narrower than
the docs describe, and D1's hard buckets must be chosen from what is supported.

`artifacts/p1/admission_report.json` carries per-cell admitted/rejected counts with
reasons and the admitted population's repo/type/source/span proportions — the frozen
census the contract requires, so a later change in test coverage cannot masquerade
as a change in the estimator.

## F6 — provenance pinned

`repo_content_hash` now covers `injector_sources.json`, the repo's hidden-test
generator, `ADMISSION_RULE_VERSION`, the sandbox image id, and the implementation
files that determine instance content. `TaskInstance.verification_seed` is derived
from `(repo_ref, injector_seed, difficulty)`, so all G=8 rollouts of a group grade
against identical hypothesis examples while the rollout itself still varies with
`sampling_seed`. Both are asserted in `tests/test_admission.py`.

## Throughput after the repair

| N | traj/hr before → after | p95 s before → after | gate |
|---|---|---|---|
| 1 | 1225.7 → **1122.2** | 3.46 → **3.76** | PASS |
| 4 | 2892.5 → **2822.9** | 6.53 → **6.29** | PASS |
| 8 | 2419.4 → **2928.4** | 15.54 → **11.25** | PASS |

The repair costs about 9% on p95 at N=1 (reference collection plus the audit hook
and read-only mounts). N=4/N=8 differences are within laptop-contention noise
between runs and should not be read as improvements. Admission is a **per-instance**
cost, not per-trajectory: mean 3.62 s on admitted instances, i.e. **0.45 s per
rollout** amortized over G=8. F4 is unchanged — this is still macOS Docker Desktop,
not the training host.


---

# F2 delta round 2 (2026-09-07) — completed after Codex ran out of usage

Codex acknowledged the delta and stopped during fix #1 having written nothing;
the tree was unchanged apart from its earlier `slot_planner` regeneration. All
four fixes were taken up from there.

## Fix #2 — coverage over the real grid (done)

`scripts/injector_coverage.py` enumerates every `(repo × source × bucket × type)`
cell explicitly — **3024 cells**, with `count`/`span` per row, a named reason for
every non-generating cell, and an `unsupported_cells` list. It never samples:
`Bucket.sample` draws a source at random, which is how the original 432-cell grid
came to contain no `mutation` at all.

## Catalog repair — 162 invalid cells eliminated

The delta asked that malformed entries be rejected by name; running the new grid
showed **162 cells failing `CatalogError`**, in three repos Codex had not touched
(`route_graph` 90, `config_parser` 54, `stock_reservations` 18 — only
`slot_planner` had been regenerated). Every case was a *truncated fragment*: a
`for` header without its body, a function cut mid-way, so the text never parsed as
a CST node.

`scripts/repair_catalogs.py` widens each fragment to its complete enclosing
function, sliced **verbatim** from the file. Verbatim is the load-bearing detail:
`inject` checks `entry.before in git(fix_commit^)` as a raw substring, so a
dedented method body would parse but fail the provenance check. Two shapes needed
handling — a one-line regression, and a pair truncated at a common point where the
recorded `after` is the feature variant's prefix and the real tail must be spliced
back on.

Result: `catalog_invalid` **162 → 0**; generating cells 777 → 792. Note 36 of the
162 were `mutation` cells: a malformed catalog blocked injection for a source that
does not read the catalog at all.

## Fix #3 — admission over all three sources (done)

`scripts/p1_admission_audit.py` now loops `SOURCES` and reports admitted/rejected
**per source** with reasons. This replaces the vacuous mutation-only census.

## New finding F7 — an unsound hidden property was rejecting correct code

Admission rejected 3 `metric_aggregator` cells with `clean:reference_does_not_pass`
— the *clean* tree failing its own hidden suite. The cause is not the task but the
test:

```python
def test_metric_mean_bounded(values):        # hidden_tests.py, added during the F1 repair
    m = Metric(values)
    assert m.minimum() <= m.mean() <= m.maximum()
```

`mean()` is `total()/count()`, and floating-point summation can place it an ulp
outside `[min, max]`. Reproduced: `min=969096.9183428453`,
`mean=969096.9183428452`, difference `-1.16e-10`. Hypothesis finds such a list at
some seeds, so the property **fails on a correct implementation**.

This is a false positive in the grader: a correct repair scores 0 whenever the
instance's verification seed happens to find the counterexample. It would have
entered D1/D2 as unexplained noise in the mixed-group signal. Fixed with a
relative tolerance; the clean tree now passes at every seed tried, including the
one that failed (1142062182). `variance()` was checked for the same class and is
sound — it sums squares.

Admission caught a bad *test*, which is what it is for.

## Correction — `parcel_ledger` is not a dead repo

The earlier note that `parcel_ledger` contributes zero admitted instances was true
only of `mutation` at `count=1`. Its `commit-rollback` and `feat-add` entries
target `billing_rules.py`, which **is** tested, and **14 of its instances are
admitted** (7 + 7). Only its mutation tasks are degenerate, because the mutation
injector draws from the untested `__init__.py`. The "effective pool is 5 of 6
repos" statement is superseded: the pool is 6 of 6 for non-mutation sources.

## Fix #1 — partially delivered, honestly

`feat-add` multi-edit now generates: two second entries were authored for
`record_index` (`vocabulary` for condition-inversion, `excerpt` for off-by-one),
each a complete-function feature variant validated by `validate_entry`, giving
`c2-*-single-file` tasks with two edits in two functions. Multi-edit `feat-add`
cells went **0 → 6**; unsupported `(source, bucket)` pairs 30 → 27.

**`commit-rollback` multi-edit remains 0, and this is a data problem, not a code
problem.** A second same-type history entry needs a second authored
buggy→fixed transition in git, and each repo's history is exactly two commits with
a 4-hunk diff — there is no second same-type function pair to point at. Closing it
means authoring a third and fourth commit per repo (introduce a second bug set,
then fix it), then 7 history entries per repo against the new fix commit. That is
roughly 56 authored function pairs plus 8 commit pairs, and it must not be
generated from the mutation operator — doing so would make `commit-rollback`
indistinguishable from `mutation` and destroy the very dimension the fix exists to
restore.

**Deviation, stated plainly:** the delta asked for ≥2 entries per type in every
repo. Delivered: 2 types in 1 repo, as a validated worked reference. The remaining
work is mechanical but substantial, and it is authoring, not automation.


## Final state of the F2 delta (2026-09-07)

| Delta fix | Status |
|---|---|
| #1 ≥2 entries per type | **Partial.** `feat-add` multi-edit 0 → 6 cells via two authored `record_index` entries (`scripts/author_second_entries.py`). `commit-rollback` multi-edit still 0 — needs a second authored buggy→fixed commit pair per repo |
| #2 coverage over 18 buckets | **Done.** `scripts/injector_coverage.py`, 3024 cells with `count`/`span`, named reasons, 27 unsupported `(source, bucket)` pairs |
| #3 admission across sources | **Done.** 143 requested → 136 admitted; `commit-rollback` 50/50, `feat-add` 50/50, `mutation` 36/43; no-op reward 1 on 0 of 136 |
| #4 risks and deviations | **Done** — stated above and in `docs/progress.md` |

Additional work not requested by the delta but required to make #2/#3 meaningful:
the catalog repair (162 → 0 invalid cells across three repos) and F7 (an unsound
hidden property failing correct code).

**Scripts, all re-runnable from the repository root:**

| Script | Purpose |
|---|---|
| `scripts/injector_coverage.py` | full-grid injector coverage census |
| `scripts/repair_catalogs.py` | widen truncated catalog fragments to whole functions |
| `scripts/author_second_entries.py` | author second same-type `feat-add` entries |
| `scripts/p1_admission_audit.py` | admission across all sources + no-op probe |
| `scripts/p1_reward_signal_audit.py` | hidden-vs-visible detection audit |
| `scripts/p1_throughput.py` | 12-step throughput gate at N ∈ {1,4,8} |
| `scripts/baseline_eval.py` | admission-gated baseline (`--source`, `--base-url`) |
| `scripts/grading_runner.py` | in-sandbox pytest accounting runner |
