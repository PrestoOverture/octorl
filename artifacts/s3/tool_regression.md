# Tool regression found during the excluded pilot

A deterministic local check supplied a valid zero-context unified diff changing only line 2 of `x.py` from `b = 2` to `b = 4`. The initial tool returned exit 1 (`patch does not apply`). The same input passed `git apply --check --unidiff-zero` with exit 0. This isolates Git's context requirement from incorrect source content or path resolution.

The fix adds `--unidiff-zero` to validation and application and supplies a final newline when a JSON diff string omits it. It does not repair JSON, recount malformed hunks, or change paths. `measure.py selftest` now applies the zero-context patch without a terminal newline and asserts the exact resulting source. All ten original bug/revert validations also pass with this tool.

The partial pilot was stopped and retained in `excluded_pilot_01/`; none of its outputs contribute to the reported gate. The final run restarts all task/rollout seeds unchanged. This is an instrument correction, not outcome-based selection of trajectories.

Before restart, a separate parser test also locked down exclusion of an unfinished thinking-only turn containing a hypothetical `read_file(...)` mention. Thinking text is not a tool call attempt. The parser now excludes such turns, while preserving the native tool-call counting rule for emitted calls.
