# Windows detached-worker qualification — 7 September 2026

**The local worker survived launcher exit. Restricted hosts refused startup
before creating or authorizing a worker.** These are separate outcomes.

The final implementation checks Windows breakaway support with an isolated
`python -I -S -c pass` process inside the startup deadline. A denied probe
returns `unsupported_reason: windows_breakaway_denied`; the approved journal
remains available for foreground resume. It never removes the breakaway flag
or retries the actual worker. Supported launches retain the lease and
one-use authorization handshake.

[Qualification record](qualification.json) binds the runtime and both test
files by SHA-256 after LF normalization, with hashes verified before and after
each run. On Windows (kernel 10.0.26200), CPython 3.11.9 in a virtual environment:

| Check | Result | Evidence |
|---|---|---|
| Complete operator and CLI files | 122 passed, 179.66 seconds | [Combined log](logs/combined.log) |
| Complete CLI file after one assertion refinement | 21 passed, 25.04 seconds | [Final CLI log](logs/final-cli.log) |
| Supported local host | Launcher exited while the worker's offline call remained blocked; worker then completed exactly one call and one artifact | [Positive proof](proofs/supported-local-host.json) |
| Restrictive job, direct interpreter | Refused before worker startup; zero calls | [Proof](proofs/restricted.json) |
| Kill-on-close job, direct interpreter | Refused before worker startup; zero calls | [Proof](proofs/kill_on_close.json) |
| Restrictive job, virtual-environment redirector | Refused before worker startup; zero calls | [Proof](proofs/restricted_venv.json) |

The final CLI refinement explicitly checks one succeeded journal call before
recording that result. The runtime and operator unit tests stayed unchanged;
the 21 CLI cases are a rerun, not additional distinct cases. Provider work used
offline providers and controlled fakes. **Zero paid provider calls.** Timing
here describes test duration, not worker performance.

The evidence does not claim survival when Windows denies breakaway, immunity
to a supervisor terminating an enclosing job, or a completed release CI run.
A naturally restricted CI host can verify refusal without proving positive
survival. The retained local positive proof exercised actual survival.

The [initial Windows CI failure](diagnostics/initial-windows-ci-failure.log)
belongs to commit `c30a6629`, before this repair. A later console-only fallback
was rejected: two restricted-host cases lost the worker after launcher exit.
Its [retained qualification summary](diagnostics/fallback-qualification-summary.txt)
is a diagnostic narrative, not a raw pytest log or a final-source binding.
That run had test edits in flight and is excluded from current qualification.
The [no-work probe log](diagnostics/restricted-job-probe.log) and its
[source text](diagnostics/restricted-job-probe.py.txt) demonstrate the original
access-denied condition; successful creation without breakaway did not prove
worker lifetime. These diagnostics remain separate from the final result.

Windows venv and `py.exe` launchers can own kill-on-close jobs, and nested job
policy can prevent descendants from leaving them. The repair refuses denied
probes for every interpreter context rather than inferring safety from its
prefix. See [CPython's launcher implementation](https://github.com/python/cpython/blob/v3.12.10/PC/launcher.c#L717-L777)
and [Microsoft's nested-job rules](https://learn.microsoft.com/en-us/windows/win32/procthread/nested-jobs).
