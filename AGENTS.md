## Repo docs

The living project guide is in `repo-docs/`. Start with `repo-docs/README.md`; when `repo-docs/walkthroughs/one-real-run.md` exists, use it as the main behavior trace.

Before answering repo architecture, onboarding, or "how does this work" questions, read the relevant guide pages and inspect the current source behind the answer. If the guide is missing, stale, or wrong, update the smallest owning page in the same turn before answering.

When code, config, data, scripts, tests, or behavior changes, run an Understanding Sync check before finishing. Patch only the guide pages that would otherwise mislead the next reader. Record meaningful guide updates in `repo-docs/change-log.md` with verification and `Synced through <sha>` when git is available.

## Experiment error recovery

The user authorizes direct diagnosis, scoped repair, verification, and continuation when an already-authorized experiment queue stops because of a program or execution error. Preserve completed valid runs, original data, the agreed protocols, baseline, seeds, and validation gates; resume from the failed phase without asking again for routine repair approval. Seek direction when recovery requires a new scientific protocol, destructive source-data changes, new spending, or other authority beyond the experiment already in scope. Report the cause, fix, and resumed state.

