## Repo docs

The living project guide is in `repo-docs/`. Start with `repo-docs/README.md`; when `repo-docs/walkthroughs/one-real-run.md` exists, use it as the main behavior trace.

Before answering repo architecture, onboarding, or "how does this work" questions, read the relevant guide pages and inspect the current source behind the answer. If the guide is missing, stale, or wrong, update the smallest owning page in the same turn before answering.

When code, config, data, scripts, tests, or behavior changes, run an Understanding Sync check before finishing. Patch only the guide pages that would otherwise mislead the next reader. Record meaningful guide updates in `repo-docs/change-log.md` with verification and `Synced through <sha>` when git is available.

## PowerShell to SSH command quoting

This Codex environment runs local shell commands through Windows PowerShell. When sending commands to a remote POSIX shell over `ssh`, do not place complex Bash or Python code directly inside a double-quoted PowerShell string.

Avoid this pattern for remote commands that contain `$`, `$()`, command substitution, `<` / `>`, heredocs, nested quotes, backticks, semicolons with Python, or multiline code:

```powershell
ssh host "cd /repo && python - <<'PY'
...
PY"
```

PowerShell may parse those tokens locally before `ssh` receives them. Prefer a single-quoted PowerShell here-string piped to remote Bash:

```powershell
@'
cd /repo || exit 1
python - <<'PY'
from pathlib import Path
print(Path("outputs").exists())
PY
'@ | ssh host 'bash -s'
```

For nontrivial Python, pipe the Python program directly and keep the remote working directory in Bash:

```powershell
@'
from pathlib import Path
print(Path("outputs").exists())
'@ | ssh host 'cd /repo && python -'
```

For simple remote checks, prefer single-quoted remote commands when no local interpolation is needed:

```powershell
ssh host 'cd /repo && wc -l outputs/window_index/window_index.jsonl'
```

Before running a remote command from PowerShell, check where each special token will be parsed: local PowerShell first, then remote shell. If the command needs Bash features, protect it with a single-quoted here-string piped to `ssh ... 'bash -s'`.
