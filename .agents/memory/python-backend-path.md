---
name: Python backend path
description: The uvicorn workflow runs from the artifact directory, not workspace root — backend module won't resolve without cd
---

The `artifacts/api-server` artifact runs its dev command from `artifacts/api-server/`. The `backend/` Python package is at the workspace root. Without an explicit `cd`, uvicorn raises `ModuleNotFoundError: No module named 'backend'`.

**Fix:** Prefix the run command with `cd /home/runner/workspace &&`:
```toml
[services.development]
run = "cd /home/runner/workspace && uvicorn backend.main:app --host 0.0.0.0 --port 8080 --reload"
```

**Why:** The artifact runner sets the CWD to the artifact directory, not the monorepo root. Python's module resolution uses CWD as the base for uninstalled packages.

**How to apply:** Any time the artifact.toml for the api-server is updated, keep the `cd /home/runner/workspace &&` prefix. Same applies to production run args.
