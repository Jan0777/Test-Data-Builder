# Synthetic Data Studio

A full-stack web app that generates synthetic tabular data. Two modes: **Replicator** (upload file → learn structure → generate replica) and **Creator** (natural language → Claude parses → generate). Both share a Generation Spec (JSON) consumed by a single Python engine.

## Run & Operate

- `cd /home/runner/workspace && uvicorn backend.main:app --host 0.0.0.0 --port 8080 --reload` — run the API server directly
- API server workflow auto-starts via artifact runner
- Frontend workflow auto-starts via artifact runner

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- Frontend: React 19 + Vite, TanStack Query, Wouter, shadcn/ui, Recharts
- API: Python 3.12 + FastAPI + Uvicorn (at `/api`)
- Data: Pandas, NumPy, SciPy, Faker
- LLM: Anthropic Claude (via `ANTHROPIC_API_KEY` secret) — used by Creator mode and Replicator semantic pass
- No database — jobs stored in-memory (job store in `backend/jobs/store.py`)

## Where things live

```
backend/
  main.py              — FastAPI entrypoint, all route handlers
  spec/models.py       — GenerationSpec + ColumnSpec Pydantic models
  jobs/store.py        — In-memory job store with Job model
  engine/generator.py  — Core generation engine (strategy-based)
  engine/validator.py  — Spec validation
  engine/topological.py — Topological sort for table ordering
  replicator/ingest.py  — File ingestion (CSV, XLSX, JSON)
  replicator/profiler.py — Column profiling → GenerationSpec
  creator/parser.py    — Claude LLM → GenerationSpec
  llm/client.py        — Anthropic API client
  validation/report.py — Fidelity report generation
  outputs/{jobId}/     — Generated CSV/Parquet files
  uploads/             — Temporary upload files (deleted after processing)

artifacts/studio/src/
  App.tsx              — Router (/, /replicate, /create, /jobs/:jobId)
  pages/home.tsx       — Dashboard with mode cards + recent jobs
  pages/replicate.tsx  — Upload → profile → review → generate flow
  pages/create.tsx     — NL query → spec → review → generate flow
  pages/job-results.tsx — Status polling + fidelity report + data preview

lib/api-client-react/  — Generated React Query hooks from OpenAPI spec
lib/api-spec/          — OpenAPI spec (openapi.yaml)
```

## Architecture decisions

- **Contract-first**: OpenAPI spec at `lib/api-spec/openapi.yaml` drives all generated hooks via Orval
- **Single engine**: Both Replicator and Creator produce a `GenerationSpec` consumed by the same `backend/engine/generator.py`
- **Background jobs**: `/replicate` and `/generate` enqueue async FastAPI background tasks; frontend polls `/api/jobs/{jobId}` at 1.5s intervals
- **File upload uses raw fetch**: The `/replicate` endpoint takes `multipart/form-data`; Orval doesn't generate typed upload bodies, so the Replicator page uses `fetch()` with `FormData` directly
- **Python path fix**: API server artifact runs from `artifacts/api-server/` but `backend/` is at workspace root — run command uses `cd /home/runner/workspace &&` prefix
- **No database**: Jobs are in-memory — they don't survive server restarts. Acceptable for a studio tool where you re-run as needed.
- **Semantic types**: Column `type` is one of `integer|float|string|categorical|datetime|boolean`. Semantic annotations (name, email, address, etc.) go in `semantic_type` and drive Faker method selection in the generator.

## Product

- Replicator: Upload CSV/XLSX → statistical profiling of distributions, correlations, cardinality → synthesize a replica that matches the source's structure
- Creator: Describe data in natural language ("1000 customers with 1-5 orders each") → Claude interprets → generate
- Results: Fidelity score (KS statistic per numeric column, overlap for categoricals), referential integrity %, data preview table, CSV/XLSX download

## User preferences

_Populate as you build — explicit user instructions worth remembering across sessions._

## Gotchas

- `ANTHROPIC_API_KEY` must be set as a Replit Secret for Creator mode and Replicator's LLM semantic pass. Without it, Creator returns a 400 error, and Replicator falls back to pure statistical profiling.
- Run command must `cd /home/runner/workspace` before uvicorn because `backend/` is at workspace root, not inside `artifacts/api-server/`.
- Column types in GenerationSpec are strict literals — don't pass semantic labels like `email` as the `type` field; use `type: "string"` + `semantic_type: "email"`.

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
- OpenAPI spec: `lib/api-spec/openapi.yaml`
- Generated hooks: `lib/api-client-react/src/generated/api.ts`
