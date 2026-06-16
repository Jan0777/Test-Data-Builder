---
name: File upload pattern
description: multipart/form-data uploads bypass Orval-generated hooks; use raw fetch + FormData in the page component
---

The `/api/replicate` endpoint accepts `multipart/form-data` with a `file` field. Orval does not generate typed request bodies for multipart uploads, so the generated hook body type would be `unknown`.

**Pattern used in `artifacts/studio/src/pages/replicate.tsx`:**
```ts
const fd = new FormData();
fd.append("file", file);
const res = await fetch("/api/replicate", { method: "POST", body: fd });
const data = await res.json();
if (!res.ok) throw new Error(data.error || "Upload failed");
```

**Why:** Using `fetch` directly avoids fighting Orval's type system for binary uploads. The mutation path (submit → get job_id → poll) still uses the generated `useGetJobStatus` hook.

**How to apply:** Any future file upload endpoint should follow the same pattern: raw `fetch` for the upload, then switch to generated hooks for polling/results.
