from __future__ import annotations
import asyncio
import io
import json
import logging
import os
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from backend.jobs.store import Job, JobMode, job_store
from backend.spec.models import GenerationSpec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Synthetic Data Studio API", root_path="/api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUTS_DIR = Path("backend/outputs")
UPLOADS_DIR = Path("backend/uploads")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


class CreatorQueryRequest(BaseModel):
    query: str
    row_count: Optional[int] = None


class GenerateRequest(BaseModel):
    spec: Dict[str, Any]
    format: str = "csv"


@app.get("/healthz")
def health_check():
    return {"status": "ok"}


@app.post("/replicate")
async def replicate_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    row_count: Optional[int] = Form(None),
):
    content = await file.read()
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower() or ".csv"

    tmp_path = UPLOADS_DIR / f"{uuid.uuid4()}{suffix}"
    with open(tmp_path, "wb") as f:
        f.write(content)

    job = await job_store.create(mode="replicate", filename=filename)
    background_tasks.add_task(_run_replicate, job.job_id, str(tmp_path), filename, row_count)

    return {"job_id": job.job_id, "status": "pending", "message": "Profiling file..."}


async def _run_replicate(job_id: str, file_path: str, filename: str, row_count: Optional[int]):
    job = job_store.get(job_id)
    if not job:
        return

    try:
        job.set_running("Reading file...")
        await asyncio.sleep(0)

        from backend.replicator.ingest import ingest_file
        from backend.replicator.profiler import profile_to_spec

        frames = ingest_file(file_path, filename)
        job.set_progress(20, "Profiling columns...")
        await asyncio.sleep(0)

        has_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))
        semantic_labels: Dict[str, Dict[str, Any]] = {}

        if has_llm:
            try:
                from backend.llm.client import semantic_pass
                for table_name, df in frames.items():
                    profile = {col: {"type": str(df[col].dtype), "unique_count": int(df[col].nunique()), "null_pct": float(df[col].isna().mean())} for col in df.columns}
                    sample_rows = df.head(5).to_dict(orient="records")
                    labels = await semantic_pass(profile, sample_rows)
                    semantic_labels[table_name] = labels
            except Exception as e:
                logger.warning(f"LLM semantic pass failed: {e}")

        job.set_progress(50, "Building spec...")
        await asyncio.sleep(0)

        spec = profile_to_spec(frames, semantic_labels if semantic_labels else None, row_count)
        job.set_progress(70, "Generating data...")
        await asyncio.sleep(0)

        from backend.engine.generator import generate
        from backend.validation.report import generate_report

        def progress(pct, msg):
            job.set_progress(70 + pct * 0.25, msg)

        generated = generate(spec, progress_cb=progress)

        job.set_progress(90, "Generating fidelity report...")
        await asyncio.sleep(0)

        report = generate_report(generated, spec, source=frames)

        result = _build_result(generated, report, job_id)
        spec_dict = spec.model_dump()

        job.row_count = result["row_count_total"]
        job.table_count = len(generated)
        job.set_complete(result, spec_dict)

        _save_outputs(generated, job_id)

    except Exception as e:
        logger.exception(f"Replicate job {job_id} failed")
        job.set_failed(str(e))
    finally:
        try:
            Path(file_path).unlink(missing_ok=True)
        except Exception:
            pass


@app.post("/create")
async def create_from_query(body: CreatorQueryRequest):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail={"error": "Query cannot be empty", "conflicts": []})

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "ANTHROPIC_API_KEY not configured. Please add it as a Replit Secret.",
                "conflicts": ["ANTHROPIC_API_KEY environment variable is missing"]
            }
        )

    try:
        from backend.creator.parser import parse_query
        spec, summary, warnings = await parse_query(body.query, body.row_count)
        return {
            "spec": spec.model_dump(),
            "summary": summary,
            "conflicts": warnings,
        }
    except ValueError as e:
        conflicts = [str(e)]
        raise HTTPException(status_code=400, detail={"error": str(e), "conflicts": conflicts})
    except Exception as e:
        logger.exception("Create from query failed")
        raise HTTPException(status_code=500, detail={"error": str(e), "conflicts": []})


@app.post("/generate")
async def generate_from_spec(body: GenerateRequest, background_tasks: BackgroundTasks):
    try:
        spec = GenerationSpec(**body.spec)
    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": f"Invalid spec: {e}", "conflicts": [str(e)]})

    from backend.engine.validator import validate_spec
    result = validate_spec(spec)
    if not result.valid:
        conflicts = [c.message for c in result.conflicts]
        raise HTTPException(status_code=400, detail={"error": "Spec validation failed", "conflicts": conflicts})

    job = await job_store.create(mode="generate")
    background_tasks.add_task(_run_generate, job.job_id, spec, body.format)

    return {"job_id": job.job_id, "status": "pending", "message": "Generation queued"}


async def _run_generate(job_id: str, spec: GenerationSpec, fmt: str):
    job = job_store.get(job_id)
    if not job:
        return

    try:
        job.set_running("Starting generation engine...")
        await asyncio.sleep(0)

        from backend.engine.generator import generate
        from backend.validation.report import generate_report

        def progress(pct, msg):
            job.set_progress(pct, msg)

        generated = await asyncio.get_event_loop().run_in_executor(
            None, lambda: generate(spec, progress_cb=progress)
        )

        job.set_progress(90, "Computing fidelity report...")
        await asyncio.sleep(0)

        report = generate_report(generated, spec)
        result = _build_result(generated, report, job_id)

        job.row_count = result["row_count_total"]
        job.table_count = len(generated)
        job.set_complete(result, spec.model_dump())

        _save_outputs(generated, job_id)

    except Exception as e:
        logger.exception(f"Generate job {job_id} failed")
        job.set_failed(str(e))


@app.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail={"error": f"Job {job_id} not found"})
    return job.model_dump()


@app.get("/jobs")
def list_jobs():
    return job_store.list_all()


@app.get("/spec/{job_id}")
def get_job_spec(job_id: str):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail={"error": f"Job {job_id} not found"})
    if not job.spec:
        raise HTTPException(status_code=404, detail={"error": "No spec available for this job"})
    return job.spec


@app.get("/download/{job_id}/{format}")
def download_results(job_id: str, format: str):
    if format not in ("csv", "xlsx"):
        raise HTTPException(status_code=400, detail={"error": "format must be 'csv' or 'xlsx'"})

    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail={"error": f"Job {job_id} not found"})
    if job.status != "complete":
        raise HTTPException(status_code=404, detail={"error": "Job not complete yet"})

    output_dir = OUTPUTS_DIR / job_id
    if not output_dir.exists():
        raise HTTPException(status_code=404, detail={"error": "Output files not found"})

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if format == "csv":
            for csv_path in output_dir.glob("*.csv"):
                zf.write(csv_path, csv_path.name)
        else:
            for parquet_path in output_dir.glob("*.parquet"):
                df = pd.read_parquet(parquet_path)
                xlsx_buf = io.BytesIO()
                df.to_excel(xlsx_buf, index=False)
                xlsx_buf.seek(0)
                zf.writestr(parquet_path.stem + ".xlsx", xlsx_buf.read())

    buf.seek(0)
    media_type = "application/zip"
    filename = f"synthetic_data_{job_id[:8]}.zip"

    return StreamingResponse(
        buf,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


def _build_result(
    generated: Dict[str, pd.DataFrame],
    report: Dict[str, Any],
    job_id: str,
) -> Dict[str, Any]:
    tables = []
    total_rows = 0

    for name, df in generated.items():
        preview = df.head(10).replace({float("nan"): None}).to_dict(orient="records")
        total_rows += len(df)
        tables.append({
            "name": name,
            "row_count": len(df),
            "columns": list(df.columns),
            "preview": preview,
        })

    return {
        "tables": tables,
        "fidelity_report": report,
        "download_url": f"/api/download/{job_id}/csv",
        "row_count_total": total_rows,
    }


def _save_outputs(generated: Dict[str, pd.DataFrame], job_id: str) -> None:
    output_dir = OUTPUTS_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, df in generated.items():
        safe_name = name.replace(" ", "_").replace("/", "_")
        csv_path = output_dir / f"{safe_name}.csv"
        df.to_csv(csv_path, index=False)
        try:
            parquet_path = output_dir / f"{safe_name}.parquet"
            df.to_parquet(parquet_path, index=False)
        except Exception:
            pass
