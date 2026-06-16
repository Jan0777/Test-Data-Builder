from __future__ import annotations
import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


JobStatus = Literal["pending", "running", "complete", "failed"]
JobMode = Literal["replicate", "create", "generate"]


class Job(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: JobStatus = "pending"
    mode: JobMode = "generate"
    progress: Optional[float] = None
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    spec: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    filename: Optional[str] = None
    row_count: Optional[int] = None
    table_count: Optional[int] = None

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def set_running(self, message: str = "Running...") -> None:
        self.status = "running"
        self.message = message
        self.progress = 0.0
        self.touch()

    def set_progress(self, pct: float, message: str = "") -> None:
        self.progress = min(100.0, max(0.0, pct))
        if message:
            self.message = message
        self.touch()

    def set_complete(self, result: Dict[str, Any], spec: Optional[Dict[str, Any]] = None) -> None:
        self.status = "complete"
        self.progress = 100.0
        self.result = result
        self.spec = spec
        self.message = "Complete"
        self.touch()

    def set_failed(self, error: str) -> None:
        self.status = "failed"
        self.error = error
        self.message = f"Failed: {error}"
        self.touch()

    def summary(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "mode": self.mode,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "row_count": self.row_count,
            "table_count": self.table_count,
            "filename": self.filename,
        }


class JobStore:
    def __init__(self, max_jobs: int = 100):
        self._jobs: Dict[str, Job] = {}
        self._order: List[str] = []
        self._max = max_jobs
        self._lock = asyncio.Lock()

    async def create(self, mode: JobMode, filename: Optional[str] = None) -> Job:
        async with self._lock:
            job = Job(mode=mode, filename=filename)
            self._jobs[job.job_id] = job
            self._order.append(job.job_id)
            if len(self._order) > self._max:
                old = self._order.pop(0)
                self._jobs.pop(old, None)
            return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def list_all(self, limit: int = 20) -> List[Dict[str, Any]]:
        recent = list(reversed(self._order))[:limit]
        return [self._jobs[jid].summary() for jid in recent if jid in self._jobs]


job_store = JobStore()
