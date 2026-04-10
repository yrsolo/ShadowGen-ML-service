from __future__ import annotations

from collections import deque
from dataclasses import replace
from datetime import datetime, timezone
from threading import Condition, Lock, Thread
from typing import Callable
from uuid import uuid4

from shadowgen_ml_service.application.models import AsyncRenderJobRecord, RenderOutcome
from shadowgen_ml_service.core.commands import RenderCommand
from shadowgen_ml_service.core.job_contracts import JobQueue, JobRepository, JobResultStore


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class InMemoryRenderJobManager(JobRepository, JobQueue, JobResultStore):
    def __init__(self) -> None:
        self._jobs: dict[str, AsyncRenderJobRecord] = {}
        self._queue: deque[tuple[str, Callable[[], RenderOutcome]]] = deque()
        self._lock = Lock()
        self._condition = Condition(self._lock)
        self._cancelled: set[str] = set()
        self._worker = Thread(target=self._worker_loop, name="shadowgen-job-worker", daemon=True)
        self._worker.start()

    def create(self, command: RenderCommand) -> AsyncRenderJobRecord:
        with self._lock:
            job = AsyncRenderJobRecord(
                job_id=uuid4().hex,
                request_id=command.request_id,
                status="pending",
                created_at=_utc_now(),
                updated_at=_utc_now(),
            )
            self._jobs[job.job_id] = job
            return job

    def get(self, job_id: str) -> AsyncRenderJobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, *, status: str, error: str | None = None, render_outcome: RenderOutcome | None = None) -> AsyncRenderJobRecord:
        with self._lock:
            current = self._jobs[job_id]
            updated = replace(
                current,
                status=status,
                error=error,
                render_outcome=render_outcome,
                updated_at=_utc_now(),
            )
            self._jobs[job_id] = updated
            return updated

    def submit(self, job_id: str, work) -> None:
        with self._condition:
            self._queue.append((job_id, work))
            self._condition.notify()

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            current = self._jobs.get(job_id)
            if current is None:
                return False
            if current.status != "pending":
                return False
            self._cancelled.add(job_id)
            self._jobs[job_id] = replace(current, status="cancelled", updated_at=_utc_now())
            return True

    def get_result(self, job_id: str) -> RenderOutcome | None:
        with self._lock:
            record = self._jobs.get(job_id)
            return None if record is None else record.render_outcome

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while not self._queue:
                    self._condition.wait()
                job_id, work = self._queue.popleft()
            with self._lock:
                if job_id in self._cancelled:
                    self._cancelled.remove(job_id)
                    continue
                if job_id not in self._jobs:
                    continue
                self._jobs[job_id] = replace(self._jobs[job_id], status="running", updated_at=_utc_now())
            try:
                result = work()
            except Exception as exc:
                self.update(job_id, status="failed", error=str(exc))
                continue
            self.update(job_id, status="completed", render_outcome=result)
