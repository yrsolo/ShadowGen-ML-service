from __future__ import annotations

from collections import deque
from dataclasses import replace
from datetime import datetime, timezone
from threading import Condition, Lock, Thread
from typing import Callable
from uuid import uuid4

from shadowgen_ml_service.application.models import AsyncRenderJobRecord, RenderOutcome
from shadowgen_ml_service.config import Settings
from shadowgen_ml_service.core.commands import RenderCommand
from shadowgen_ml_service.core.job_contracts import JobCapacityProvider, JobQueue, JobRepository, JobResultStore
from shadowgen_ml_service.core.models import JobExecutionInfo


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class InMemoryRenderJobManager(JobRepository, JobQueue, JobResultStore, JobCapacityProvider):
    def __init__(self, settings: Settings) -> None:
        self._jobs: dict[str, AsyncRenderJobRecord] = {}
        self._request_index: dict[str, str] = {}
        self._queue: deque[tuple[str, Callable[[], RenderOutcome]]] = deque()
        self._lock = Lock()
        self._condition = Condition(self._lock)
        self._cancelled: set[str] = set()
        self._running_jobs = 0
        self._max_running = max(1, settings.job_max_running)
        self._max_pending = max(1, settings.job_max_pending)
        self._accepting_enabled = settings.job_accepting_enabled
        self._cancel_mode = settings.job_cancel_mode
        self._queue_backend = settings.async_backend
        self._workers = [
            Thread(target=self._worker_loop, name=f"shadowgen-job-worker-{index}", daemon=True)
            for index in range(self._max_running)
        ]
        for worker in self._workers:
            worker.start()

    def create(self, command: RenderCommand) -> AsyncRenderJobRecord:
        with self._lock:
            job = AsyncRenderJobRecord(
                job_id=uuid4().hex,
                request_id=command.request_id,
                status="pending",
                created_at=_utc_now(),
                updated_at=_utc_now(),
                submit_mode="async",
                capacity_snapshot=self._snapshot_unlocked(),
            )
            self._jobs[job.job_id] = job
            if command.request_id:
                self._request_index[command.request_id] = job.job_id
            return job

    def get(self, job_id: str) -> AsyncRenderJobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def get_by_request_id(self, request_id: str) -> AsyncRenderJobRecord | None:
        with self._lock:
            job_id = self._request_index.get(request_id)
            return None if job_id is None else self._jobs.get(job_id)

    def update(self, job_id: str, *, status: str, error: str | None = None, render_outcome: RenderOutcome | None = None) -> AsyncRenderJobRecord:
        with self._lock:
            current = self._jobs[job_id]
            updated = replace(
                current,
                status=status,
                error=error,
                render_outcome=render_outcome,
                updated_at=_utc_now(),
                capacity_snapshot=self._snapshot_unlocked(),
            )
            self._jobs[job_id] = updated
            return updated

    def submit(self, job_id: str, work) -> None:
        with self._condition:
            self._queue.append((job_id, work))
            record = self._jobs.get(job_id)
            if record is not None:
                self._jobs[job_id] = replace(record, capacity_snapshot=self._snapshot_unlocked(), updated_at=_utc_now())
            self._condition.notify()

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            current = self._jobs.get(job_id)
            if current is None:
                return False
            if self._cancel_mode == "pending_only" and current.status != "pending":
                return False
            if current.status in {"completed", "failed", "cancelled"}:
                return False
            self._cancelled.add(job_id)
            self._jobs[job_id] = replace(current, status="cancelled", updated_at=_utc_now(), capacity_snapshot=self._snapshot_unlocked())
            return True

    def get_result(self, job_id: str) -> RenderOutcome | None:
        with self._lock:
            record = self._jobs.get(job_id)
            return None if record is None else record.render_outcome

    def capacity_snapshot(self) -> JobExecutionInfo:
        with self._lock:
            return self._snapshot_unlocked()

    def set_accepting(self, accepting: bool) -> None:
        with self._lock:
            self._accepting_enabled = accepting

    def _snapshot_unlocked(self) -> JobExecutionInfo:
        return JobExecutionInfo(
            queue_backend=self._queue_backend,
            accepting_jobs=self._accepting_enabled,
            max_running_jobs=self._max_running,
            max_pending_jobs=self._max_pending,
            running_jobs=self._running_jobs,
            pending_jobs=len(self._queue),
            cancel_mode=self._cancel_mode,
            idempotency_supported=True,
        )

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while not self._queue:
                    self._condition.wait()
                job_id, work = self._queue.popleft()
                self._running_jobs += 1
                current = self._jobs.get(job_id)
                if current is not None and current.status != "cancelled":
                    self._jobs[job_id] = replace(current, status="running", updated_at=_utc_now(), capacity_snapshot=self._snapshot_unlocked())
            try:
                with self._lock:
                    if job_id in self._cancelled:
                        self._cancelled.remove(job_id)
                        continue
                    if job_id not in self._jobs:
                        continue
                result = work()
            except Exception as exc:
                with self._lock:
                    if job_id in self._cancelled:
                        self._cancelled.remove(job_id)
                        current = self._jobs.get(job_id)
                        if current is not None:
                            self._jobs[job_id] = replace(current, status="cancelled", updated_at=_utc_now(), capacity_snapshot=self._snapshot_unlocked())
                        continue
                self.update(job_id, status="failed", error=str(exc))
            else:
                with self._lock:
                    if job_id in self._cancelled:
                        self._cancelled.remove(job_id)
                        current = self._jobs.get(job_id)
                        if current is not None:
                            self._jobs[job_id] = replace(current, status="cancelled", updated_at=_utc_now(), capacity_snapshot=self._snapshot_unlocked())
                        continue
                self.update(job_id, status="completed", render_outcome=result)
            finally:
                with self._lock:
                    self._running_jobs = max(0, self._running_jobs - 1)
                    current = self._jobs.get(job_id)
                    if current is not None:
                        self._jobs[job_id] = replace(current, capacity_snapshot=self._snapshot_unlocked(), updated_at=_utc_now())
