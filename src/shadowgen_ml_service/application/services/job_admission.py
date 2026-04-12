from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from threading import Lock

from shadowgen_ml_service.config import Settings
from shadowgen_ml_service.core.errors import AsyncDisabledServiceError, NotAcceptingJobsServiceError, QueueFullServiceError
from shadowgen_ml_service.core.job_contracts import JobCapacityProvider


class JobAdmissionController:
    def __init__(self, settings: Settings, capacity: JobCapacityProvider) -> None:
        self.settings = settings
        self.capacity = capacity
        self._sync_lock = Lock()
        self._sync_running = 0

    def snapshot(self):
        snapshot = self.capacity.capacity_snapshot()
        with self._sync_lock:
            sync_running = self._sync_running
        return replace(snapshot, running_jobs=snapshot.running_jobs + sync_running)

    def preferred_submit_mode(self) -> str:
        return "async" if self.settings.async_enabled else "sync"

    def supported_submit_modes(self) -> tuple[str, ...]:
        return ("sync", "async") if self.settings.async_enabled else ("sync",)

    def batching_strategy(self) -> str:
        return "internal-stage-microbatch" if self.settings.batching_enabled else "none"

    def ensure_async_submit_allowed(self) -> None:
        if not self.settings.async_enabled:
            raise AsyncDisabledServiceError("async job submission is disabled")
        snapshot = self.snapshot()
        if not snapshot.accepting_jobs:
            raise NotAcceptingJobsServiceError("service is not accepting new async jobs")
        if snapshot.pending_jobs >= snapshot.max_pending_jobs:
            raise QueueFullServiceError("async job queue is full")

    def ensure_sync_submit_allowed(self) -> None:
        snapshot = self.snapshot()
        if not snapshot.accepting_jobs:
            raise NotAcceptingJobsServiceError("service is not accepting new sync jobs")
        if snapshot.running_jobs >= snapshot.max_running_jobs:
            raise QueueFullServiceError("service is overloaded")

    @contextmanager
    def track_sync_execution(self):
        with self._sync_lock:
            self._sync_running += 1
        try:
            yield
        finally:
            with self._sync_lock:
                self._sync_running = max(0, self._sync_running - 1)
