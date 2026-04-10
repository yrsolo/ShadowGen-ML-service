from __future__ import annotations

from shadowgen_ml_service.application.models import AsyncRenderJobRecord
from shadowgen_ml_service.core.errors import ValidationServiceError
from shadowgen_ml_service.core.job_contracts import JobQueue, JobRepository


class CancelRenderJobUseCase:
    def __init__(self, repository: JobRepository, queue: JobQueue) -> None:
        self.repository = repository
        self.queue = queue

    def execute(self, job_id: str) -> AsyncRenderJobRecord:
        record = self.repository.get(job_id)
        if record is None:
            raise ValidationServiceError(f"unknown job_id: {job_id}")
        self.queue.cancel(job_id)
        updated = self.repository.get(job_id)
        if updated is None:
            raise ValidationServiceError(f"unknown job_id: {job_id}")
        return updated
