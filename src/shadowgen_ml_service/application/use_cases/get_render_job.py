from __future__ import annotations

from shadowgen_ml_service.application.models import AsyncRenderJobRecord
from shadowgen_ml_service.core.errors import ValidationServiceError
from shadowgen_ml_service.core.job_contracts import JobRepository


class GetRenderJobUseCase:
    def __init__(self, repository: JobRepository) -> None:
        self.repository = repository

    def execute(self, job_id: str) -> AsyncRenderJobRecord:
        record = self.repository.get(job_id)
        if record is None:
            raise ValidationServiceError(f"unknown job_id: {job_id}")
        return record
