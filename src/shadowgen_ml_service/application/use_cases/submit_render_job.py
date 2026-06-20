from __future__ import annotations

from shadowgen_ml_service.application.services.job_admission import JobAdmissionController
from shadowgen_ml_service.application.use_cases.render_pipeline import RenderPipelineUseCase
from shadowgen_ml_service.core.commands import RenderCommand
from shadowgen_ml_service.core.job_contracts import JobQueue, JobRepository
from shadowgen_ml_service.core.models import AsyncRenderJobRecord


class SubmitRenderJobUseCase:
    def __init__(
        self,
        repository: JobRepository,
        queue: JobQueue,
        render_use_case: RenderPipelineUseCase,
        admission: JobAdmissionController,
    ) -> None:
        self.repository = repository
        self.queue = queue
        self.render_use_case = render_use_case
        self.admission = admission

    def execute(self, command: RenderCommand) -> AsyncRenderJobRecord:
        if command.request_id:
            existing = self.repository.get_by_request_id(command.request_id)
            if existing is not None and existing.status in {"pending", "running", "completed"}:
                return existing
        self.admission.ensure_async_submit_allowed()
        job = self.repository.create(command)
        self.queue.submit(job.job_id, lambda: self.render_use_case.execute(command))
        return job
