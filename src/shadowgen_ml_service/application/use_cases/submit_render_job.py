from __future__ import annotations

from shadowgen_ml_service.application.models import AsyncRenderJobRecord
from shadowgen_ml_service.application.use_cases.render_pipeline import RenderPipelineUseCase
from shadowgen_ml_service.core.commands import RenderCommand
from shadowgen_ml_service.core.job_contracts import JobQueue, JobRepository


class SubmitRenderJobUseCase:
    def __init__(self, repository: JobRepository, queue: JobQueue, render_use_case: RenderPipelineUseCase) -> None:
        self.repository = repository
        self.queue = queue
        self.render_use_case = render_use_case

    def execute(self, command: RenderCommand) -> AsyncRenderJobRecord:
        job = self.repository.create(command)
        self.queue.submit(job.job_id, lambda: self.render_use_case.execute(command))
        return job
