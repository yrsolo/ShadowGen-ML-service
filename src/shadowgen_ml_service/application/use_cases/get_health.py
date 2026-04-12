from __future__ import annotations

from shadowgen_ml_service.application.dependencies import PipelineRuntime
from shadowgen_ml_service.application.services.job_admission import JobAdmissionController
from shadowgen_ml_service.application.models import HealthOutcome
from shadowgen_ml_service.config import Settings
from shadowgen_ml_service.core.models import HealthStatus


class GetHealthUseCase:
    def __init__(self, settings: Settings, runtime: PipelineRuntime, admission: JobAdmissionController) -> None:
        self.settings = settings
        self.runtime = runtime
        self.admission = admission

    def execute(self) -> HealthOutcome:
        snapshot = self.admission.snapshot()
        if not snapshot.accepting_jobs:
            status = "draining"
        elif snapshot.pending_jobs >= snapshot.max_pending_jobs or snapshot.running_jobs >= snapshot.max_running_jobs:
            status = "overloaded"
        elif self.runtime.descriptor.degraded:
            status = "degraded"
        else:
            status = "ok"
        return HealthOutcome(
            payload=HealthStatus(
                status=status,
                service_version=self.settings.service_version,
                active_backend_mode=self.runtime.descriptor.mode,
                async_enabled=self.runtime.descriptor.async_enabled,
                accepting_jobs=snapshot.accepting_jobs,
                preferred_submit_mode=self.admission.preferred_submit_mode(),
            )
        )
