from __future__ import annotations

from shadowgen_ml_service.application.dependencies import PipelineRuntime
from shadowgen_ml_service.application.models import HealthOutcome
from shadowgen_ml_service.config import Settings
from shadowgen_ml_service.core.models import HealthStatus


class GetHealthUseCase:
    def __init__(self, settings: Settings, runtime: PipelineRuntime) -> None:
        self.settings = settings
        self.runtime = runtime

    def execute(self) -> HealthOutcome:
        return HealthOutcome(
            payload=HealthStatus(
                status="ok",
                service_version=self.settings.service_version,
                active_backend_mode=self.runtime.descriptor.mode,
                async_enabled=self.runtime.descriptor.async_enabled,
            )
        )
