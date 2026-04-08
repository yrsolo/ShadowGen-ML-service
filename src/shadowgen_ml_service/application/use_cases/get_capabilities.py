from __future__ import annotations

from shadowgen_ml_service.application.dependencies import PipelineRuntime
from shadowgen_ml_service.application.models import CapabilitiesOutcome
from shadowgen_ml_service.config import Settings
from shadowgen_ml_service.core.models import CapabilitiesInfo


class GetCapabilitiesUseCase:
    def __init__(self, settings: Settings, runtime: PipelineRuntime) -> None:
        self.settings = settings
        self.runtime = runtime

    def execute(self) -> CapabilitiesOutcome:
        return CapabilitiesOutcome(
            payload=CapabilitiesInfo(
                service_version=self.settings.service_version,
                model_version=self.runtime.descriptor.model_version,
                supported_input_mime_types=("image/jpeg", "image/png", "image/webp"),
                supported_output_formats=("png", "webp"),
                supports_debug_artifacts=True,
                max_image_bytes=self.settings.max_image_bytes,
                active_backend_mode=self.runtime.descriptor.mode,
                degraded=self.runtime.descriptor.degraded,
                components=self.runtime.descriptor.components,
            )
        )
