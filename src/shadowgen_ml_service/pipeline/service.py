from shadowgen_ml_service.application.use_cases.debug_pipeline import DebugPipelineUseCase
from shadowgen_ml_service.application.use_cases.get_capabilities import GetCapabilitiesUseCase
from shadowgen_ml_service.application.use_cases.get_health import GetHealthUseCase
from shadowgen_ml_service.application.use_cases.render_pipeline import RenderPipelineUseCase
from shadowgen_ml_service.core.errors import (
    ProcessingFailedServiceError,
    ServiceError,
    TimeoutServiceError,
    UnsupportedInputServiceError,
    ValidationServiceError,
)
from shadowgen_ml_service.interfaces.http.mappers import (
    capabilities_outcome_to_response,
    debug_outcome_to_response,
    health_outcome_to_response,
    render_outcome_to_response,
)
from shadowgen_ml_service.interfaces.http.mappers import debug_request_to_command, render_request_to_command


class RenderService:
    def __init__(
        self,
        *,
        settings,
        runtime,
        health_use_case: GetHealthUseCase,
        capabilities_use_case: GetCapabilitiesUseCase,
        render_use_case: RenderPipelineUseCase,
        debug_use_case: DebugPipelineUseCase,
    ) -> None:
        self.settings = settings
        self.runtime = runtime
        self._health_use_case = health_use_case
        self._capabilities_use_case = capabilities_use_case
        self._render_use_case = render_use_case
        self._debug_use_case = debug_use_case

    def health(self):
        return health_outcome_to_response(self._health_use_case.execute())

    def capabilities(self):
        return capabilities_outcome_to_response(self._capabilities_use_case.execute())

    def render(self, payload):
        return render_outcome_to_response(self._render_use_case.execute(render_request_to_command(payload)))

    def run_debug_pipeline(self, payload, stop_after: str | None = None):
        return debug_outcome_to_response(self._debug_use_case.execute(debug_request_to_command(payload), stop_after=stop_after))

__all__ = [
    "ProcessingFailedServiceError",
    "RenderService",
    "ServiceError",
    "TimeoutServiceError",
    "UnsupportedInputServiceError",
    "ValidationServiceError",
]
