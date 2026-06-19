from __future__ import annotations

from shadowgen_ml_service.application.services.job_admission import JobAdmissionController
from shadowgen_ml_service.application.use_cases.cancel_render_job import CancelRenderJobUseCase
from shadowgen_ml_service.application.use_cases.debug_pipeline import DebugPipelineUseCase
from shadowgen_ml_service.application.use_cases.get_capabilities import GetCapabilitiesUseCase
from shadowgen_ml_service.application.use_cases.get_health import GetHealthUseCase
from shadowgen_ml_service.application.use_cases.get_render_job import GetRenderJobUseCase
from shadowgen_ml_service.application.use_cases.render_pipeline import RenderPipelineUseCase
from shadowgen_ml_service.application.use_cases.submit_render_job import SubmitRenderJobUseCase
from shadowgen_ml_service.interfaces.http.mappers import (
    async_job_to_response,
    async_job_to_submit_response,
    capabilities_outcome_to_response,
    debug_outcome_to_response,
    debug_request_to_command,
    health_outcome_to_response,
    render_outcome_to_response,
    render_request_to_command,
)


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
        admission: JobAdmissionController,
        submit_job_use_case: SubmitRenderJobUseCase,
        get_job_use_case: GetRenderJobUseCase,
        cancel_job_use_case: CancelRenderJobUseCase,
    ) -> None:
        self.settings = settings
        self.runtime = runtime
        self._health_use_case = health_use_case
        self._capabilities_use_case = capabilities_use_case
        self._render_use_case = render_use_case
        self._debug_use_case = debug_use_case
        self._admission = admission
        self._submit_job_use_case = submit_job_use_case
        self._get_job_use_case = get_job_use_case
        self._cancel_job_use_case = cancel_job_use_case

    def health(self):
        return health_outcome_to_response(self._health_use_case.execute())

    def capabilities(self):
        return capabilities_outcome_to_response(self._capabilities_use_case.execute())

    def render(self, payload):
        self._admission.ensure_sync_submit_allowed()
        with self._admission.track_sync_execution():
            return render_outcome_to_response(self._render_use_case.execute(render_request_to_command(payload)))

    def run_debug_pipeline(self, payload, stop_after: str | None = None):
        return debug_outcome_to_response(self._debug_use_case.execute(debug_request_to_command(payload), stop_after=stop_after))

    def submit_render_job(self, payload):
        return async_job_to_submit_response(self._submit_job_use_case.execute(render_request_to_command(payload)))

    def get_render_job(self, job_id: str):
        return async_job_to_response(self._get_job_use_case.execute(job_id))

    def cancel_render_job(self, job_id: str):
        return async_job_to_response(self._cancel_job_use_case.execute(job_id))
