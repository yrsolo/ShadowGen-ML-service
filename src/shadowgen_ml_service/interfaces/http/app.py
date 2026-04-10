from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from shadowgen_ml_service.application.use_cases.debug_pipeline import DebugPipelineUseCase
from shadowgen_ml_service.application.use_cases.cancel_render_job import CancelRenderJobUseCase
from shadowgen_ml_service.application.use_cases.get_render_job import GetRenderJobUseCase
from shadowgen_ml_service.application.use_cases.get_capabilities import GetCapabilitiesUseCase
from shadowgen_ml_service.application.use_cases.get_health import GetHealthUseCase
from shadowgen_ml_service.application.use_cases.render_pipeline import RenderPipelineUseCase
from shadowgen_ml_service.application.use_cases.submit_render_job import SubmitRenderJobUseCase
from shadowgen_ml_service.bootstrap.container import build_runtime
from shadowgen_ml_service.config import Settings, get_settings
from shadowgen_ml_service.core.errors import ServiceError
from shadowgen_ml_service.infrastructure.jobs.in_memory import InMemoryRenderJobManager
from shadowgen_ml_service.pipeline.service import RenderService
from shadowgen_ml_service.interfaces.http.dev_routes import build_dev_router
from shadowgen_ml_service.interfaces.http.mappers import service_error_to_response
from shadowgen_ml_service.interfaces.http.public_routes import build_public_router


def create_app(settings: Settings | None = None) -> FastAPI:
    runtime_settings = settings or get_settings()
    runtime = build_runtime(runtime_settings)

    health_use_case = GetHealthUseCase(runtime_settings, runtime)
    capabilities_use_case = GetCapabilitiesUseCase(runtime_settings, runtime)
    render_use_case = RenderPipelineUseCase(runtime_settings, runtime)
    debug_use_case = DebugPipelineUseCase(runtime_settings, runtime)
    job_manager = InMemoryRenderJobManager()
    submit_job_use_case = SubmitRenderJobUseCase(job_manager, job_manager, render_use_case)
    get_job_use_case = GetRenderJobUseCase(job_manager)
    cancel_job_use_case = CancelRenderJobUseCase(job_manager, job_manager)

    app = FastAPI(title=runtime_settings.service_name, version=runtime_settings.service_version)
    app.state.runtime = runtime
    app.state.settings = runtime_settings
    app.state.health_use_case = health_use_case
    app.state.capabilities_use_case = capabilities_use_case
    app.state.render_use_case = render_use_case
    app.state.debug_use_case = debug_use_case
    app.state.submit_job_use_case = submit_job_use_case
    app.state.get_job_use_case = get_job_use_case
    app.state.cancel_job_use_case = cancel_job_use_case
    app.state.render_service = RenderService(
        settings=runtime_settings,
        runtime=runtime,
        health_use_case=health_use_case,
        capabilities_use_case=capabilities_use_case,
        render_use_case=render_use_case,
        debug_use_case=debug_use_case,
        submit_job_use_case=submit_job_use_case,
        get_job_use_case=get_job_use_case,
        cancel_job_use_case=cancel_job_use_case,
    )

    @app.exception_handler(ServiceError)
    async def handle_service_error(_: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content=service_error_to_response(exc).model_dump(exclude_none=True))

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "request validation failed",
                    "details": {"errors": str(exc.errors())},
                }
            },
        )

    app.include_router(build_public_router(health_use_case, capabilities_use_case, render_use_case))
    app.include_router(build_dev_router(debug_use_case))
    return app
