from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from shadowgen_ml_service.config import Settings, get_settings
from shadowgen_ml_service.pipeline.runtime import build_runtime
from shadowgen_ml_service.pipeline.service import RenderService, ServiceError
from shadowgen_ml_service.schemas import PipelineDebugRequest, RenderRequest
from shadowgen_ml_service.web_ui import render_playground_html


def create_app(settings: Settings | None = None) -> FastAPI:
    runtime_settings = settings or get_settings()
    runtime = build_runtime(runtime_settings)
    render_service = RenderService(settings=runtime_settings, runtime=runtime)

    app = FastAPI(title=runtime_settings.service_name, version=runtime_settings.service_version)
    app.state.render_service = render_service

    @app.exception_handler(ServiceError)
    async def handle_service_error(_: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status,
            content=exc.to_response().model_dump(exclude_none=True),
        )

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

    @app.get("/health")
    async def health(request: Request):
        return request.app.state.render_service.health()

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(url="/playground", status_code=307)

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return Response(status_code=204)

    @app.get("/v1/capabilities")
    async def capabilities(request: Request):
        return request.app.state.render_service.capabilities()

    @app.post("/v1/render")
    async def render(payload: RenderRequest, request: Request):
        return request.app.state.render_service.render(payload)

    @app.get("/playground", response_class=HTMLResponse)
    async def playground():
        return HTMLResponse(render_playground_html())

    @app.post("/v1/dev/pipeline/run-all")
    async def run_all_pipeline(payload: PipelineDebugRequest, request: Request):
        return request.app.state.render_service.run_debug_pipeline(payload)

    @app.post("/v1/dev/pipeline/run-stage/{stage_key}")
    async def run_stage_pipeline(stage_key: str, payload: PipelineDebugRequest, request: Request):
        return request.app.state.render_service.run_debug_pipeline(payload, stop_after=stage_key)

    return app
