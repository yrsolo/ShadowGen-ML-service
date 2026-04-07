from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from shadowgen_ml_service.config import Settings, get_settings
from shadowgen_ml_service.pipeline.runtime import build_runtime
from shadowgen_ml_service.pipeline.service import RenderService, ServiceError
from shadowgen_ml_service.schemas import RenderRequest


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

    @app.get("/v1/capabilities")
    async def capabilities(request: Request):
        return request.app.state.render_service.capabilities()

    @app.post("/v1/render")
    async def render(payload: RenderRequest, request: Request):
        return request.app.state.render_service.render(payload)

    return app
