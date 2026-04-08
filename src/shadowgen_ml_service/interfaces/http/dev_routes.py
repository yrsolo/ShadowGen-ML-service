from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from shadowgen_ml_service.interfaces.dev.playground import render_playground_html
from shadowgen_ml_service.interfaces.http.dev_schemas import PipelineDebugRequest


def build_dev_router(debug_use_case) -> APIRouter:
    router = APIRouter()

    @router.get("/", include_in_schema=False)
    async def root():
        from fastapi.responses import RedirectResponse

        return RedirectResponse(url="/playground", status_code=307)

    @router.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        from fastapi.responses import Response

        return Response(status_code=204)

    @router.get("/playground", response_class=HTMLResponse)
    async def playground():
        return HTMLResponse(render_playground_html())

    @router.post("/v1/dev/pipeline/run-all")
    async def run_all_pipeline(payload: PipelineDebugRequest, request: Request):
        return request.app.state.render_service.run_debug_pipeline(payload)

    @router.post("/v1/dev/pipeline/run-stage/{stage_key}")
    async def run_stage_pipeline(stage_key: str, payload: PipelineDebugRequest, request: Request):
        return request.app.state.render_service.run_debug_pipeline(payload, stop_after=stage_key)

    return router
