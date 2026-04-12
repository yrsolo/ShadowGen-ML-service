from __future__ import annotations

from fastapi import APIRouter, Request

from shadowgen_ml_service.interfaces.http.public_schemas import RenderRequest


def build_public_router(
    health_use_case,
    capabilities_use_case,
    render_use_case,
) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    async def health(request: Request):
        return request.app.state.render_service.health()

    @router.get("/v1/capabilities")
    async def capabilities(request: Request):
        return request.app.state.render_service.capabilities()

    @router.post("/v1/render")
    async def render(payload: RenderRequest, request: Request):
        return request.app.state.render_service.render(payload)

    @router.post("/v1/render/jobs", status_code=202)
    async def submit_render_job(payload: RenderRequest, request: Request):
        return request.app.state.render_service.submit_render_job(payload)

    @router.get("/v1/render/jobs/{job_id}")
    async def get_render_job(job_id: str, request: Request):
        return request.app.state.render_service.get_render_job(job_id)

    @router.delete("/v1/render/jobs/{job_id}")
    async def cancel_render_job(job_id: str, request: Request):
        return request.app.state.render_service.cancel_render_job(job_id)

    return router
