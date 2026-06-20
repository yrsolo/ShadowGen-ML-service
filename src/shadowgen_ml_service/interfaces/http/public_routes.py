from __future__ import annotations

from fastapi import APIRouter

from shadowgen_ml_service.interfaces.http.public_schemas import RenderRequest


def build_public_router(render_service) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    async def health():
        return render_service.health()

    @router.get("/v1/capabilities")
    async def capabilities():
        return render_service.capabilities()

    @router.post("/v1/render")
    async def render(payload: RenderRequest):
        return render_service.render(payload)

    @router.post("/v1/render/jobs", status_code=202)
    async def submit_render_job(payload: RenderRequest):
        return render_service.submit_render_job(payload)

    @router.get("/v1/render/jobs/{job_id}")
    async def get_render_job(job_id: str):
        return render_service.get_render_job(job_id)

    @router.delete("/v1/render/jobs/{job_id}")
    async def cancel_render_job(job_id: str):
        return render_service.cancel_render_job(job_id)

    return router
