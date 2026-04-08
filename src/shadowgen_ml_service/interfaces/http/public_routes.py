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

    return router
