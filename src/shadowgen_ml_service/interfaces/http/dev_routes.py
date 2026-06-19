from __future__ import annotations

import os
import threading
import time

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from shadowgen_ml_service.interfaces.dev.playground import render_playground_html
from shadowgen_ml_service.interfaces.http.dev_schemas import PipelineDebugRequest


def _schedule_process_shutdown() -> dict[str, object]:
    pid = os.getpid()

    def stop_process() -> None:
        time.sleep(0.35)
        os._exit(0)

    threading.Thread(target=stop_process, name="shadowgen-dev-shutdown", daemon=True).start()
    return {
        "status": "terminating",
        "pid": pid,
        "message": "Current ShadowGen ML Service process is terminating.",
    }


def build_dev_router(render_service, *, shutdown_enabled: bool) -> APIRouter:
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
    async def run_all_pipeline(payload: PipelineDebugRequest):
        return render_service.run_debug_pipeline(payload)

    @router.post("/v1/dev/pipeline/run-stage/{stage_key}")
    async def run_stage_pipeline(stage_key: str, payload: PipelineDebugRequest):
        return render_service.run_debug_pipeline(payload, stop_after=stage_key)

    if shutdown_enabled:
        @router.post("/v1/dev/service/shutdown")
        async def shutdown_service(request: Request):
            handler = getattr(request.app.state, "dev_shutdown_handler", None)
            if handler is not None:
                return handler()
            return _schedule_process_shutdown()

    return router
