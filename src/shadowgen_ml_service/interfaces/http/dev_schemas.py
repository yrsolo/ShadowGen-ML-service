from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from shadowgen_ml_service.interfaces.http.public_schemas import RenderRequest


class StageModesPayload(BaseModel):
    detector: Literal["mock", "real"] = "mock"
    geometry_estimator: Literal["mock", "real"] = "mock"
    segmenter: Literal["mock", "real"] = "mock"
    depth_estimator: Literal["mock", "real"] = "mock"
    normal_estimator: Literal["mock", "real"] = "real"
    shadow_generator: Literal["mock", "real"] = "real"
    composer: Literal["mock", "real"] = "real"


class PipelineDebugRequest(BaseModel):
    render_request: RenderRequest
    stage_modes: StageModesPayload = Field(default_factory=StageModesPayload)


class StagePreviewResponse(BaseModel):
    name: str
    mime_type: str
    image_base64: str


class StageExecutionResponse(BaseModel):
    stage_key: str
    title: str
    description: str
    status: Literal["completed", "failed", "skipped"]
    requested_mode: Literal["mock", "real"]
    actual_mode: str
    elapsed_ms: int | None = None
    error: str | None = None
    details: dict[str, str | int | float | bool] | None = None
    previews: list[StagePreviewResponse] = Field(default_factory=list)


class PipelineDebugResponse(BaseModel):
    request_id: str | None = None
    stages: list[StageExecutionResponse]
    warnings: list[str] = Field(default_factory=list)
