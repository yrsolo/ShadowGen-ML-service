from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from shadowgen_ml_service.interfaces.http.public_schemas import RenderRequest


class StageModesPayload(BaseModel):
    detector: Literal["mock", "real"] = "mock"
    geometry_estimator: Literal["mock", "real"] = "mock"
    segmenter: Literal["mock", "real"] = "mock"
    foreground_refiner: Literal["mock", "real"] = "real"
    depth_estimator: Literal["mock", "real"] = "mock"
    normal_estimator: Literal["mock", "real"] = "real"
    shadow_generator: Literal["mock", "v1-gan", "v2-diff"] = "v1-gan"
    composer: Literal["mock", "real"] = "real"


class StageBackendKindsPayload(BaseModel):
    detector: Literal["mock", "local", "triton"] = "local"
    geometry_estimator: Literal["mock", "local"] = "local"
    segmenter: Literal["mock", "local", "triton"] = "local"
    foreground_refiner: Literal["mock", "local"] = "local"
    depth_estimator: Literal["mock", "local", "triton"] = "local"
    normal_estimator: Literal["mock", "local", "triton"] = "local"
    shadow_generator: Literal["mock", "local", "triton"] = "local"
    composer: Literal["mock", "local"] = "local"


class StageVariantsPayload(BaseModel):
    detector: str = "grounding-dino"
    geometry_estimator: str = "geocalib"
    segmenter: str = "birefnet"
    foreground_refiner: str = "fast-foreground-estimation"
    depth_estimator: str = "depth-anything-v2-small"
    normal_estimator: str = "stable-normal"
    shadow_generator: Literal["mock", "v1-gan", "v2-diff"] = "v1-gan"
    composer: str = "python-composer"


class PipelineDebugRequest(BaseModel):
    render_request: RenderRequest
    stage_modes: StageModesPayload = Field(default_factory=StageModesPayload)
    stage_backend_kinds: StageBackendKindsPayload = Field(default_factory=StageBackendKindsPayload)
    stage_variants: StageVariantsPayload = Field(default_factory=StageVariantsPayload)


class StagePreviewResponse(BaseModel):
    name: str
    mime_type: str
    image_base64: str


class StageExecutionResponse(BaseModel):
    stage_key: str
    title: str
    description: str
    status: Literal["completed", "failed", "skipped"]
    requested_mode: str
    actual_mode: str
    requested_backend_kind: str = "mock"
    actual_backend_kind: str = "internal"
    model_variant: str = "default"
    model_name: str | None = None
    model_version: str | None = None
    device: str | None = None
    endpoint: str | None = None
    cache_status: str | None = None
    fallback_reason: str | None = None
    elapsed_ms: int | None = None
    error: str | None = None
    details: dict[str, str | int | float | bool] | None = None
    previews: list[StagePreviewResponse] = Field(default_factory=list)


class PipelineDebugResponse(BaseModel):
    request_id: str | None = None
    stages: list[StageExecutionResponse]
    warnings: list[str] = Field(default_factory=list)
