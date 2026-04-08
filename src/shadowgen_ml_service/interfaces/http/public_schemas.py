from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SourcePayload(BaseModel):
    mime_type: Literal["image/jpeg", "image/png", "image/webp"]
    image_base64: str = Field(min_length=8)


class PreprocessPayload(BaseModel):
    padding_px: int = Field(default=100, ge=0, le=4096)


class ShadowPayload(BaseModel):
    angle_deg: float = Field(ge=0, le=360)
    elevation_deg: float = Field(ge=0, le=90)
    softness: float = Field(ge=0, le=1)
    opacity: float = Field(ge=0, le=1)
    reflection: float = Field(ge=0, le=1)


class BackgroundPayload(BaseModel):
    mode: Literal["solid"]
    color_hex: str

    @field_validator("color_hex")
    @classmethod
    def validate_hex(cls, value: str) -> str:
        if len(value) != 7 or not value.startswith("#"):
            raise ValueError("background.color_hex must match #RRGGBB")
        int(value[1:], 16)
        return value.upper()


class OutputPayload(BaseModel):
    format: Literal["png", "webp"]
    width: int | None = Field(default=None, ge=1, le=8192)
    height: int | None = Field(default=None, ge=1, le=8192)
    return_debug: bool


class RenderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str | None = Field(default=None, max_length=200)
    pipeline_version: str
    source: SourcePayload
    preprocess: PreprocessPayload = Field(default_factory=PreprocessPayload)
    shadow: ShadowPayload
    background: BackgroundPayload
    output: OutputPayload

    @model_validator(mode="after")
    def validate_pipeline(self) -> "RenderRequest":
        if not self.pipeline_version.strip():
            raise ValueError("pipeline_version must be a non-empty string")
        return self


class ArtifactResponse(BaseModel):
    name: str
    kind: Literal["final", "debug"]
    mime_type: str
    image_base64: str


class MetricsResponse(BaseModel):
    total_ms: int
    decode_ms: int | None = None
    geometry_ms: int | None = None
    detection_ms: int | None = None
    segmentation_ms: int | None = None
    foreground_refinement_ms: int | None = None
    depth_ms: int | None = None
    normals_ms: int | None = None
    shadow_ms: int | None = None
    composition_ms: int | None = None
    encode_ms: int | None = None
    cache_ms: int | None = None


class ModelInfoResponse(BaseModel):
    service_version: str
    model_version: str


class RenderResponse(BaseModel):
    request_id: str | None = None
    artifacts: list[ArtifactResponse]
    metrics: MetricsResponse
    warnings: list[str]
    model_info: ModelInfoResponse


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, str] | None = None
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class ComponentCapabilities(BaseModel):
    name: str
    implementation: str
    model_name: str
    model_version: str
    available: bool
    using_mock: bool
    detail: str | None = None


class CapabilitiesResponse(BaseModel):
    service_version: str
    model_version: str
    supported_input_mime_types: tuple[str, ...]
    supported_output_formats: tuple[str, ...]
    supports_debug_artifacts: bool
    max_image_bytes: int
    active_backend_mode: str
    degraded: bool
    components: list[ComponentCapabilities]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service_version: str
    active_backend_mode: str
