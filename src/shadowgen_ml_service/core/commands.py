from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ShadowSpec:
    angle_deg: float
    elevation_deg: float
    softness: float
    opacity: float
    reflection: float


@dataclass(frozen=True)
class BackgroundSpec:
    mode: str
    color_hex: str


@dataclass(frozen=True)
class OutputSpec:
    format: str
    width: int | None
    height: int | None
    return_debug: bool


@dataclass(frozen=True)
class SourceImage:
    mime_type: str
    image_base64: str


@dataclass(frozen=True)
class RenderCommand:
    request_id: str | None
    pipeline_version: str
    source: SourceImage
    padding_px: int
    shadow: ShadowSpec
    background: BackgroundSpec
    output: OutputSpec


@dataclass(frozen=True)
class DebugPipelineCommand:
    render: RenderCommand
    stage_modes: dict[str, str] = field(default_factory=dict)
