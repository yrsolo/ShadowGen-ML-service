from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from PIL import Image

from shadowgen_ml_service.core.commands import RenderCommand
from shadowgen_ml_service.core.models import (
    CapabilitiesInfo,
    CompositionResult,
    DepthResult,
    DetectionResult,
    EncodedArtifact,
    ForegroundRefinementResult,
    GeometryResult,
    HealthStatus,
    NormalResult,
    PreprocessSnapshot,
    SegmentationResult,
    ShadowResult,
)


@dataclass(frozen=True)
class StageDefinition:
    key: str
    title: str
    description: str


@dataclass
class MetricsCollector:
    started_at: float = field(default_factory=perf_counter)
    values: dict[str, int] = field(default_factory=dict)

    def set(self, key: str, value: int) -> None:
        self.values[key] = value

    def measure_from(self, key: str, started_at: float) -> int:
        elapsed = int((perf_counter() - started_at) * 1000)
        self.values[key] = elapsed
        return elapsed

    def total(self) -> int:
        total_ms = int((perf_counter() - self.started_at) * 1000)
        self.values["total_ms"] = total_ms
        return total_ms


@dataclass
class PipelineContext:
    command: RenderCommand
    warnings: list[str] = field(default_factory=list)
    metrics: MetricsCollector = field(default_factory=MetricsCollector)
    raw_bytes: bytes | None = None
    source_rgba: Image.Image | None = None
    cache_key: str | None = None
    preprocess_snapshot: PreprocessSnapshot | None = None
    working_crop: Image.Image | None = None
    pre_refinement_cutout: Image.Image | None = None
    detection: DetectionResult | None = None
    geometry: GeometryResult | None = None
    segmentation: SegmentationResult | None = None
    foreground_refinement: ForegroundRefinementResult | None = None
    depth: DepthResult | None = None
    normals: NormalResult | None = None
    shadow: ShadowResult | None = None
    composition: CompositionResult | None = None


@dataclass(frozen=True)
class StageBackendSelection:
    requested_mode: str
    actual_mode: str
    unavailable_message: str | None = None


@dataclass
class StageExecution:
    stage_key: str
    title: str
    description: str
    status: str
    requested_mode: str
    actual_mode: str
    elapsed_ms: int | None = None
    error: str | None = None
    details: dict[str, str | int | float | bool] | None = None
    previews: dict[str, Image.Image] = field(default_factory=dict)


@dataclass
class RenderOutcome:
    request_id: str | None
    artifacts: list[EncodedArtifact]
    metrics: dict[str, int]
    warnings: list[str]
    service_version: str
    model_version: str


@dataclass
class DebugPipelineOutcome:
    request_id: str | None
    stages: list[StageExecution]
    warnings: list[str]


@dataclass(frozen=True)
class HealthOutcome:
    payload: HealthStatus


@dataclass(frozen=True)
class CapabilitiesOutcome:
    payload: CapabilitiesInfo
