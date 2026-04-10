from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal

from PIL import Image

from shadowgen_ml_service.core.assets import RasterAsset
from shadowgen_ml_service.core.commands import RenderCommand
from shadowgen_ml_service.core.models import (
    BackendKind,
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
    StageBackendDescriptor,
    StageBackendId,
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
    pre_refinement_cutout: RasterAsset | None = None
    detection: DetectionResult | None = None
    geometry: GeometryResult | None = None
    segmentation: SegmentationResult | None = None
    foreground_refinement: ForegroundRefinementResult | None = None
    depth: DepthResult | None = None
    normals: NormalResult | None = None
    shadow: ShadowResult | None = None
    composition: CompositionResult | None = None


@dataclass(frozen=True)
class ExecutionSelection:
    stage_key: str
    backend_id: StageBackendId | None
    requested_backend_kind: BackendKind
    actual_backend_kind: BackendKind | Literal["unavailable", "internal"]
    requested_variant: str = "default"
    actual_variant: str = "default"
    actual_mode: str = "internal"
    unavailable_message: str | None = None
    fallback_reason: str | None = None
    descriptor: StageBackendDescriptor | None = None

    @property
    def requested_mode(self) -> str:
        return self.requested_backend_kind


@dataclass
class StageExecution:
    stage_key: str
    title: str
    description: str
    status: str
    requested_mode: str
    actual_mode: str
    requested_backend_kind: BackendKind = "mock"
    actual_backend_kind: BackendKind | Literal["unavailable", "internal"] = "internal"
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
    previews: dict[str, RasterAsset] = field(default_factory=dict)


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


@dataclass(frozen=True)
class AsyncRenderJobRecord:
    job_id: str
    request_id: str | None
    status: str
    created_at: str
    updated_at: str
    error: str | None = None
    render_outcome: RenderOutcome | None = None
